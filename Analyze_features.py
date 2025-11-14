#ターミナルに以下を打つと、labelごとに解析できる
# cd C:\path\to\your\project
# python Analyze_Feature_Correlation.py --labelled-dir Labelled_data --by-label --min-samples 50

# Analyze_Feature_Correlation.py
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= 出力設定（ここを書き換えるだけで出力を制御） =================
OUTPUT_OPTS = {
    "save_corr": True,         # 相関行列 CSV（*_corr_*.csv）
    "save_pairs_all": False,   # 全ペア一覧 CSV（*_pairs_all_*.csv）
    "save_pairs_thr": False,   # 閾値超えペア CSV（*_pairs_thr*.csv）
    "save_pairs_top": False,   # 上位TopKペア CSV（*_pairs_top*.csv）
    "save_heatmap": False,      # 相関ヒートマップ PNG
    "save_describe": False,    # describe.csv（基本統計）
    "do_by_label": True,       # ラベル別出力を行う
    "methods": ["pearson"],    # 相関係数の種類: "pearson", "spearman", "kendall"
    "topk": 200,               # 上位ペア数（save_pairs_top=True のときのみ使用）
    "thr": 0.95,               # 閾値ペア抽出のしきい値（save_pairs_thr=True のときのみ）
}
# ======================================================================


# ==== Build_RF_dataset からウィンドウ生成と特徴量計算を再利用 ====
try:
    from Build_RF_dataset import (
        FEATURE_NAMES,
        STAT_FUNCS,
        load_set_series,
        label_at_window_median,
        compute_window_features,
        sliding_windows_by_count,
        DEFAULT_STATS as BRD_DEFAULT_STATS,
    )
except Exception:
    FEATURE_NAMES = []  # type: ignore
    STAT_FUNCS = {}    # type: ignore
    load_set_series = None  # type: ignore
    label_at_window_median = None  # type: ignore
    compute_window_features = None  # type: ignore
    sliding_windows_by_count = None  # type: ignore
    BRD_DEFAULT_STATS = []  # type: ignore

# ==== 追加: SSC & 絶対値総和の実装 & 登録 ====
def _ssc_1d(x: np.ndarray, threshold: float = 0.0) -> int:
    if x.size < 3:
        return 0
    if np.isnan(x).any():
        x = np.nan_to_num(x, nan=0.0)
    a = x[1:-1] - x[:-2]
    b = x[1:-1] - x[2:]
    return int(np.sum((a * b) > threshold))

def ssc_stat(arr: np.ndarray, *args, **kwargs) -> np.ndarray:
    arr = np.asarray(arr)
    thr = float(kwargs.get("threshold", 0.0))
    if arr.ndim == 1:
        return np.array(_ssc_1d(arr, threshold=thr), dtype=float)
    elif arr.ndim == 2:
        return np.array([_ssc_1d(arr[:, c], threshold=thr) for c in range(arr.shape[1])], dtype=float)
    else:
        raise ValueError("ssc_stat: arr must be 1D or 2D")

def abs_sum_stat(arr: np.ndarray, *args, **kwargs) -> np.ndarray:
    arr = np.asarray(arr)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)
    if arr.ndim == 1:
        return np.array(np.sum(np.abs(arr)), dtype=float)
    elif arr.ndim == 2:
        return np.sum(np.abs(arr), axis=0).astype(float)
    else:
        raise ValueError("abs_sum_stat: arr must be 1D or 2D")

try:
    import Build_RF_dataset as BRD
    if hasattr(BRD, "STAT_FUNCS") and isinstance(BRD.STAT_FUNCS, dict):
        BRD.STAT_FUNCS.update({"ssc": ssc_stat, "abs_sum": abs_sum_stat})
    if isinstance(STAT_FUNCS, dict):
        STAT_FUNCS.update({"ssc": ssc_stat, "abs_sum": abs_sum_stat})
except Exception:
    pass
# ==== 追加ここまで ====


# ================= 既定ユーザー設定 =================
USER_SELECTED_STATS: List[str] = ["mean", "std", "max", "rms", "min", "abs_sum"]
USER_WINDOW_MS: int = 250
USER_HOP_MS: int = 25
USER_FS_HZ: Optional[float] = 200.0
USER_OUT_DIR: Path = Path("feature_corr")
# ====================================================

@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    columns: List[str]
    labels_seq: List[str]  # 各行のラベル（by-label相関用）

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _infer_used_cols_by_feature_names(
    feature_names: Sequence[str], stats: Sequence[str]
) -> List[str]:
    available_stats = set(STAT_FUNCS.keys()) if isinstance(STAT_FUNCS, dict) else set(stats)
    chosen_stats = [st for st in stats if st in available_stats]
    unknown = [st for st in stats if st not in available_stats]
    if unknown:
        print(f"[WARN] 未登録の統計をスキップ: {unknown}", flush=True)

    out: List[str] = []
    for ch in feature_names:
        for st in chosen_stats:
            out.append(f"{ch}_{st}")
    return out

def build_features_from_txt(
    txt_path: Path,
    used_cols: Sequence[str],
    window_ms: int,
    hop_ms: int,
    fs_hz: Optional[float],
) -> Dataset:
    if load_set_series is None or compute_window_features is None or sliding_windows_by_count is None:
        raise SystemExit("Build_RF_dataset.py の関数をインポートできませんでした。配置を確認してください。")

    series = load_set_series(txt_path)

    # サンプリング周波数の決定
    if fs_hz is None:
        if series.t_s.size >= 2:
            dt = float(np.median(np.diff(series.t_s)))
            fs_hz = (1.0 / dt) if dt > 0 else 100.0
        else:
            fs_hz = 100.0

    win_samp = max(int(round((window_ms / 1000.0) * fs_hz)), 2)
    hop_samp = max(int(round((hop_ms / 1000.0) * fs_hz)), 1)

    # used_cols の末尾一致から stat_names を逆推定
    if FEATURE_NAMES and STAT_FUNCS:
        known_stats = sorted(STAT_FUNCS.keys(), key=len, reverse=True)
        stat_names: List[str] = []
        for c in used_cols:
            for st in known_stats:
                if c.endswith("_" + st):
                    if st not in stat_names:
                        stat_names.append(st)
                    break
        if not stat_names:
            stat_names = list(BRD_DEFAULT_STATS) if BRD_DEFAULT_STATS else ["mean", "std", "min", "max"]
    else:
        stat_names = ["mean", "std", "min", "max"]

    # ウィンドウ列挙
    windows = sliding_windows_by_count(series.t_s.size, win_samp, hop_samp)

    all_headers: List[str] = []
    for ch in FEATURE_NAMES:
        for st in stat_names:
            all_headers.append(f"{ch}_{st}")

    rows: List[List[float]] = []
    labels_seq: List[str] = []

    for a, b in windows:
        t_sub = series.t_s[a : b + 1]
        d_sub = series.data[a : b + 1, :]
        lbl_sub = series.labels[a : b + 1]
        lbl = label_at_window_median(lbl_sub)
        if lbl is None:
            continue
        feats_all = compute_window_features(d_sub, t_sub, stat_names)
        rows.append(feats_all)
        labels_seq.append(lbl)

    if not rows:
        return Dataset(
            X=np.empty((0, len(used_cols))),
            y=np.empty((0,), dtype=object),
            columns=list(used_cols),
            labels_seq=[],
        )

    df_all = pd.DataFrame(rows, columns=all_headers)
    missing = [c for c in used_cols if c not in df_all.columns]
    if missing:
        raise SystemExit(f"{txt_path} の特徴量に不足があります。欠落列: {missing}")

    X = df_all[used_cols].to_numpy(dtype=float, copy=False)
    y = pd.Series(labels_seq).to_numpy()
    return Dataset(X=X, y=y, columns=list(used_cols), labels_seq=labels_seq)

_SET_RE = re.compile(r"set-(\d+)")
def _extract_set_id(name: str) -> Optional[str]:
    m = _SET_RE.search(name)
    return m.group(1) if m else None

def find_sets(base_dir: Path, subjects: Sequence[str]) -> Dict[Tuple[str, str, str], Path]:
    """
    Labelled_data 配下の *.txt を走査し、キー = (subject, set_id, kind[ramp/stair/other]) で Path を返す。
    """
    out: Dict[Tuple[str, str, str], Path] = {}
    for p in base_dir.rglob("*.txt"):
        low = p.name.lower()
        subj_hit = None
        for subj in subjects:
            if subj.lower() in low:
                subj_hit = subj
                break
        if subj_hit is None:
            continue
        sid = _extract_set_id(low)
        if not sid:
            continue
        kind = "ramp" if "ramp" in low else ("stair" if "stair" in low else "other")
        out[(subj_hit, sid, kind)] = p
    return out

def save_corr_outputs(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    methods: Sequence[str],
    topk: int,
    thr: float,
    save_corr: bool,
    save_pairs_all: bool,
    save_pairs_thr: bool,
    save_pairs_top: bool,
    save_heatmap: bool,
) -> None:
    _ensure_dir(out_dir)

    for method in methods:
        # 相関行列を計算
        if method == "pearson":
            corr = df.corr(method="pearson")
        elif method == "spearman":
            corr = df.corr(method="spearman")
        elif method == "kendall":
            corr = df.corr(method="kendall")
        else:
            print(f"[WARN] 未対応 method をスキップ: {method}")
            continue

        cols = corr.columns.tolist()

        # 相関行列 CSV
        if save_corr:
            corr.to_csv(out_dir / f"{prefix}_corr_{method}.csv", encoding="utf-8")

        # ペア一覧（必要なものだけ）
        pairs = []
        if save_pairs_all or save_pairs_thr or (save_pairs_top and topk > 0):
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    val = corr.iat[i, j]
                    if pd.isna(val):
                        continue
                    pairs.append((cols[i], cols[j], float(val), float(abs(val))))
            pairs.sort(key=lambda t: t[3], reverse=True)

        if save_pairs_all:
            pd.DataFrame(pairs, columns=["feat_i", "feat_j", "corr", "abs_corr"]).to_csv(
                out_dir / f"{prefix}_pairs_all_{method}.csv", index=False, encoding="utf-8"
            )
        if save_pairs_thr:
            pairs_thr = [p for p in pairs if p[3] >= thr]
            pd.DataFrame(pairs_thr, columns=["feat_i", "feat_j", "corr", "abs_corr"]).to_csv(
                out_dir / f"{prefix}_pairs_thr{thr:g}_{method}.csv", index=False, encoding="utf-8"
            )
        if save_pairs_top and topk > 0:
            top_pairs = pairs[:topk]
            pd.DataFrame(top_pairs, columns=["feat_i", "feat_j", "corr", "abs_corr"]).to_csv(
                out_dir / f"{prefix}_pairs_top{topk}_{method}.csv", index=False, encoding="utf-8"
            )

        # ヒートマップ PNG
        if save_heatmap:
            fig = plt.figure(figsize=(max(6, 0.25 * len(cols)), max(5, 0.25 * len(cols))))
            ax = fig.add_subplot(111)
            im = ax.imshow(corr.values, interpolation="nearest", aspect="auto")
            ax.set_xticks(range(len(cols)))
            ax.set_yticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=90, fontsize=6)
            ax.set_yticklabels(cols, fontsize=6)
            ax.set_title(f"Correlation ({method})")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out_dir / f"{prefix}_corr_{method}.png", dpi=200)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="各 set（ramp/stair/other 含む）ごとに特徴量を生成し、特徴量間相関を分析・保存（CSV/PNG）"
    )
    parser.add_argument("--labelled-dir", type=Path, default=Path("Labelled_data"))
    parser.add_argument("--subjects", type=str, default="kanoga,kaneishi")
    parser.add_argument("--use-stats", type=str, default=",".join(USER_SELECTED_STATS),
                        help="使用する統計のカンマ区切り（例: mean,std,max,rms,min,ssc,abs_sum）")
    parser.add_argument("--window-ms", type=int, default=USER_WINDOW_MS)
    parser.add_argument("--hop-ms", type=int, default=USER_HOP_MS)
    parser.add_argument("--fs-hz", type=float, default=(USER_FS_HZ if USER_FS_HZ is not None else np.nan))
    parser.add_argument("--methods", type=str, default="pearson,spearman",
                        help="相関係数の種類（カンマ区切り）: pearson,spearman,kendall")
    parser.add_argument("--by-label", action="store_true",
                        help="ラベル別に相関を計算して保存（各ラベルで行数が十分ある場合）")
    parser.add_argument("--min-samples", type=int, default=30,
                        help="相関計算に必要な最小ウィンドウ数（全体/ラベル別）")
    parser.add_argument("--topk", type=int, default=200, help="上位相関ペアの保存数（絶対値順）")
    parser.add_argument("--thr", type=float, default=0.95, help="高相関ペア抽出の絶対値しきい値")
    parser.add_argument("--out-dir", type=Path, default=USER_OUT_DIR)

    args = parser.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    if not subjects:
        raise SystemExit("--subjects が空です")

    fs_hz_arg = None if (isinstance(args.fs_hz, float) and np.isnan(args.fs_hz)) else float(args.fs_hz)

    # 列名（全 set で共通）
    if not FEATURE_NAMES:
        raise SystemExit("Build_RF_dataset.FEATURE_NAMES が必要です（センサ次元のチャンネル名）")
    selected_stats = [s.strip() for s in args.use_stats.split(",") if s.strip()]
    used_cols = _infer_used_cols_by_feature_names(FEATURE_NAMES, selected_stats)
    print(f"[INFO] 使用列（{len(used_cols)}）例: {used_cols[:8]} ...", flush=True)

    # 入力走査
    table = find_sets(args.labelled_dir, subjects)
    if not table:
        raise SystemExit(f"[ERR] *.txt が見つかりません: {args.labelled_dir}")

    # 各ファイル（= 各 set & kind）を単独で処理
    for (subj, set_id, kind), path in sorted(table.items(), key=lambda kv: (kv[0][0], int(kv[0][1]), kv[0][2])):
        print(f"[RUN] subject={subj}, set-{set_id}, kind={kind} | {path.name}", flush=True)

        # 特徴量生成（この set のウィンドウ全体）
        ds = build_features_from_txt(path, used_cols, args.window_ms, args.hop_ms, fs_hz_arg)
        if ds.X.shape[0] < args.min_samples:
            print(f"  [SKIP] 窓数が不足 ({ds.X.shape[0]} < {args.min_samples}) -> 相関計算をスキップ", flush=True)
            continue

        df = pd.DataFrame(ds.X, columns=ds.columns)

        # 出力先
        out_root = args.out_dir / subj / f"set-{set_id}" / kind
        _ensure_dir(out_root)

        # 1) 全体相関
        save_corr_outputs(
            df=df,
            out_dir=out_root,
            prefix=f"{subj}_set-{set_id}_{kind}_ALL",
            methods=OUTPUT_OPTS["methods"],
            topk=int(OUTPUT_OPTS["topk"]),
            thr=float(OUTPUT_OPTS["thr"]),
            save_corr=bool(OUTPUT_OPTS["save_corr"]),
            save_pairs_all=bool(OUTPUT_OPTS["save_pairs_all"]),
            save_pairs_thr=bool(OUTPUT_OPTS["save_pairs_thr"]),
            save_pairs_top=bool(OUTPUT_OPTS["save_pairs_top"]),
            save_heatmap=bool(OUTPUT_OPTS["save_heatmap"]),
        )

        # 2) ラベル別相関（必要なら）
        if OUTPUT_OPTS["do_by_label"]:   # ← args.by_label の代わりに設定で制御
            labels = pd.Series(ds.labels_seq, dtype=str)
            for lbl, idx in labels.groupby(labels).groups.items():
                if len(idx) < args.min_samples:
                    print(f"  [BY-LABEL SKIP] label={lbl}: {len(idx)} < {args.min_samples}", flush=True)
                    continue
                dfl = df.iloc[list(idx)].copy()
                save_corr_outputs(
                    df=dfl,
                    out_dir=out_root / "by_label",
                    prefix=f"{subj}_set-{set_id}_{kind}_LABEL-{lbl}",
                    methods=OUTPUT_OPTS["methods"],
                    topk=int(OUTPUT_OPTS["topk"]),
                    thr=float(OUTPUT_OPTS["thr"]),
                    save_corr=bool(OUTPUT_OPTS["save_corr"]),
                    save_pairs_all=bool(OUTPUT_OPTS["save_pairs_all"]),
                    save_pairs_thr=bool(OUTPUT_OPTS["save_pairs_thr"]),
                    save_pairs_top=bool(OUTPUT_OPTS["save_pairs_top"]),
                    save_heatmap=bool(OUTPUT_OPTS["save_heatmap"]),
                )

        # 3) describe の保存を設定で制御
        if OUTPUT_OPTS["save_describe"]:
            desc = df.describe(include="all")
            desc.to_csv(out_root / f"{subj}_set-{set_id}_{kind}_describe.csv", encoding="utf-8")


if __name__ == "__main__":
    main()
