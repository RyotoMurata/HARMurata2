from __future__ import annotations

import argparse
import fnmatch
import pickle
import re
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ================= ユーザー設定（先頭でモデル切替！） =================
# モデル切替: "rf" | "lda" | "qda"
USER_MODEL: str = "qda"

# LDA 用
USER_LDA_SOLVER: str = "svd"        # "svd" | "lsqr" | "eigen"
USER_LDA_SHRINKAGE: Optional[str] = None  # None | "auto"

# QDA 用
USER_QDA_REG: float = 0.0           # 0.0〜1.0 推奨0〜0.1

# 特徴統計の選択
USER_SELECTED_STATS: List[str] = ["mean", "std", "max", "rms", "min", "abs_sum"]

# RF の主なハイパラ
USER_RANDOM_STATE: int = 0
USER_N_ESTIMATORS: int = 50
USER_MAX_DEPTH: Optional[int] = 10  # None で無制限
USER_OUT_DIR: Path = Path("models") / "multi_eval"
USER_WINDOW_MS: int = 250
USER_HOP_MS: int = 25
USER_FS_HZ: Optional[float] = 200.0  # None で自動推定
USER_MAX_FEATURES = 8
USER_MIN_SAMPLES_LEAF = 2
USER_MIN_SAMPLES_SPLIT = 4
USER_CCP_ALPHA = 1e-4
USER_MAX_LEAF_NODES = 512
USER_BOOTSTRAP = True
USER_MAX_SAMPLES = 0.8   # None なら全サンプル
USER_CLASS_WEIGHT = "balanced_subsample"  # or None
USER_CRITERION = "gini"

# 手動指定の特徴量（空なら自動で FEATURE_NAMES × USER_SELECTED_STATS）
MANUAL_FEATURES: List[str] = []
MANUAL_FEATURES_FILE: Optional[Path] = None
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

except Exception as e:
    FEATURE_NAMES = []  # type: ignore
    STAT_FUNCS = {}  # type: ignore
    load_set_series = None  # type: ignore
    label_at_window_median = None  # type: ignore
    compute_window_features = None  # type: ignore
    sliding_windows_by_count = None  # type: ignore
    BRD_DEFAULT_STATS = []  # type: ignore

# ==== 追加: SSC と 絶対値総和 の実装 & 登録 ====
def _ssc_1d(x: np.ndarray, threshold: float = 0.0) -> int:
    """1D配列のSSCを数える（EMGでよく使う定義）。thresholdで微小ノイズを抑制。"""
    if x.size < 3:
        return 0
    if np.isnan(x).any():
        x = np.nan_to_num(x, nan=0.0)
    a = x[1:-1] - x[:-2]
    b = x[1:-1] - x[2:]
    return int(np.sum((a * b) > threshold))

def ssc_stat(arr: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Window内データ arr の SSC をチャネル毎に返す統計関数。
    arr: shape (N,) or (N, C)
    返り値: shape (C,) もしくは ()（1Dならスカラー）
    """
    arr = np.asarray(arr)
    thr = float(kwargs.get("threshold", 0.0))
    if arr.ndim == 1:
        return np.array(_ssc_1d(arr, threshold=thr), dtype=float)
    elif arr.ndim == 2:
        return np.array([_ssc_1d(arr[:, c], threshold=thr) for c in range(arr.shape[1])], dtype=float)
    else:
        raise ValueError("ssc_stat: arr must be 1D or 2D")

def abs_sum_stat(arr: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Window内データ arr の 絶対値総和（Sum of Absolute Values）をチャネル毎に返す。
    """
    arr = np.asarray(arr)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)
    if arr.ndim == 1:
        return np.array(np.sum(np.abs(arr)), dtype=float)
    elif arr.ndim == 2:
        return np.sum(np.abs(arr), axis=0).astype(float)
    else:
        raise ValueError("abs_sum_stat: arr must be 1D or 2D")

# Build_RF_dataset 側の STAT_FUNCS をモンキーパッチして登録
try:
    import Build_RF_dataset as BRD  # モジュールそのものを取得
    if hasattr(BRD, "STAT_FUNCS") and isinstance(BRD.STAT_FUNCS, dict):
        BRD.STAT_FUNCS.update({
            "ssc": ssc_stat,
            "abs_sum": abs_sum_stat,
        })
    if isinstance(STAT_FUNCS, dict):
        STAT_FUNCS.update({
            "ssc": ssc_stat,
            "abs_sum": abs_sum_stat,
        })
except Exception:
    pass
# ==== 追加ここまで ====


@dataclass
class Dataset:
    """学習/評価用の配列データを保持"""
    X: np.ndarray
    y: np.ndarray
    columns: List[str]
    t_sec: np.ndarray            # ウィンドウ代表時刻（秒, shape=(N,)）
    src: List[str]               # 各行の由来ファイル名（shape=(N,)相当）


def format_hp(params: Dict[str, object], keys: Sequence[str]) -> str:
    out = []
    for k in keys:
        v = params.get(k, None)
        if isinstance(v, float):
            out.append(f"{k}={v:.6g}")
        else:
            out.append(f"{k}={v}")
    return ", ".join(out)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def model_size_bytes(model) -> int:
    try:
        data = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        return len(data)
    except Exception:
        return -1

def _infer_used_cols_by_feature_names(
    feature_names: Sequence[str], stats: Sequence[str]
) -> List[str]:
    """
    FEATURE_NAMES × stats から列名を合成。
    Build_RF_dataset.STAT_FUNCS に未登録の統計名は自動スキップして警告。
    """
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

def _all_possible_cols(feature_names: Sequence[str],
                       stat_funcs: Dict[str, object],
                       stats_hint: Sequence[str]) -> List[str]:
    if isinstance(stat_funcs, dict) and stat_funcs:
        stats = list(stat_funcs.keys())
    else:
        stats = list(stats_hint)
    out: List[str] = []
    for ch in feature_names:
        for st in stats:
            out.append(f"{ch}_{st}")
    return out

def _read_features_file(path: Path) -> List[str]:
    items: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            items.append(t)
    return items

def _resolve_feature_patterns(candidates: Sequence[str],
                              patterns: Sequence[str]) -> List[str]:
    selected: List[str] = []
    seen = set()
    for pat in patterns:
        if pat in candidates and pat not in seen:
            selected.append(pat); seen.add(pat); continue
        matched = [c for c in candidates if fnmatch.fnmatch(c, pat)]
        for m in matched:
            if m not in seen:
                selected.append(m); seen.add(m)
    return selected


# ========== モデル生成（RF/LDA/QDA 切替） ==========
def build_classifier(
    model_name: str,
    *,
    # RF 用
    n_estimators: int,
    max_depth: Optional[int],
    random_state: int,
    max_features: int,
    min_samples_leaf: int,
    min_samples_split: int,
    ccp_alpha: float,
    max_leaf_nodes: Optional[int],
    bootstrap: bool,
    max_samples: Optional[float],
    class_weight: Optional[str],
    criterion: str,
    # LDA 用
    lda_solver: str = "svd",
    lda_shrinkage: Optional[str] = None,
    # QDA 用
    qda_reg: float = 0.0,
) -> object:
    """
    model_name に応じて学習器を返す。出力・評価コードは従来どおりでOK（.fit/.predictがあれば動く）。
    LDA/QDAはスケーリングの影響を受けやすいので StandardScaler を前段に挟む。
    """
    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            ccp_alpha=ccp_alpha,
            max_leaf_nodes=max_leaf_nodes,
            bootstrap=bootstrap,
            max_samples=max_samples,
            class_weight=class_weight,
            criterion=criterion,
            n_jobs=-1,
            random_state=random_state,
        )
        return clf

    elif model_name == "lda":
        # shrinkageを使う場合は solver を lsqr/eigen に
        if lda_shrinkage is not None and lda_solver == "svd":
            lda_solver = "lsqr"
        lda = LinearDiscriminantAnalysis(solver=lda_solver, shrinkage=lda_shrinkage)
        return make_pipeline(StandardScaler(with_mean=True, with_std=True), lda)

    elif model_name == "qda":
        qda = QuadraticDiscriminantAnalysis(reg_param=float(qda_reg))
        return make_pipeline(StandardScaler(with_mean=True, with_std=True), qda)

    else:
        raise SystemExit(f"[ERR] unknown model_name: {model_name}")


def describe_model_complexity(clf: object) -> Tuple[str, str]:
    """
    出力テキストの 'Model complexity: total_nodes=..., max_tree_depth=...' を壊さないための互換ラッパ。
    RF以外は 'N/A' を返す。
    """
    try:
        if isinstance(clf, RandomForestClassifier):
            total_nodes = sum(t.tree_.node_count for t in clf.estimators_)
            max_tree_depth = max(t.tree_.max_depth for t in clf.estimators_)
            return str(total_nodes), str(max_tree_depth)
        else:
            return "N/A", "N/A"
    except Exception:
        return "N/A", "N/A"


# ========== 特徴量ビルド ==========
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

    if fs_hz is None:
        if series.t_s.size >= 2:
            dt = float(np.median(np.diff(series.t_s)))
            fs_hz = (1.0 / dt) if dt > 0 else 100.0
        else:
            fs_hz = 100.0

    win_samp = max(int(round((window_ms / 1000.0) * fs_hz)), 2)
    hop_samp = max(int(round((hop_ms / 1000.0) * fs_hz)), 1)

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

    windows = sliding_windows_by_count(series.t_s.size, win_samp, hop_samp)
    rows: List[List[float]] = []
    labels: List[str] = []
    t_mids: List[float] = []
    src_names: List[str] = []

    all_headers: List[str] = []
    for ch in FEATURE_NAMES:
        for st in stat_names:
            all_headers.append(f"{ch}_{st}")

    for a, b in windows:
        t_sub = series.t_s[a : b + 1]
        d_sub = series.data[a : b + 1, :]
        lbl_sub = series.labels[a : b + 1]
        lbl = label_at_window_median(lbl_sub)
        if lbl is None:
            continue
        feats_all = compute_window_features(d_sub, t_sub, stat_names)
        rows.append(feats_all)
        labels.append(lbl)
        t_mids.append(float(np.median(t_sub)))
        src_names.append(txt_path.name)

    if not rows:
        return Dataset(
            X=np.empty((0, len(used_cols))),
            y=np.empty((0,), dtype=object),
            columns=list(used_cols),
            t_sec=np.empty((0,), dtype=float),
            src=[],
        )

    df_all = pd.DataFrame(rows, columns=all_headers)

    missing = [c for c in used_cols if c not in df_all.columns]
    if missing:
        raise SystemExit(f"{txt_path} の特徴量に不足があります。欠落列: {missing}")

    X = df_all[used_cols].to_numpy(dtype=float, copy=False)
    y = pd.Series(labels).to_numpy()
    return Dataset(
        X=X, y=y,
        columns=list(used_cols),
        t_sec=np.asarray(t_mids, dtype=float),
        src=src_names,
    )

def concat_datasets(dsets: Iterable[Dataset], used_cols: Sequence[str]) -> Dataset:
    xs, ys = [], []
    ts, ss = [], []
    for ds in dsets:
        if list(ds.columns) != list(used_cols):
            raise SystemExit("列順が一致しません。used_cols に揃えてください。")
        xs.append(ds.X)
        ys.append(ds.y)
        ts.append(ds.t_sec)
        ss.extend(ds.src)
    X = np.concatenate(xs, axis=0) if xs else np.empty((0, len(used_cols)), dtype=float)
    y = np.concatenate(ys, axis=0) if ys else np.empty((0,), dtype=object)
    t = np.concatenate(ts, axis=0) if ts else np.empty((0,), dtype=float)
    return Dataset(X=X, y=y, columns=list(used_cols), t_sec=t, src=ss)

def evaluate_classifier(
    clf: object,
    X: np.ndarray,
    y: np.ndarray,
    labels_order: Optional[Sequence[str]] = None,
) -> Tuple[float, float, str, np.ndarray, List[str]]:
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    report = classification_report(y, y_pred, labels=labels_order, zero_division=0)
    if labels_order is None:
        labels_list = sorted(list({*map(str, y)}))
    else:
        labels_list = list(labels_order)
    cm = confusion_matrix(y, y_pred, labels=labels_list)
    return acc, macro_f1, report, cm, labels_list


# ------------------ ここから ユーティリティ ------------------
_SET_RE = re.compile(r"set-(\d+)")

def _extract_set_id(name: str) -> Optional[str]:
    m = _SET_RE.search(name)
    return m.group(1) if m else None

def find_pairs_by_subject(
    base_dir: Path,
    subject_keys: Sequence[str],
) -> Dict[str, List[str]]:
    """
    Labelled_data 配下から、subject ごとに ramp/stair のペアが揃う set-XXXX を列挙。
    戻り値: { "kanoga": ["0001","0002",...], ... }
    """
    subject_to_files: Dict[str, List[Path]] = {subj: [] for subj in subject_keys}
    for p in base_dir.rglob("*.txt"):
        low = p.name.lower()
        for subj in subject_keys:
            if subj.lower() in low:
                subject_to_files[subj].append(p)
                break

    subject_to_sets: Dict[str, List[str]] = {}
    for subj, files in subject_to_files.items():
        by_set: Dict[str, Dict[str, Path]] = {}
        for fp in files:
            name = fp.name.lower()
            sid = _extract_set_id(name)
            if sid is None:
                continue
            kind = "ramp" if "ramp" in name else ("stair" if "stair" in name else None)
            if kind is None:
                continue
            by_set.setdefault(sid, {})
            by_set[sid][kind] = fp

        ok_sets = [sid for sid, d in by_set.items() if "ramp" in d and "stair" in d]
        ok_sets.sort(key=lambda s: int(s))
        subject_to_sets[subj] = ok_sets
    return subject_to_sets

def build_pair_dataset(
    base_dir: Path,
    subject: str,
    set_id: str,
    used_cols: Sequence[str],
    window_ms: int,
    hop_ms: int,
    fs_hz: Optional[float],
) -> Dataset:
    """subject の set-XXXX の ramp+stair をまとめて 1 データセットとして特徴量生成"""
    ramp = None
    stair = None
    for p in base_dir.rglob(f"*{subject}*set-{set_id}*.txt"):
        low = p.name.lower()
        if "ramp" in low:
            ramp = p
        elif "stair" in low:
            stair = p
    if ramp is None or stair is None:
        raise SystemExit(f"{subject} set-{set_id}: ramp/stair のどちらかが見つかりません")

    ds_r = build_features_from_txt(ramp, used_cols, window_ms, hop_ms, fs_hz)
    ds_s = build_features_from_txt(stair, used_cols, window_ms, hop_ms, fs_hz)
    return concat_datasets([ds_r, ds_s], used_cols)


# ------------------ メイン処理 ------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LOSO（Leave-One-Subject-Out）か被験者内k-foldを選べる評価スクリプト\n"
            "- LOSO: 指定被験者のうち1名をテスト、残り全員で学習→テスト被験者の全setを評価・平均\n"
            "- k-fold: 被験者内でsetをfoldに分けて学習/評価（従来仕様）\n"
            "※ 先頭の USER_MODEL でもモデル切替可能（CLIで上書きも可）"
        )
    )
    parser.add_argument("--labelled-dir", type=Path, default=Path("Labelled_data"), help="ラベル付きTXTのルート")
    parser.add_argument("--subjects", type=str, default="kanoga,kaneishi,murata", help="対象被験者（カンマ区切り）")
    parser.add_argument("--use-first-k", type=int, default=11, help="各被験者で使用する先頭ペア数（k-fold と LOSOの評価上限）")
    parser.add_argument("--cv-mode", type=str, choices=["loso", "kfold"], default="loso", help="評価モード")

    # --- モデル選択（先頭の設定で既定、CLIで上書き可） ---
    parser.add_argument("--model", type=str, choices=["rf", "lda", "qda"], default=USER_MODEL,
                        help="学習モデルを選択（既定: USER_MODEL）")

    # --- LDA用（必要なときだけ使われる） ---
    parser.add_argument("--lda-solver", type=str, default=USER_LDA_SOLVER,
                        choices=["svd", "lsqr", "eigen"],
                        help="LDA solver（shrinkageを使う場合は lsqr/eigen）")
    parser.add_argument("--lda-shrinkage", type=str, default=USER_LDA_SHRINKAGE,
                        choices=[None, "auto"],
                        help="LDA shrinkage（None または 'auto'）")

    # --- QDA用（必要なときだけ使われる） ---
    parser.add_argument("--qda-reg", type=float, default=USER_QDA_REG,
                        help="QDA の正則化 reg_param (0.0〜1.0 推奨は 0.0〜0.1)")

    # --- RF ハイパラ（RF時のみ使用。LDA/QDA時は無視されます） ---
    parser.add_argument("--window-ms", type=int, default=USER_WINDOW_MS)
    parser.add_argument("--hop-ms", type=int, default=USER_HOP_MS)
    parser.add_argument("--fs-hz", type=float, default=USER_FS_HZ if USER_FS_HZ is not None else np.nan)
    parser.add_argument("--n-estimators", type=int, default=USER_N_ESTIMATORS)
    parser.add_argument("--max-depth", type=int, default=USER_MAX_DEPTH if USER_MAX_DEPTH is not None else -1)
    parser.add_argument("--random-state", type=int, default=USER_RANDOM_STATE)
    parser.add_argument("--max-features", type=int, default=USER_MAX_FEATURES)
    parser.add_argument("--min-samples-leaf", type=int, default=USER_MIN_SAMPLES_LEAF)
    parser.add_argument("--min-samples-split", type=int, default=USER_MIN_SAMPLES_SPLIT)
    parser.add_argument("--ccp-alpha", type=float, default=USER_CCP_ALPHA)
    parser.add_argument("--max-leaf-nodes", type=int, default=USER_MAX_LEAF_NODES)
    parser.add_argument("--bootstrap", action="store_true", default=USER_BOOTSTRAP)
    parser.add_argument("--max-samples", type=float, default=USER_MAX_SAMPLES if USER_MAX_SAMPLES is not None else -1.0)
    parser.add_argument("--class-weight", type=str, default=USER_CLASS_WEIGHT)
    parser.add_argument("--criterion", type=str, default=USER_CRITERION)
    parser.add_argument("--out-dir", type=Path, default=USER_OUT_DIR)

    args = parser.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    if not subjects:
        raise SystemExit("--subjects が空です")

    fs_hz_arg = None if (isinstance(args.fs_hz, float) and np.isnan(args.fs_hz)) else float(args.fs_hz)
    max_depth_arg = None if args.max_depth is None or args.max_depth < 0 else int(args.max_depth)
    max_samples_arg = None if (args.max_samples is not None and args.max_samples < 0) else args.max_samples
    max_leaf_nodes_arg = None if (args.max_leaf_nodes is not None and args.max_leaf_nodes <= 0) else args.max_leaf_nodes
    class_weight_arg = None if (isinstance(args.class_weight, str) and args.class_weight.lower() in ["none", "null", ""]) else args.class_weight

    # 出力先（既存命名を維持）
    run_dir = args.out_dir / f"ne{args.n_estimators}_md{('None' if max_depth_arg is None else max_depth_arg)}_rs{args.random_state}_LOSO_qda_cv"
    eval_dir = run_dir / "evals"
    _ensure_dir(run_dir)
    _ensure_dir(eval_dir)
    print(f"[CV] 出力先: {run_dir}（詳細: {eval_dir}）", flush=True)

    if not FEATURE_NAMES:
        raise SystemExit("Build_RF_dataset.FEATURE_NAMES が必要です")

    # 特徴列の決定
    auto_used_cols = _infer_used_cols_by_feature_names(FEATURE_NAMES, USER_SELECTED_STATS)
    manual_patterns: List[str] = list(MANUAL_FEATURES)
    if MANUAL_FEATURES_FILE is not None:
        if not MANUAL_FEATURES_FILE.exists():
            raise SystemExit(f"[ERR] MANUAL_FEATURES_FILE が見つかりません: {MANUAL_FEATURES_FILE}")
        manual_patterns += _read_features_file(MANUAL_FEATURES_FILE)

    if manual_patterns:
        candidates = _all_possible_cols(FEATURE_NAMES, STAT_FUNCS, USER_SELECTED_STATS)
        used_cols = _resolve_feature_patterns(candidates, manual_patterns)
        if not used_cols:
            raise SystemExit(
                "手動指定の特徴名（またはワイルドカード）に一致する列がありませんでした。\n"
                f"  patterns={manual_patterns[:10]} ..."
            )
        not_matched = [p for p in manual_patterns if len(_resolve_feature_patterns(candidates, [p])) == 0]
        if not_matched:
            print(f"[WARN] マッチしなかった指定: {not_matched}", flush=True)
        print(f"[CV] 特徴量列（手動指定）: {len(used_cols)} 列", flush=True)
    else:
        used_cols = auto_used_cols
        print(f"[CV] 特徴量列（自動: FEATURE_NAMES × USER_SELECTED_STATS）: {len(used_cols)} 列", flush=True)

    # ペア列挙
    subj_to_sets = find_pairs_by_subject(args.labelled_dir, subjects)
    for subj, sets in subj_to_sets.items():
        print(f"[CV] {subj}: 検出ペア（ramp+stair）= {sets}", flush=True)

    # サマリ
    summary_lines: List[str] = []
    summary_lines.append(f"PARAMS\tn_estimators={args.n_estimators}\tmax_depth={max_depth_arg}\trandom_state={args.random_state}")
    summary_lines.append(f"WINDOW\t{args.window_ms}ms\tHOP\t{args.hop_ms}ms\tfs\t{fs_hz_arg if fs_hz_arg is not None else 'auto'}")
    summary_lines.append(f"FEATURES\t{len(used_cols)}\t" + ",".join(used_cols))
    summary_lines.append(f"MODE\t{args.cv_mode}")
    summary_lines.append(f"MODEL\t{args.model}")

    if args.cv_mode == "kfold":
        # ===== 被験者内k-fold =====
        for subject in subjects:
            all_sets = subj_to_sets.get(subject, [])
            if not all_sets:
                print(f"[CV:{subject}] 利用可能なペアがありません。スキップします。", flush=True)
                continue

            k = min(args.use_first_k, len(all_sets))
            use_sets = all_sets[:k]
            print(f"[CV:{subject}] 使用ペア（先頭 {k} 件）= {use_sets}", flush=True)

            fold_accs: List[float] = []
            fold_f1s: List[float] = []

            for fold_idx, test_set in enumerate(use_sets, start=1):
                train_sets = [sid for sid in use_sets if sid != test_set]
                train_dsets: List[Dataset] = []
                for sid in train_sets:
                    ds = build_pair_dataset(
                        args.labelled_dir, subject, sid, used_cols, args.window_ms, args.hop_ms, fs_hz_arg
                    )
                    train_dsets.append(ds)
                train_data = concat_datasets(train_dsets, used_cols)

                test_data = build_pair_dataset(
                    args.labelled_dir, subject, test_set, used_cols, args.window_ms, args.hop_ms, fs_hz_arg
                )

                print(
                    f"[CV:{subject}] Fold {fold_idx}/{k} | train X={train_data.X.shape}, test X={test_data.X.shape}",
                    flush=True,
                )

                clf = build_classifier(
                    args.model,
                    n_estimators=args.n_estimators,
                    max_depth=max_depth_arg,
                    random_state=args.random_state,
                    max_features=args.max_features,
                    min_samples_leaf=args.min_samples_leaf,
                    min_samples_split=args.min_samples_split,
                    ccp_alpha=args.ccp_alpha,
                    max_leaf_nodes=max_leaf_nodes_arg,
                    bootstrap=args.bootstrap,
                    max_samples=max_samples_arg,
                    class_weight=class_weight_arg,
                    criterion=args.criterion,
                    lda_solver=args.lda_solver,
                    lda_shrinkage=args.lda_shrinkage,
                    qda_reg=args.qda_reg,
                )
                clf.fit(train_data.X, train_data.y)

                size_b = model_size_bytes(clf)
                size_info = f"{size_b} B" if size_b >= 0 else "N/A"
                try:
                    hp = clf.get_params(deep=False)
                except Exception:
                    hp = {}
                hp_keys = [
                    "n_estimators", "max_depth", "max_features",
                    "min_samples_leaf", "min_samples_split",
                    "ccp_alpha", "max_leaf_nodes",
                    "bootstrap", "max_samples",
                    "class_weight", "criterion", "random_state",
                    "steps", "lda__solver", "lda__shrinkage", "quadraticdiscriminantanalysis__reg_param",
                ]
                hp_str = format_hp(hp, hp_keys)
                total_nodes_str, max_tree_depth_str = describe_model_complexity(clf)

                acc, macro_f1, report, cm, labels_order = evaluate_classifier(clf, test_data.X, test_data.y)
                fold_accs.append(acc)
                fold_f1s.append(macro_f1)

                y_pred = clf.predict(test_data.X)
                mis_idx = np.where(y_pred != test_data.y)[0]
                mis_out = eval_dir / f"CV_{subject}_fold{fold_idx}_misclassified.csv"
                with mis_out.open("w", encoding="utf-8") as fw:
                    fw.write("t_sec,src,true_label,pred_label\n")
                    for i in mis_idx:
                        fw.write(f"{test_data.t_sec[i]:.6f},{test_data.src[i]},{test_data.y[i]},{y_pred[i]}\n")
                print(f"  -> 誤分類ログ: {mis_out} | 件数={len(mis_idx)}", flush=True)

                print(f"[CV:{subject}] Fold {fold_idx}: acc={acc:.6f}, macroF1={macro_f1:.6f}（model size={size_info}）", flush=True)

                out_txt = eval_dir / f"CV_{subject}_fold{fold_idx}.txt"
                with out_txt.open("w", encoding="utf-8") as f:
                    f.write(f"Subject: {subject}\n")
                    f.write(f"Fold: {fold_idx}/{k}\n")
                    f.write(f"Train sets: {', '.join(train_sets)}\n")
                    f.write(f"Test set: {test_set}\n")
                    f.write(f"Model size: {size_info}\n")
                    f.write(f"Hyper params: {hp_str}\n")
                    f.write(f"Model complexity: total_nodes={total_nodes_str}, max_tree_depth={max_tree_depth_str}\n")
                    f.write(f"Window: {args.window_ms} ms, Hop: {args.hop_ms} ms, fs: {fs_hz_arg if fs_hz_arg is not None else 'auto'} Hz\n")
                    f.write(f"Features ({len(used_cols)}): {', '.join(used_cols)}\n\n")
                    f.write(f"Accuracy: {acc:.6f}\n")
                    f.write(f"Macro-F1: {macro_f1:.6f}\n\n")
                    f.write("Classification report:\n")
                    f.write(report + "\n\n")
                    f.write("Labels order:\n")
                    f.write(", ".join(labels_order) + "\n\n")
                    f.write("Confusion matrix (row=true, col=pred):\n")
                    for row in cm:
                        f.write("\t".join(str(int(v)) for v in row) + "\n")
                print(f"  -> 保存: {out_txt}", flush=True)

                summary_lines.append(f"CVFOLD\t{subject}\t{fold_idx}\tacc\t{acc:.6f}")
                summary_lines.append(f"CVFOLD\t{subject}\t{fold_idx}\tmacroF1\t{macro_f1:.6f}")

            if fold_accs:
                mean_acc = float(np.mean(fold_accs))
                mean_f1 = float(np.mean(fold_f1s))
                print(f"[CV:{subject}] 平均: acc={mean_acc:.6f}, macroF1={mean_f1:.6f}", flush=True)
                summary_lines.append(f"CVMEAN\t{subject}\tacc\t{mean_acc:.6f}")
                summary_lines.append(f"CVMEAN\t{subject}\tmacroF1\t{mean_f1:.6f}")

                fold_ids = np.arange(1, len(fold_accs) + 1)
                width = 0.42
                fig, ax = plt.subplots(figsize=(9, 5))
                bars_acc = ax.bar(fold_ids - width/2, fold_accs, width, label="Accuracy", color="tab:blue")
                bars_f1  = ax.bar(fold_ids + width/2, fold_f1s,  width, label="Macro-F1", color="tab:orange")
                ax.axhline(mean_acc, linestyle="--", linewidth=1.5, color="tab:blue",  label=f"Acc mean = {mean_acc:.3f}")
                ax.axhline(mean_f1,  linestyle=":",  linewidth=1.5, color="tab:orange", label=f"F1 mean = {mean_f1:.3f}")
                ax.set_title(f"{subject} - Accuracy & Macro-F1 per fold (k={len(fold_accs)})")
                ax.set_xlabel("Fold")
                ax.set_ylabel("Score")
                ax.set_xticks(fold_ids)
                ax.set_ylim(0, 1.0)
                ax.grid(True, axis="y", linestyle=":", linewidth=0.8)
                ax.legend()
                def _add_labels(bars):
                    for b in bars:
                        h = b.get_height()
                        ax.text(b.get_x() + b.get_width()/2, h, f"{h:.3f}",
                                ha="center", va="bottom", fontsize=9)
                _add_labels(bars_acc)
                _add_labels(bars_f1)
                out_png = eval_dir / f"{subject}_acc_f1_bar.png"
                fig.savefig(out_png, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"[CV:{subject}] 可視化出力: {out_png}", flush=True)

    else:
        # ===== LOSO: 1被験者をテスト、残り全員で学習 =====
        for test_subject in subjects:
            test_sets_all = subj_to_sets.get(test_subject, [])
            if not test_sets_all:
                print(f"[LOSO:test={test_subject}] 利用可能なペアがありません。スキップします。", flush=True)
                continue

            k = min(args.use_first_k, len(test_sets_all))
            test_sets = test_sets_all[:k]
            train_subjects = [s for s in subjects if s != test_subject]

            # ---- 学習データ（全訓練被験者 × その全set）
            train_dsets: List[Dataset] = []
            for subj in train_subjects:
                train_sets = subj_to_sets.get(subj, [])
                if not train_sets:
                    continue
                for sid in train_sets:
                    ds = build_pair_dataset(
                        args.labelled_dir, subj, sid, used_cols, args.window_ms, args.hop_ms, fs_hz_arg
                    )
                    train_dsets.append(ds)
            if not train_dsets:
                print(f"[LOSO:test={test_subject}] 学習データが空です。スキップします。", flush=True)
                continue
            train_data = concat_datasets(train_dsets, used_cols)

            print(f"[LOSO:test={test_subject}] Train X={train_data.X.shape} (subjects={train_subjects})", flush=True)

            # ---- 一度だけ学習
            clf = build_classifier(
                args.model,
                n_estimators=args.n_estimators,
                max_depth=max_depth_arg,
                random_state=args.random_state,
                max_features=args.max_features,
                min_samples_leaf=args.min_samples_leaf,
                min_samples_split=args.min_samples_split,
                ccp_alpha=args.ccp_alpha,
                max_leaf_nodes=max_leaf_nodes_arg,
                bootstrap=args.bootstrap,
                max_samples=max_samples_arg,
                class_weight=class_weight_arg,
                criterion=args.criterion,
                lda_solver=args.lda_solver,
                lda_shrinkage=args.lda_shrinkage,
                qda_reg=args.qda_reg,
            )
            clf.fit(train_data.X, train_data.y)

            size_b = model_size_bytes(clf)
            size_info = f"{size_b} B" if size_b >= 0 else "N/A"
            try:
                hp = clf.get_params(deep=False)
            except Exception:
                hp = {}
            hp_keys = [
                "n_estimators", "max_depth", "max_features",
                "min_samples_leaf", "min_samples_split",
                "ccp_alpha", "max_leaf_nodes",
                "bootstrap", "max_samples",
                "class_weight", "criterion", "random_state",
                "steps", "lda__solver", "lda__shrinkage", "quadraticdiscriminantanalysis__reg_param",
            ]
            hp_str = format_hp(hp, hp_keys)
            total_nodes_str, max_tree_depth_str = describe_model_complexity(clf)

            # ---- テスト: テスト被験者の各setを"fold"として評価し、平均
            fold_accs: List[float] = []
            fold_f1s: List[float] = []

            for fold_idx, test_set in enumerate(test_sets, start=1):
                test_data = build_pair_dataset(
                    args.labelled_dir, test_subject, test_set, used_cols, args.window_ms, args.hop_ms, fs_hz_arg
                )
                print(f"[LOSO:test={test_subject}] Fold {fold_idx}/{k} | test X={test_data.X.shape}", flush=True)

                acc, macro_f1, report, cm, labels_order = evaluate_classifier(clf, test_data.X, test_data.y)
                fold_accs.append(acc)
                fold_f1s.append(macro_f1)

                y_pred = clf.predict(test_data.X)
                mis_idx = np.where(y_pred != test_data.y)[0]
                mis_out = eval_dir / f"CV_{test_subject}_fold{fold_idx}_misclassified.csv"  # 既存命名を維持
                with mis_out.open("w", encoding="utf-8") as fw:
                    fw.write("t_sec,src,true_label,pred_label\n")
                    for i in mis_idx:
                        fw.write(f"{test_data.t_sec[i]:.6f},{test_data.src[i]},{test_data.y[i]},{y_pred[i]}\n")
                print(f"  -> 誤分類ログ: {mis_out} | 件数={len(mis_idx)}", flush=True)

                print(f"[LOSO:test={test_subject}] Fold {fold_idx}: acc={acc:.6f}, macroF1={macro_f1:.6f}（model size={size_info}）", flush=True)

                out_txt = eval_dir / f"CV_{test_subject}_fold{fold_idx}.txt"  # 既存命名を維持
                with out_txt.open("w", encoding="utf-8") as f:
                    f.write(f"Subject: {test_subject}\n")
                    f.write(f"Fold: {fold_idx}/{k}\n")
                    f.write(f"Train subjects: {', '.join(train_subjects)}\n")
                    f.write(f"Train sets per subject: " + ", ".join(f"{s}:{len(subj_to_sets.get(s, []))}" for s in train_subjects) + "\n")
                    f.write(f"Test set: {test_set}\n")
                    f.write(f"Model size: {size_info}\n")
                    f.write(f"Hyper params: {hp_str}\n")
                    f.write(f"Model complexity: total_nodes={total_nodes_str}, max_tree_depth={max_tree_depth_str}\n")
                    f.write(f"Window: {args.window_ms} ms, Hop: {args.hop_ms} ms, fs: {fs_hz_arg if fs_hz_arg is not None else 'auto'} Hz\n")
                    f.write(f"Features ({len(used_cols)}): {', '.join(used_cols)}\n\n")
                    f.write(f"Accuracy: {acc:.6f}\n")
                    f.write(f"Macro-F1: {macro_f1:.6f}\n\n")
                    f.write("Classification report:\n")
                    f.write(report + "\n\n")
                    f.write("Labels order:\n")
                    f.write(", ".join(labels_order) + "\n\n")
                    f.write("Confusion matrix (row=true, col=pred):\n")
                    for row in cm:
                        f.write("\t".join(str(int(v)) for v in row) + "\n")
                print(f"  -> 保存: {out_txt}", flush=True)

                summary_lines.append(f"CVFOLD\t{test_subject}\t{fold_idx}\tacc\t{acc:.6f}")
                summary_lines.append(f"CVFOLD\t{test_subject}\t{fold_idx}\tmacroF1\t{macro_f1:.6f}")

            if fold_accs:
                mean_acc = float(np.mean(fold_accs))
                mean_f1 = float(np.mean(fold_f1s))
                print(f"[LOSO:test={test_subject}] 平均: acc={mean_acc:.6f}, macroF1={mean_f1:.6f}", flush=True)
                summary_lines.append(f"CVMEAN\t{test_subject}\tacc\t{mean_acc:.6f}")
                summary_lines.append(f"CVMEAN\t{test_subject}\tmacroF1\t{mean_f1:.6f}")

                fold_ids = np.arange(1, len(fold_accs) + 1)
                width = 0.42
                fig, ax = plt.subplots(figsize=(9, 5))
                bars_acc = ax.bar(fold_ids - width/2, fold_accs, width, label="Accuracy", color="tab:blue")
                bars_f1  = ax.bar(fold_ids + width/2, fold_f1s,  width, label="Macro-F1", color="tab:orange")
                ax.axhline(mean_acc, linestyle="--", linewidth=1.5, color="tab:blue",  label=f"Acc mean = {mean_acc:.3f}")
                ax.axhline(mean_f1,  linestyle=":",  linewidth=1.5, color="tab:orange", label=f"F1 mean = {mean_f1:.3f}")
                ax.set_title(f"{test_subject} - Accuracy & Macro-F1 per fold (LOSO, k={len(fold_accs)})")
                ax.set_xlabel("Fold (test set id)")
                ax.set_ylabel("Score")
                ax.set_xticks(fold_ids)
                ax.set_ylim(0, 1.0)
                ax.grid(True, axis="y", linestyle=":", linewidth=0.8)
                ax.legend()
                def _add_labels(bars):
                    for b in bars:
                        h = b.get_height()
                        ax.text(b.get_x() + b.get_width()/2, h, f"{h:.3f}",
                                ha="center", va="bottom", fontsize=9)
                _add_labels(bars_acc)
                _add_labels(bars_f1)
                out_png = eval_dir / f"{test_subject}_acc_f1_bar.png"
                fig.savefig(out_png, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"[LOSO:test={test_subject}] 可視化出力: {out_png}", flush=True)

    # サマリーを書き出し
    with (run_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("# Run summary\n")
        f.write(
            f"hyperparams: n_estimators={args.n_estimators}, max_depth={max_depth_arg}, random_state={args.random_state}\n"
        )
        f.write(
            f"window_ms={args.window_ms}, hop_ms={args.hop_ms}, fs={fs_hz_arg if fs_hz_arg is not None else 'auto'}\n"
        )
        f.write(f"mode: {args.cv_mode}\n")
        f.write(f"model: {args.model}\n")
        f.write(f"output_dir: {run_dir}\n\n")
        f.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
