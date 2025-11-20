from __future__ import annotations

# ===============================================================
# ユーザー設定（スクリプト冒頭で編集）
# ===============================================================
# ラベル付きTXTのルート
LABELLED_DIR = "Labelled_data"

# 対象被験者（ファイル名に含まれるキー文字列でマッチ）
# SUBJECTS = ["kanoga", "kaneishi", "murata"]
SUBJECTS = ["murata"]

# 各被験者で使う（ramp+stair）ペアsetの先頭K件（k-foldのkになる）
USE_FIRST_K = 11

# サンプリング周波数（Hz）。Noneで自動推定（t_sの差分中央値から）
USER_FS_HZ: float | None = 200.0

# ==== グリッド候補（ここを編集して探索範囲を変える）====
USER_GRID_WINDOWS_MS = [150,200,250,350,400]  # ウィンドウ長候補 [ms]
USER_GRID_HOPS_MS    = [20, 30, 40, 50, 100]            # ホップ長候補 [ms]
# ===============================================================

import csv
import fnmatch
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ========== 進捗表示ユーティリティ ==========
try:
    # tqdm があれば使う（ネスト対応）
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

def _simple_progress(i: int, total: int, prefix: str = "", width: int = 30):
    """tqdm不在時の簡易プログレス表示"""
    ratio = (i + 1) / max(total, 1)
    filled = int(ratio * width)
    bar = "█" * filled + "-" * (width - filled)
    print(f"\r{prefix} |{bar}| {int(ratio*100):3d}% ({i+1}/{total})", end="", flush=True)
    if i + 1 == total:
        print("", flush=True)

# ==== Build_RF_dataset から再利用 ====
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
    raise SystemExit(
        "[ERR] Build_RF_dataset.py をインポートできませんでした。配置を確認してください。\n"
        f"詳細: {e}"
    )

# ==== 追加: SSC と 絶対値総和の登録（未登録時のみ）====
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
    raise ValueError("ssc_stat: arr must be 1D or 2D")

def abs_sum_stat(arr: np.ndarray, *args, **kwargs) -> np.ndarray:
    arr = np.asarray(arr)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)
    if arr.ndim == 1:
        return np.array(np.sum(np.abs(arr)), dtype=float)
    elif arr.ndim == 2:
        return np.sum(np.abs(arr), axis=0).astype(float)
    raise ValueError("abs_sum_stat: arr must be 1D or 2D")

if isinstance(STAT_FUNCS, dict):
    STAT_FUNCS.setdefault("ssc", ssc_stat)
    STAT_FUNCS.setdefault("abs_sum", abs_sum_stat)

# ==== ここからユーティリティ ====
@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    columns: List[str]
    t_sec: np.ndarray
    src: List[str]

# 使う統計（列名生成に利用）。Build_RF_dataset.STAT_FUNCS のキーに合わせてください。
USER_SELECTED_STATS: List[str] = ["mean", "std", "max", "rms", "min", "abs_sum"]

# 手動で使う列を完全名/ワイルドカードで指定したいときはここで列挙（空なら自動）
MANUAL_FEATURES: List[str] = []           # 例: ["acc_*_mean", "gyro_z_rms"]
MANUAL_FEATURES_FILE: Optional[Path] = None

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

def _make_used_cols() -> List[str]:
    if not FEATURE_NAMES:
        raise SystemExit("Build_RF_dataset.FEATURE_NAMES が必要です")
    auto_used = _infer_used_cols_by_feature_names(FEATURE_NAMES, USER_SELECTED_STATS)
    manual_patterns: List[str] = list(MANUAL_FEATURES)
    if MANUAL_FEATURES_FILE is not None:
        if not MANUAL_FEATURES_FILE.exists():
            raise SystemExit(f"[ERR] MANUAL_FEATURES_FILE が見つかりません: {MANUAL_FEATURES_FILE}")
        manual_patterns += _read_features_file(MANUAL_FEATURES_FILE)
    if not manual_patterns:
        return auto_used
    candidates = _all_possible_cols(FEATURE_NAMES, STAT_FUNCS, USER_SELECTED_STATS)
    used = _resolve_feature_patterns(candidates, manual_patterns)
    if not used:
        raise SystemExit("手動指定の特徴（またはワイルドカード）に一致する列がありません")
    not_matched = [p for p in manual_patterns if len(_resolve_feature_patterns(candidates, [p])) == 0]
    if not_matched:
        print(f"[WARN] マッチしなかった指定: {not_matched}", flush=True)
    return used

# --- データ読み出しの軽量キャッシュ（窓長/ホップが変わっても生データは同じ） ---
_SERIES_CACHE: Dict[Path, object] = {}

def build_features_from_txt(
    txt_path: Path,
    used_cols: Sequence[str],
    window_ms: int,
    hop_ms: int,
    fs_hz: Optional[float],
) -> Dataset:
    if load_set_series is None or compute_window_features is None or sliding_windows_by_count is None:
        raise SystemExit("Build_RF_dataset.py の関数を使用できません。")

    series = _SERIES_CACHE.get(txt_path)
    if series is None:
        series = load_set_series(txt_path)
        _SERIES_CACHE[txt_path] = series

    # サンプリング周波数の決定
    if fs_hz is None:
        if series.t_s.size >= 2:
            dt = float(np.median(np.diff(series.t_s)))
            fs_hz = (1.0 / dt) if dt > 0 else 100.0
        else:
            fs_hz = 100.0

    win_samp = max(int(round((window_ms / 1000.0) * fs_hz)), 2)
    hop_samp = max(int(round((hop_ms / 1000.0) * fs_hz)), 1)

    # used_cols の末尾から必要統計を逆推定
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

    # ウィンドウ列挙
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
        raise SystemExit(f"{txt_path} の特徴量に不足: {missing}")

    X = df_all[used_cols].to_numpy(dtype=float, copy=False)
    y = pd.Series(labels).to_numpy()
    return Dataset(X=X, y=y, columns=list(used_cols), t_sec=np.asarray(t_mids), src=src_names)

def concat_datasets(dsets: Iterable[Dataset], used_cols: Sequence[str]) -> Dataset:
    xs, ys, ts, ss = [], [], [], []
    for ds in dsets:
        if list(ds.columns) != list(used_cols):
            raise SystemExit("列順不一致: used_cols に揃えてください")
        xs.append(ds.X); ys.append(ds.y); ts.append(ds.t_sec); ss.extend(ds.src)
    X = np.concatenate(xs, axis=0) if xs else np.empty((0, len(used_cols)))
    y = np.concatenate(ys, axis=0) if ys else np.empty((0,), dtype=object)
    t = np.concatenate(ts, axis=0) if ts else np.empty((0,))
    return Dataset(X=X, y=y, columns=list(used_cols), t_sec=t, src=ss)

_SET_RE = re.compile(r"set-(\d+)")

def _extract_set_id(name: str) -> Optional[str]:
    m = _SET_RE.search(name)
    return m.group(1) if m else None

def find_pairs_by_subject(
    base_dir: Path | str,
    subject_keys: Sequence[str],
) -> Dict[str, List[str]]:
    base = Path(base_dir)
    subject_to_files: Dict[str, List[Path]] = {subj: [] for subj in subject_keys}
    for p in base.rglob("*.txt"):
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
    base_dir: Path | str,
    subject: str,
    set_id: str,
    used_cols: Sequence[str],
    window_ms: int,
    hop_ms: int,
    fs_hz: Optional[float],
) -> Dataset:
    base = Path(base_dir)
    ramp = None
    stair = None
    for p in base.rglob(f"*{subject}*set-{set_id}*.txt"):
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

def cv_accuracy_for_subject(
    labelled_dir: Path | str,
    subject: str,
    use_first_k: int,
    used_cols: Sequence[str],
    window_ms: int,
    hop_ms: int,
    fs_hz: Optional[float],
    rf_params: Dict[str, object],
) -> float:
    subj_to_sets = find_pairs_by_subject(labelled_dir, [subject])
    sets = subj_to_sets.get(subject, [])
    if not sets:
        return float("nan")
    k = min(use_first_k, len(sets))
    use_sets = sets[:k]

    accs: List[float] = []
    if _HAS_TQDM:
        inner = tqdm(use_sets, desc=f"{subject} CV folds", leave=False)
    else:
        inner = use_sets

    for i, test_sid in enumerate(inner):
        train_sids = [sid for sid in use_sets if sid != test_sid]
        train_ds = concat_datasets(
            [build_pair_dataset(labelled_dir, subject, sid, used_cols, window_ms, hop_ms, fs_hz)
             for sid in train_sids],
            used_cols
        )
        test_ds = build_pair_dataset(labelled_dir, subject, test_sid, used_cols, window_ms, hop_ms, fs_hz)
        if train_ds.X.size == 0 or test_ds.X.size == 0:
            if not _HAS_TQDM:
                _simple_progress(i, k, prefix=f"[{subject}] CV")
            continue
        clf = RandomForestClassifier(**rf_params)
        clf.fit(train_ds.X, train_ds.y)
        y_pred = clf.predict(test_ds.X)
        accs.append(accuracy_score(test_ds.y, y_pred))
        if _HAS_TQDM:
            inner.set_postfix(acc=np.mean(accs) if accs else 0.0)
        else:
            _simple_progress(i, k, prefix=f"[{subject}] CV")
    return float(np.mean(accs)) if accs else float("nan")

def plot_for_subjects(
    labelled_dir: Path | str,
    subjects: List[str],
    use_first_k: int,
    windows_ms: List[int],
    hops_ms: List[int],
    fs_hz_arg: Optional[float],
    rf_params: Dict[str, object],
):
    used_cols = _make_used_cols()

    # 全被験者: (w,h) -> acc を計算
    acc_map: Dict[str, Dict[Tuple[int, int], float]] = {s: {} for s in subjects}

    valid_pairs = [(w, h) for w in windows_ms for h in hops_ms if h <= w]
    print(f"[INFO] 探索ペア数 per subject: {len(valid_pairs)} "
          f"(windows={len(windows_ms)} × hops={len(hops_ms)} / 無効組合せ除外)")

    for subj in subjects:
        print(f"\n[SUBJECT] {subj} — 総組合せ {len(valid_pairs)}")
        if _HAS_TQDM:
            outer = tqdm(valid_pairs, desc=f"{subj} (window×hop)")
        else:
            outer = valid_pairs

        start_t = time.time()
        for j, (w, h) in enumerate(outer):
            acc = cv_accuracy_for_subject(
                labelled_dir, subj, use_first_k, used_cols, w, h, fs_hz_arg, rf_params
            )
            acc_map[subj][(w, h)] = acc
            if _HAS_TQDM:
                outer.set_postfix(window=w, hop=h, acc=f"{acc:.4f}")
            else:
                _simple_progress(j, len(valid_pairs), prefix=f"[{subj}] grid")
        elapsed = time.time() - start_t
        print(f"[SUBJECT] {subj} 完了: {elapsed:.1f}s")

    # ====== CSV 出力 ======
    output_dir = Path("ACC_grid_csv")
    output_dir.mkdir(exist_ok=True)

    for subj in subjects:
        csv_path = output_dir / f"ACC_grid_{subj}.csv"
        print(f"[INFO] CSV 出力: {csv_path}")

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # 1行目: ACC, Sub. A, Window length, ...
            header1 = ["ACC", f"Sub. {subj}", "Window length"] + [""] * (len(windows_ms) - 1)
            writer.writerow(header1)

            # 2行目: Hop length, , 100, 150, 200, ...
            header2 = ["Hop length", ""] + [str(w) for w in windows_ms]
            writer.writerow(header2)

            # 3行目以降: hop ごとの行
            for h in sorted(hops_ms):
                row = ["", h]
                for w in sorted(windows_ms):
                    if h > w:
                        # 無効な組合せは空欄
                        row.append("")
                    else:
                        acc = acc_map[subj].get((w, h), float("nan"))
                        if np.isnan(acc):
                            row.append("")
                        else:
                            # 小数はお好みで桁数調整
                            row.append(f"{acc:.6f}")
                writer.writerow(row)

    # ====== 可視化（被験者ごとに2枚表示）======
    for subj in subjects:
        # ① window固定で hop を掃く
        plt.figure(figsize=(8, 5))
        for w in sorted(windows_ms):
            xs, ys = [], []
            for h in sorted(hops_ms):
                if h > w:
                    continue
                xs.append(h)
                ys.append(acc_map[subj].get((w, h), np.nan))
            if xs:
                plt.plot(xs, ys, marker="o", label=f"window={w} ms")
        plt.title(f"{subj}: Accuracy vs Hop (window fixed)")
        plt.xlabel("Hop length [ms]")
        plt.ylabel("Accuracy")
        plt.ylim(0.0, 1.0)
        plt.grid(True, linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ② hop固定で window を掃く
        plt.figure(figsize=(8, 5))
        for h in sorted(hops_ms):
            xs, ys = [], []
            for w in sorted(windows_ms):
                if h > w:
                    continue
                xs.append(w)
                ys.append(acc_map[subj].get((w, h), np.nan))
            if xs:
                plt.plot(xs, ys, marker="o", label=f"hop={h} ms")
        plt.title(f"{subj}: Accuracy vs Window (hop fixed)")
        plt.xlabel("Window length [ms]")
        plt.ylabel("Accuracy")
        plt.ylim(0.0, 1.0)
        plt.grid(True, linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.show()


def main() -> None:
    fs_hz_arg = USER_FS_HZ if USER_FS_HZ is not None else None

    # RFのハイパラ（固定して比較の公平性を確保）
    rf_params = dict(
        n_estimators=50,
        max_depth=10,
        max_features=8,
        min_samples_leaf=2,
        min_samples_split=4,
        ccp_alpha=1e-4,
        max_leaf_nodes=512,
        bootstrap=True,
        max_samples=0.8,
        class_weight="balanced_subsample",
        criterion="gini",
        n_jobs=-1,
        random_state=0,
    )

    # グリッドの妥当性チェック（hop ≤ window）
    windows_ms = sorted(set(int(w) for w in USER_GRID_WINDOWS_MS if int(w) > 0))
    hops_ms    = sorted(set(int(h) for h in USER_GRID_HOPS_MS if int(h) > 0))
    if not windows_ms or not hops_ms:
        raise SystemExit("[ERR] USER_GRID_WINDOWS_MS / USER_GRID_HOPS_MS を確認してください。")

    # 概要を表示
    print("[CONFIG]")
    print(f"  LABELLED_DIR : {LABELLED_DIR}")
    print(f"  SUBJECTS     : {', '.join(SUBJECTS)}")
    print(f"  USE_FIRST_K  : {USE_FIRST_K}")
    print(f"  fs_hz        : {fs_hz_arg if fs_hz_arg is not None else 'auto'}")
    print(f"  windows(ms)  : {windows_ms}")
    print(f"  hops(ms)     : {hops_ms}")
    print(f"  tqdm         : {'ON' if _HAS_TQDM else 'OFF (fallback)'}")

    plot_for_subjects(
        labelled_dir=LABELLED_DIR,
        subjects=SUBJECTS,
        use_first_k=USE_FIRST_K,
        windows_ms=windows_ms,
        hops_ms=hops_ms,
        fs_hz_arg=fs_hz_arg,
        rf_params=rf_params,
    )

if __name__ == "__main__":
    main()
