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
# === 追加: LDA/QDA + 標準化用 ===
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
    STAT_FUNCS = {}  # type: ignore
    load_set_series = None  # type: ignore
    label_at_window_median = None  # type: ignore
    compute_window_features = None  # type: ignore
    sliding_windows_by_count = None  # type: ignore
    BRD_DEFAULT_STATS = []  # type: ignore

# ==== 追加: モデル切替とスケーリングの既定 ====
DEFAULT_MODEL_KIND = "rf"       # 'rf' / 'lda' / 'qda' のいずれか
DEFAULT_DA_SCALE   = True       # LDA/QDA のとき StandardScaler を適用（--no-scale で無効化可）

# ==== 追加: SSC と 絶対値総和 の実装 & 登録 ====
def _ssc_1d(x: np.ndarray, threshold: float = 0.0) -> int:
    """1D配列のSSC（slope sign changes）を数える。thresholdで微小ノイズを抑制。"""
    if x.size < 3:
        return 0
    if np.isnan(x).any():
        x = np.nan_to_num(x, nan=0.0)
    a = x[1:-1] - x[:-2]
    b = x[1:-1] - x[2:]
    return int(np.sum((a * b) > threshold))

def ssc_stat(arr: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Window内データのSSCをチャネル毎に返す。"""
    arr = np.asarray(arr)
    thr = float(kwargs.get("threshold", 0.0))
    if arr.ndim == 1:
        return np.array(_ssc_1d(arr, threshold=thr), dtype=float)
    elif arr.ndim == 2:
        return np.array([_ssc_1d(arr[:, c], threshold=thr) for c in range(arr.shape[1])], dtype=float)
    else:
        raise ValueError("ssc_stat: arr must be 1D or 2D")

def abs_sum_stat(arr: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Window内データの絶対値総和（Sum of Absolute Values）をチャネル毎に返す。"""
    arr = np.asarray(arr)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)
    if arr.ndim == 1:
        return np.array(np.sum(np.abs(arr)), dtype=float)
    elif arr.ndim == 2:
        return np.sum(np.abs(arr), axis=0).astype(float)
    else:
        raise ValueError("abs_sum_stat: arr must be 1D or 2D")

# STAT_FUNCS に登録（存在する場合）
try:
    import Build_RF_dataset as BRD  # 型: ignore
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

# ==== 追加: 右から9〜4番目の6列を左足IMUとして扱う ====
EXTRA_L_NAMES = ["gyro_x_l", "gyro_y_l", "gyro_z_l", "acc_x_l", "acc_y_l", "acc_z_l"]
_NUM_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")

def _extract_extra_l_columns_from_txt(txt_path: Path) -> Optional[np.ndarray]:
    """
    各行の最初のダブルクォートより左側（数値部）から数値トークンを抽出し、
    右から9〜4番目の6値を [gyro_x_l, gyro_y_l, gyro_z_l, acc_x_l, acc_y_l, acc_z_l] として返す。
    どこかの行で6値を安全に取れなければ None を返す（＝このファイルは拡張6列なしとみなす）。
    """
    rows: List[List[float]] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            qpos = s.find('"')
            left = s[:qpos] if qpos >= 0 else s
            nums = _NUM_RE.findall(left)
            # 右から9〜4番目を取りたいので、最低9個必要
            if len(nums) < 9:
                return None
            try:
                six = [float(nums[-9]), float(nums[-8]), float(nums[-7]),
                       float(nums[-6]), float(nums[-5]), float(nums[-4])]
            except Exception:
                return None
            rows.append(six)
    if not rows:
        return None
    return np.asarray(rows, dtype=float)
# ==== 追加ここまで ====

# ================= ユーザー設定（必要に応じて変更） =================
USER_SELECTED_STATS: List[str] = ["mean", "std", "max", "rms", "min", "abs_sum"]  # 追加統計もOK
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

# ===== 手動指定ゾーン（空なら FEATURE_NAMES×USER_SELECTED_STATS の自動） =====
MANUAL_FEATURES: List[str] = [
    # 例: "gyro_x_l_mean", "acc_z_l_rms", "acc_*_abs_sum", "gyro_*_mean"
]
MANUAL_FEATURES_FILE: Optional[Path] = None  # 例: Path("features.txt")
# ======================================================================

@dataclass
class Dataset:
    """学習/評価用の配列データを保持"""
    X: np.ndarray
    y: np.ndarray
    columns: List[str]
    t_sec: np.ndarray            # ウィンドウ代表時刻（秒, shape=(N,)）
    src: List[str]               # 各行の由来ファイル名

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
    """FEATURE_NAMES × stats から列名を合成。未登録統計はスキップして警告。"""
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
    """FEATURE_NAMES + EXTRA_L_NAMES × (STAT_FUNCS or stats_hint) の全候補"""
    base_features = list(feature_names) + list(EXTRA_L_NAMES)
    if isinstance(stat_funcs, dict) and stat_funcs:
        stats = list(stat_funcs.keys())
    else:
        stats = list(stats_hint)
    out: List[str] = []
    for ch in base_features:
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
    """ワイルドカード（* ?）を candidates にマッチさせ、順序を保ってユニークで返す"""
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

# ==== ここで6列を series に連結する改修 ====
def build_features_from_txt(
    txt_path: Path,
    used_cols: Sequence[str],
    window_ms: int,
    hop_ms: int,
    fs_hz: Optional[float],
) -> Dataset:
    """ラベル付き *.txt からウィンドウ特徴量を生成（列順は used_cols に合わせる）"""
    if load_set_series is None or compute_window_features is None or sliding_windows_by_count is None:
        raise SystemExit("Build_RF_dataset.py の関数をインポートできませんでした。配置を確認してください。")

    series = load_set_series(txt_path)

    # 追加: 右から9〜4番目6列（左足IMU）を抽出して後方に連結（行数一致時のみ）
    extra6 = _extract_extra_l_columns_from_txt(txt_path)
    use_extra = extra6 is not None and extra6.shape[0] == series.t_s.shape[0]
    if use_extra:
        data_aug = np.hstack([series.data, extra6])
        feature_names_for_this_file = list(FEATURE_NAMES) + list(EXTRA_L_NAMES)
    else:
        data_aug = series.data
        feature_names_for_this_file = list(FEATURE_NAMES)

    # サンプリング周波数の決定
    if fs_hz is None:
        if series.t_s.size >= 2:
            dt = float(np.median(np.diff(series.t_s)))
            fs_hz = (1.0 / dt) if dt > 0 else 100.0
        else:
            fs_hz = 100.0

    win_samp = max(int(round((window_ms / 1000.0) * fs_hz)), 2)
    hop_samp = max(int(round((hop_ms / 1000.0) * fs_hz)), 1)

    # 使用統計量は used_cols の末尾一致で逆推定（fallbackあり）
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
    rows: List[List[float]] = []
    labels: List[str] = []
    t_mids: List[float] = []
    src_names: List[str] = []

    # 実際に使うチャネルでヘッダを組む
    all_headers: List[str] = []
    for ch in feature_names_for_this_file:
        for st in stat_names:
            all_headers.append(f"{ch}_{st}")

    for a, b in windows:
        t_sub = series.t_s[a : b + 1]
        d_sub = data_aug[a : b + 1, :]
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

    # used_cols が df_all に存在するかチェック
    missing = [c for c in used_cols if c not in df_all.columns]
    if missing:
        raise SystemExit(f"{txt_path} の特徴量に不足があります。欠落列: {missing}")

    X = df_all[used_cols].to_numpy(dtype=float, copy=False)
    y = pd.Series(labels).to_numpy()
    return Dataset(
        X=X, y=y, columns=list(used_cols),
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

def train_rf_classifier(
    X: np.ndarray, y: np.ndarray, n_estimators: int, max_depth: Optional[int], random_state: int,
    max_features: int,
    min_samples_leaf: int, min_samples_split: int,
    ccp_alpha: float, max_leaf_nodes: Optional[int],
    bootstrap: bool, max_samples: Optional[float],
    class_weight: Optional[str], criterion: str,
) -> RandomForestClassifier:
    if X.size == 0:
        raise SystemExit("学習データが空です")
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
    clf.fit(X, y)
    return clf

def evaluate_classifier(
    clf,
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

# ------------------ CV 用ユーティリティ ------------------
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
    戻り値: { "kanoga": ["0001","0002",...], "kaneishi":[...] }
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

        # ramp と stair が両方揃っている set だけ採用
        ok_sets = [sid for sid, d in by_set.items() if "ramp" in d and "stair" in d]
        ok_sets.sort(key=lambda s: int(s))  # 数値昇順
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

# === 追加: モデルビルダー（RF/LDA/QDA を統一的に生成） ===
def build_estimator(
    model_kind: str,
    *,
    # RF
    n_estimators: int, max_depth: Optional[int], random_state: int,
    max_features: int, min_samples_leaf: int, min_samples_split: int,
    ccp_alpha: float, max_leaf_nodes: Optional[int],
    bootstrap: bool, max_samples: Optional[float],
    class_weight: Optional[str], criterion: str,
    # DA
    scale: bool, lda_solver: str, qda_reg: Optional[float],
):
    model_kind = model_kind.lower()
    if model_kind == "rf":
        return RandomForestClassifier(
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
    elif model_kind == "lda":
        lda = LinearDiscriminantAnalysis(solver=lda_solver)
        if scale:
            return Pipeline([("scaler", StandardScaler()), ("lda", lda)])
        else:
            return lda
    elif model_kind == "qda":
        qda = QuadraticDiscriminantAnalysis(reg_param=0.0 if qda_reg is None else float(qda_reg))
        if scale:
            return Pipeline([("scaler", StandardScaler()), ("qda", qda)])
        else:
            return qda
    else:
        raise SystemExit(f"[ERR] unknown --model: {model_kind}")

def estimator_params_str(est) -> str:
    """保存用に主要ハイパラを整形（RFとDAでキーを切替）"""
    try:
        if isinstance(est, Pipeline):
            # 最後のステップを取り出す
            last = est.steps[-1][1]
            if isinstance(last, LinearDiscriminantAnalysis):
                hp = last.get_params(deep=False)
                return format_hp(hp, ["solver"])
            elif isinstance(last, QuadraticDiscriminantAnalysis):
                hp = last.get_params(deep=False)
                return format_hp(hp, ["reg_param"])
            else:
                hp = last.get_params(deep=False)
                return ", ".join([f"{k}={v}" for k, v in hp.items()])
        elif isinstance(est, LinearDiscriminantAnalysis):
            hp = est.get_params(deep=False)
            return format_hp(hp, ["solver"])
        elif isinstance(est, QuadraticDiscriminantAnalysis):
            hp = est.get_params(deep=False)
            return format_hp(hp, ["reg_param"])
        elif isinstance(est, RandomForestClassifier):
            hp = est.get_params(deep=False)
            keys = [
                "n_estimators", "max_depth", "max_features",
                "min_samples_leaf", "min_samples_split",
                "ccp_alpha", "max_leaf_nodes",
                "bootstrap", "max_samples",
                "class_weight", "criterion", "random_state",
            ]
            return format_hp(hp, keys)
        else:
            return "N/A"
    except Exception:
        return "N/A"

# ------------------ メイン処理 ------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="kanoga/kaneishi/murata それぞれで（ramp+stair）を1ペアとする被験者内 k-fold CV を実行"
    )
    parser.add_argument("--labelled-dir", type=Path, default=Path("Labelled_data/merged"), help="ラベル付きTXTのルート")
    parser.add_argument("--subjects", type=str, default="sasaki", help="対象被験者（カンマ区切り）")
    parser.add_argument("--use-first-k", type=int, default=11, help="各被験者で使用する先頭ペア数（k-fold の k）")
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

    # === 追加: モデル切替/スケーリング/DAハイパラ ===
    parser.add_argument("--model", type=str, choices=["rf", "lda", "qda"], default=DEFAULT_MODEL_KIND,
                        help="学習モデルを選択（rf/lda/qda）")
    parser.add_argument("--scale", action="store_true", default=DEFAULT_DA_SCALE,
                        help="LDA/QDAで標準化を適用する（既定: 有効）")
    parser.add_argument("--no-scale", dest="scale", action="store_false",
                        help="標準化を適用しない（LDA/QDA向け）")
    parser.add_argument("--lda-solver", type=str, default="svd",
                        choices=["svd", "lsqr", "eigen"], help="LDAのsolver（既定: svd）")
    parser.add_argument("--qda-reg", type=float, default=None,
                        help="QDAのreg_param（0.0〜、未指定なら0.0）")

    args = parser.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    if not subjects:
        raise SystemExit("--subjects が空です")

    fs_hz_arg = None if (isinstance(args.fs_hz, float) and np.isnan(args.fs_hz)) else float(args.fs_hz)
    max_depth_arg = None if args.max_depth is None or args.max_depth < 0 else int(args.max_depth)
    max_samples_arg = None if (args.max_samples is not None and args.max_samples < 0) else args.max_samples
    max_leaf_nodes_arg = None if (args.max_leaf_nodes is not None and args.max_leaf_nodes <= 0) else args.max_leaf_nodes
    class_weight_arg = None if (isinstance(args.class_weight, str) and args.class_weight.lower() in ["none", "null", ""]) else args.class_weight

    # 出力先ディレクトリ（モデル名も含める）
    model_tag = f"{args.model}{'_sc' if (args.model in ['lda','qda'] and args.scale) else ''}"
    run_dir = args.out_dir / f"{model_tag}_ne{args.n_estimators}_md{('None' if max_depth_arg is None else max_depth_arg)}_rs{args.random_state}_k-fold_with_left_sasaki cv"
    eval_dir = run_dir / "evals"
    _ensure_dir(run_dir)
    _ensure_dir(eval_dir)
    print(f"[CV] 出力先: {run_dir}（詳細: {eval_dir}）", flush=True)

    # 列名（全 fold で共通にしたい）
    if not FEATURE_NAMES:
        raise SystemExit("Build_RF_dataset.FEATURE_NAMES が必要です")

    # 自動候補（EXTRA_L_NAMESも含める）
    auto_used_cols = _infer_used_cols_by_feature_names(FEATURE_NAMES + EXTRA_L_NAMES, USER_SELECTED_STATS)

    # 手動指定の収集（配列＋ファイル）
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
        print(f"[CV] 特徴量列（自動: (FEATURE_NAMES+EXTRA_L_NAMES) × USER_SELECTED_STATS）: {len(used_cols)} 列", flush=True)

    # 被験者ごとに ramp/stair のペア（set-XXXX）を列挙
    subj_to_sets = find_pairs_by_subject(args.labelled_dir, subjects)
    for subj, sets in subj_to_sets.items():
        print(f"[CV] {subj}: 検出ペア（ramp+stair）= {sets}", flush=True)

    # サマリアウト
    summary_lines: List[str] = []
    summary_lines.append(f"MODEL\t{args.model}\tscale={args.scale}\tlda_solver={args.lda_solver}\tqda_reg={args.qda_reg}")
    summary_lines.append(f"PARAMS\tn_estimators={args.n_estimators}\tmax_depth={max_depth_arg}\trandom_state={args.random_state}")
    summary_lines.append(f"WINDOW\t{args.window_ms}ms\tHOP\t{args.hop_ms}ms\tfs\t{fs_hz_arg if fs_hz_arg is not None else 'auto'}")
    summary_lines.append(f"FEATURES\t{len(used_cols)}\t" + ",".join(used_cols))

    # 各被験者で CV 実行
    for subject in subjects:
        all_sets = subj_to_sets.get(subject, [])
        if not all_sets:
            print(f"[CV:{subject}] 利用可能なペアがありません。スキップします。", flush=True)
            continue

        k = min(args.use_first_k, len(all_sets))
        use_sets = all_sets[:k]
        print(f"[CV:{subject}] 使用ペア（先頭 {k} 件）= {use_sets}", flush=True)

        # 平均算出用バッファ
        fold_accs: List[float] = []
        fold_f1s: List[float] = []

        # k-fold: 各回で1つの set をテスト、残りを学習
        for fold_idx, test_set in enumerate(use_sets, start=1):
            # 学習データ
            train_sets = [sid for sid in use_sets if sid != test_set]
            train_dsets: List[Dataset] = []
            for sid in train_sets:
                ds = build_pair_dataset(
                    args.labelled_dir, subject, sid, used_cols, args.window_ms, args.hop_ms, fs_hz_arg
                )
                train_dsets.append(ds)
            train_data = concat_datasets(train_dsets, used_cols)

            # テストデータ
            test_data = build_pair_dataset(
                args.labelled_dir, subject, test_set, used_cols, args.window_ms, args.hop_ms, fs_hz_arg
            )

            print(
                f"[CV:{subject}] Fold {fold_idx}/{k} | train X={train_data.X.shape}, test X={test_data.X.shape}",
                flush=True,
            )

            # === 学習（モデル切替） ===
            if args.model == "rf":
                clf = train_rf_classifier(
                    train_data.X, train_data.y, args.n_estimators, max_depth_arg, args.random_state,
                    args.max_features,
                    args.min_samples_leaf, args.min_samples_split,
                    args.ccp_alpha, max_leaf_nodes_arg,
                    args.bootstrap, max_samples_arg,
                    class_weight_arg, args.criterion,
                )
                est = clf
                size_b = model_size_bytes(est)
                size_info = f"{size_b} B" if size_b >= 0 else "N/A"
                hp_str = estimator_params_str(est)
                total_nodes = sum(t.tree_.node_count for t in clf.estimators_)
                max_tree_depth = max(t.tree_.max_depth for t in clf.estimators_)
            else:
                est = build_estimator(
                    args.model,
                    n_estimators=args.n_estimators, max_depth=max_depth_arg, random_state=args.random_state,
                    max_features=args.max_features, min_samples_leaf=args.min_samples_leaf, min_samples_split=args.min_samples_split,
                    ccp_alpha=args.ccp_alpha, max_leaf_nodes=max_leaf_nodes_arg,
                    bootstrap=args.bootstrap, max_samples=max_samples_arg,
                    class_weight=class_weight_arg, criterion=args.criterion,
                    scale=args.scale, lda_solver=args.lda_solver, qda_reg=args.qda_reg,
                )
                est.fit(train_data.X, train_data.y)
                size_b = model_size_bytes(est)
                size_info = f"{size_b} B" if size_b >= 0 else "N/A"
                hp_str = estimator_params_str(est)
                total_nodes = "N/A"
                max_tree_depth = "N/A"

            # 評価
            acc, macro_f1, report, cm, labels_order = evaluate_classifier(est, test_data.X, test_data.y)
            fold_accs.append(acc)
            fold_f1s.append(macro_f1)

            # 誤分類ログ（t_sec, src を活用）
            y_pred = est.predict(test_data.X)
            mis_idx = np.where(y_pred != test_data.y)[0]
            mis_out = eval_dir / f"CV_{subject}_fold{fold_idx}_misclassified.csv"
            with mis_out.open("w", encoding="utf-8") as fw:
                fw.write("t_sec,src,true_label,pred_label\n")
                for i in mis_idx:
                    fw.write(f"{test_data.t_sec[i]:.6f},{test_data.src[i]},{test_data.y[i]},{y_pred[i]}\n")
            print(f"  -> 誤分類ログ: {mis_out} | 件数={len(mis_idx)}", flush=True)

            print(f"[CV:{subject}] Fold {fold_idx}: acc={acc:.6f}, macroF1={macro_f1:.6f}（model size={size_info}）", flush=True)

            # 詳細保存
            out_txt = eval_dir / f"CV_{subject}_fold{fold_idx}.txt"
            with out_txt.open("w", encoding="utf-8") as f:
                f.write(f"Subject: {subject}\n")
                f.write(f"Fold: {fold_idx}/{k}\n")
                f.write(f"Train sets: {', '.join(train_sets)}\n")
                f.write(f"Test set: {test_set}\n")
                f.write(f"Model: {args.model} (scale={args.scale}, lda_solver={args.lda_solver}, qda_reg={args.qda_reg})\n")
                f.write(f"Model size: {size_info}\n")
                f.write(f"Hyper params: {hp_str}\n")
                f.write(f"Model complexity: total_nodes={total_nodes}, max_tree_depth={max_tree_depth}\n")
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

            # サマリ（fold単位）
            summary_lines.append(f"CVFOLD\t{subject}\t{fold_idx}\tacc\t{acc:.6f}")
            summary_lines.append(f"CVFOLD\t{subject}\t{fold_idx}\tmacroF1\t{macro_f1:.6f}")

        # 被験者ごとの平均
        if fold_accs:
            mean_acc = float(np.mean(fold_accs))
            mean_f1 = float(np.mean(fold_f1s))
            print(f"[CV:{subject}] 平均: acc={mean_acc:.6f}, macroF1={mean_f1:.6f}", flush=True)
            summary_lines.append(f"CVMEAN\t{subject}\tacc\t{mean_acc:.6f}")
            summary_lines.append(f"CVMEAN\t{subject}\tmacroF1\t{mean_f1:.6f}")

            # 可視化（Acc/F1棒グラフ）
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
            # データラベル
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

    # サマリー出力
    with (run_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("# Run summary\n")
        f.write(f"model: {args.model}, scale={args.scale}, lda_solver={args.lda_solver}, qda_reg={args.qda_reg}\n")
        f.write(
            f"hyperparams (rf): n_estimators={args.n_estimators}, max_depth={max_depth_arg}, random_state={args.random_state}\n"
        )
        f.write(
            f"window_ms={args.window_ms}, hop_ms={args.hop_ms}, fs={fs_hz_arg if fs_hz_arg is not None else 'auto'}\n"
        )
        f.write(f"output_dir: {run_dir}\n\n")
        f.write("\n".join(summary_lines) + "\n")

if __name__ == "__main__":
    main()
