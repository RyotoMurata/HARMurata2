"""
Train_RF_from_csv
=================

目的:
  - Build_RF_dataset.py で作成した特徴量CSVからランダムフォレスト分類器を学習します。
  - K分割の層化交差検証（StratifiedKFold）で精度とF1を評価できます。
  - さらに、外部の「セットID付きラベルテキスト（*_set-*.txt）」からスライディングウィンドウで
    特徴量を再計算し、学習済みモデルで予測する外部評価も行えます。

モデル保存:
  - --save-model オプションで、学習済みモデルをファイルに保存できます。
    モデル本体に加え、使用した特徴列名や統計量名などのメタ情報を同梱します。

使い方（例）:
  - 交差検証のみ:
      python Train_RF_from_csv.py --csv Featured_data/rf_dataset.csv \
        --stats mean,std,iqr,rms,kurtosis
  - 外部評価（Labelled_data下の *_set-*.txt を自動検索）:
      python Train_RF_from_csv.py --eval-input-dir Labelled_data --eval-glob "*_set-*.txt"
  - 特定ファイルで外部評価:
      python Train_RF_from_csv.py --eval-files Labelled_data/stair_t8_set-0006.txt

メモ:
  - --stats で指定できる統計量名は Build_RF_dataset.py と合わせてください
    （mean, std, min, max, range, median, iqr, rms, skewness, kurtosis, zcr, abs_integral など）。
  - 最大5種類まで指定可能です（列数が膨らみすぎるのを防ぐため）。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
import sys
from datetime import datetime as _dt_all
try:
    import joblib as _joblib
except Exception:  # joblib が使えない場合は pickle にフォールバック
    _joblib = None
    import pickle as _pickle


# ===== ユーザーが調整しやすい既定値（必要に応じて変更） =====
# Build_RF_dataset.py が作る学習用CSVのパス
CSV_PATH = Path("Featured_data/rf_dataset_kanoga_all.csv")

# 列名の接尾辞（統計量名）から、最大5種類だけ採用します。順序もそのまま反映されます。
# 例: mean, std, min, max, range, median, iqr, rms, skewness, kurtosis, zcr, abs_integral
SELECTED_STATS: List[str] = ["mean", "std", "iqr", "rms", "kurtosis"]

# 外部評価用の既定ファイル（SetID付きのラベルテキスト）。
# --eval-files が省略された場合に使用します。ワイルドカード可。
# 例: Labelled_data/*_set-000[1-2].txt
DEFAULT_EVAL_PATHS: List[str] = ["Labelled_data/stair_t8_kanoga_set-0006.txt"]

# ===== モデル保存に関する既定値（必要に応じて変更） =====
# 既定で自動保存するか（Trueで自動保存、Falseで保存しない）
SAVE_MODEL_BY_DEFAULT: bool = True
# 保存ディレクトリとファイル名（--save-model 未指定時の保存先を決定）
MODEL_SAVE_DIR: Path = Path("models")
MODEL_SAVE_NAME: str = "rf_kanoga_all_h250.joblib"
# ================================================

# RandomForest のハイパーパラメータ
N_ESTIMATORS: int = 300
MAX_DEPTH: int | None = None
MIN_SAMPLES_SPLIT: int = 2
MIN_SAMPLES_LEAF: int = 1
MAX_FEATURES: str | int | float | None = "sqrt"  # 例: "sqrt", "log2", None
CLASS_WEIGHT: str | dict | None = None  # 例: "balanced"
RANDOM_STATE: int = 42
N_JOBS: int = -1

# 交差検証
CV_SPLITS: int = 5
CV_SHUFFLE: bool = True

# 進行ログの冗長度（0=静か, 1=sklearnから各ツリーの進捗表示）
VERBOSE: int = 1
# ================================================


HOUSEKEEPING_COLS = {"file", "set_id", "t_start_s", "t_end_s", "label"}
FEATURE_NAMES: List[str] = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "encoder_angle",
    "grf_x",
    "grf_y",
    "grf_z",
    "grt_x",
    "grt_y",
    "grt_z",
]




def select_feature_columns(columns: Sequence[str], stats: Sequence[str]) -> List[str]:
    """列名から、指定した統計量の接尾辞で終わる特徴列だけを選択します。

    例: 列名 "acc_x_mean" は統計量名 "mean" によって選択されます（"_mean" で終端）。
    最大5種類の統計量だけを使用します。
    """
    chosen = list(stats[:5])  # keep at most 5
    out: List[str] = []
    for c in columns:
        if c in HOUSEKEEPING_COLS:
            continue
        for s in chosen:
            if c.endswith(f"_{s}"):
                out.append(c)
                break
    return out


def load_dataset(csv_path: Path, stats: Sequence[str]) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """CSVを読み込み、指定統計量に対応する特徴列だけを抽出して X, y を返します。

    戻り値:
      - X: 特徴量行列 (float, shape = [Nサンプル, D特徴])
      - y: ラベル配列 (str, shape = [Nサンプル])
      - feat_cols: 使用した列名リスト（学習と外部評価の整合に利用）
    """
    print(f"[1/4] CSVを読み込み中: {csv_path}")
    df = pd.read_csv(csv_path)
    # Ensure label exists
    if "label" not in df.columns:
        raise SystemExit("CSV missing 'label' column.")
    # Pick columns
    feat_cols = select_feature_columns(df.columns, stats)
    if not feat_cols:
        raise SystemExit("No feature columns match the selected stats. Check SELECTED_STATS or CSV headers.")
    print(f"       統計量={list(stats[:5])} で {len(feat_cols)} 列を使用")
    # X, y
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["label"].astype(str).str.strip('\"').to_numpy()
    return X, y, feat_cols


def cross_validate_and_report(X: np.ndarray, y: np.ndarray, feat_cols: List[str]) -> None:
    """層化K分割交差検証を実施し、平均精度・F1と詳細レポートを表示します。"""
    print(f"[2/4] 層化K分割を開始 (k={CV_SPLITS}, shuffle={CV_SHUFFLE}) …")
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=CV_SHUFFLE, random_state=RANDOM_STATE)

    fold_accs: List[float] = []
    fold_f1s: List[float] = []
    y_true_all: List[str] = []
    y_pred_all: List[str] = []

    importances_list: List[np.ndarray] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(
            f"-> Fold {fold}/{CV_SPLITS}: 学習={X_train.shape[0]}, 検証={X_test.shape[0]}, 特徴次元={X_train.shape[1]}"
        )
        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            class_weight=CLASS_WEIGHT,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            verbose=VERBOSE,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        fold_accs.append(acc)
        fold_f1s.append(f1m)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        print(f"   Fold {fold} 指標: 正解率={acc:.4f}, Macro-F1={f1m:.4f}")
        # collect feature importances for this fold
        if hasattr(rf, "feature_importances_"):
            importances_list.append(rf.feature_importances_.astype(float))

    print("[3/4] 結果を集計 …")
    acc_mean, acc_std = float(np.mean(fold_accs)), float(np.std(fold_accs))
    f1_mean, f1_std = float(np.mean(fold_f1s)), float(np.std(fold_f1s))
    print(f"CV 正解率: 平均={acc_mean:.4f} ± {acc_std:.4f}")
    print(f"CV Macro-F1: 平均={f1_mean:.4f} ± {f1_std:.4f}")

    print("[4/4] 詳細レポート（全Fold結合） …")
    labels_unique = list(np.unique(y))
    print("\n分類レポート:\n" + classification_report(y_true_all, y_pred_all, labels=labels_unique))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels_unique)
    print("混同行列（行=true, 列=pred）:")
    header = "      " + " ".join(f"{lbl:>15s}" for lbl in labels_unique)
    print(header)
    for i, row in enumerate(cm):
        print(f"{labels_unique[i]:>6s} " + " ".join(f"{v:15d}" for v in row))

    # Feature importance (mean ± std over folds)
    if importances_list:
        imps = np.vstack(importances_list)
        imp_mean = imps.mean(axis=0)
        imp_std = imps.std(axis=0)
        order = np.argsort(imp_mean)[::-1]
        top_k = min(20, len(feat_cols))
        print("\n平均重要度による上位特徴:")
        print("   rank  importance(mean±std)  feature")
        for rank, idx in enumerate(order[:top_k], start=1):
            print(f"   {rank:>4d}   {imp_mean[idx]:.6f}±{imp_std[idx]:.6f}   {feat_cols[idx]}")
    else:
        print("\n特徴量重要度は利用できません。")


# --------- External evaluation on set-labeled text files ---------
def parse_timestamp(line: str):
    """先頭の角括弧 [YYYY-mm-dd HH:MM:SS(.fff)] を日時として解析します。"""
    line = line.strip()
    if not line.startswith("["):
        return None
    try:
        idx = line.index("]")
    except ValueError:
        return None
    ts_str = line[1:idx].strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            from datetime import datetime

            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None


def parse_values_label_setid(line: str):
    """1行からセンサ値（17列想定）とラベル（"..."）を抽出します。

    戻り値: (values[0:17], label or None) / 解析に失敗したら None
    """
    import re as _re

    try:
        idx = line.index("]")
    except ValueError:
        return None
    rest = line[idx + 1 :].strip()
    rest = (
        rest.replace("uart:~$", " ")
        .replace("uart:$", " ")
        .replace("uart:", " ")
        .strip()
    )
    quoted = _re.findall(r'"([^"]+)"', rest)
    label = None
    if quoted:
        for q in quoted:
            if not q.startswith("SetID="):
                label = q
                break
        qpos = rest.find('"')
        if qpos >= 0:
            rest = rest[:qpos].strip()
    vals: List[float] = []
    for tok in rest.split():
        try:
            vals.append(float(tok))
        except ValueError:
            return None
    if len(vals) < 17:
        return None
    return vals[:17], label


def sliding_windows(t: np.ndarray, window_sec: float, hop_sec: float) -> List[tuple[int, int]]:
    """時刻配列 t に対して、秒単位の窓幅・ホップ幅で [start_idx, end_idx] のペアを作成。"""
    if t.size == 0:
        return []
    n = t.size
    t_start = float(t[0])
    t_end_total = float(t[-1])
    starts: List[int] = []
    ends: List[int] = []
    current = t_start
    i0 = 0
    while current + window_sec <= t_end_total + 1e-9:
        while i0 < n and t[i0] < current - 1e-12:
            i0 += 1
        wend = current + window_sec + 1e-12
        i1 = i0
        while i1 < n and t[i1] <= wend:
            i1 += 1
        if i1 - i0 >= 2:
            starts.append(i0)
            ends.append(i1 - 1)
        current += hop_sec
    return list(zip(starts, ends))


def label_at_window_median(labels: Sequence[object]):
    """ウィンドウ内の中央時刻付近に最初に現れる非Noneラベルを返します。"""
    n = len(labels)
    if n == 0:
        return None
    mid = n // 2
    if labels[mid] is not None:
        return labels[mid]
    for off in range(1, n):
        for cand in (mid - off, mid + off):
            if 0 <= cand < n and labels[cand] is not None:
                return labels[cand]
    return None


def compute_window_features(data: np.ndarray, t: np.ndarray, stat_names: Sequence[str]) -> List[float]:
    """ウィンドウ内データから、指定された統計量で特徴量ベクトルを作成します。

    Build_RF_dataset.py と同じ統計量定義をここでも再現しています。
    """
    # Build_RF_dataset と同等の統計量群
    def stat_mean(arr, tt):
        return float(np.mean(arr))

    def stat_std(arr, tt):
        return float(np.std(arr, ddof=0))

    def stat_min(arr, tt):
        return float(np.min(arr))

    def stat_max(arr, tt):
        return float(np.max(arr))

    def stat_range(arr, tt):
        return float(np.max(arr) - np.min(arr))

    def stat_median(arr, tt):
        return float(np.median(arr))

    def stat_iqr(arr, tt):
        q75 = np.percentile(arr, 75)
        q25 = np.percentile(arr, 25)
        return float(q75 - q25)

    def stat_rms(arr, tt):
        return float(np.sqrt(np.mean(np.square(arr))))

    def stat_skewness(arr, tt):
        mu = np.mean(arr)
        sd = np.std(arr, ddof=0)
        if sd == 0:
            return 0.0
        m3 = np.mean((arr - mu) ** 3)
        return float(m3 / (sd ** 3))

    def stat_kurtosis(arr, tt):
        mu = np.mean(arr)
        sd = np.std(arr, ddof=0)
        if sd == 0:
            return 0.0
        m4 = np.mean((arr - mu) ** 4)
        return float(m4 / (sd ** 4))

    def stat_zcr(arr, tt):
        s = np.sign(arr)
        s[s == 0] = 1
        return float(np.sum(s[1:] * s[:-1] < 0))

    def stat_abs_integral(arr, tt):
        if len(arr) < 2:
            return 0.0
        trapezoid = getattr(np, "trapezoid", None)
        if trapezoid is not None:
            return float(trapezoid(np.abs(arr), tt))
        else:
            return float(np.trapz(np.abs(arr), tt))

    STAT_FUNCS = {
        "mean": stat_mean,
        "std": stat_std,
        "min": stat_min,
        "max": stat_max,
        "range": stat_range,
        "median": stat_median,
        "iqr": stat_iqr,
        "rms": stat_rms,
        "skewness": stat_skewness,
        "kurtosis": stat_kurtosis,
        "zcr": stat_zcr,
        "abs_integral": stat_abs_integral,
    }

    row: List[float] = []
    for col in range(data.shape[1]):
        x = data[:, col]
        for st in stat_names:
            row.append(float(STAT_FUNCS[st](x, t)))
    return row


def feature_headers_for(stats: Sequence[str]) -> List[str]:
    """特徴名と統計量名から、学習列ヘッダ（例: acc_x_mean）を生成。"""
    headers: List[str] = []
    for ch in FEATURE_NAMES:
        for st in stats:
            headers.append(f"{ch}_{st}")
    return headers


def build_eval_matrix(files: List[Path], stats: Sequence[str], window_sec: float, hop_sec: float):
    """外部のセット付きラベルテキスト群から、スライディングウィンドウで特徴量行列とラベルを生成。"""
    X_rows: List[List[float]] = []
    y_rows: List[str] = []
    from datetime import datetime as _dt

    for fp in files:
        t0: _dt | None = None
        times: List[float] = []
        rows: List[List[float]] = []
        labels: List[object] = []
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                ts = parse_timestamp(line)
                if ts is None:
                    continue
                parsed = parse_values_label_setid(line)
                if parsed is None:
                    continue
                vals, lbl = parsed
                if t0 is None:
                    t0 = ts
                times.append((ts - t0).total_seconds())
                rows.append(vals)
                labels.append(lbl)
        if not rows:
            continue
        t = np.asarray(times, dtype=float)
        data = np.asarray(rows, dtype=float)
        for a, b in sliding_windows(t, window_sec, hop_sec):
            t_sub = t[a : b + 1]
            d_sub = data[a : b + 1, :]
            lbl_sub = labels[a : b + 1]
            lbl = label_at_window_median(lbl_sub)
            if lbl is None:
                continue
            feats = compute_window_features(d_sub, t_sub, stats)
            X_rows.append([float(v) for v in feats])
            y_rows.append(str(lbl))
    X = np.asarray(X_rows, dtype=float) if X_rows else np.zeros((0, len(FEATURE_NAMES) * len(stats)))
    y = np.asarray(y_rows, dtype=str) if y_rows else np.zeros((0,), dtype=str)
    return X, y


def _expand_paths_from_strings(paths: Sequence[str]) -> List[Path]:
    """ワイルドカード対応のパス展開（重複除去、存在確認込み）。"""
    out: List[Path] = []
    for s in paths:
        if any(ch in s for ch in "*?[]"):
            out.extend(Path().glob(s))
        else:
            out.append(Path(s))
    # unique, keep order
    uniq: List[Path] = []
    seen = set()
    for p in out:
        try:
            real = p.resolve()
        except Exception:
            continue
        if not p.is_file() or real in seen:
            continue
        seen.add(real)
        uniq.append(p)
    return uniq


def _train_rf_on_full(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """現在のハイパーパラメータでCSV全体に学習したモデルを返します。"""
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        class_weight=CLASS_WEIGHT,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=VERBOSE,
    )
    rf.fit(X, y)
    return rf


def _save_model_bundle(path: Path, rf: RandomForestClassifier, feat_cols: List[str], stats: Sequence[str], X: np.ndarray, y: np.ndarray) -> None:
    """モデルと付随メタ情報を保存します（joblib優先、不可ならpickle）。"""
    bundle = {
        "model": rf,
        "feature_columns": list(feat_cols),
        "stats": list(stats),
        "meta": {
            "trained_on_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]) if X.ndim == 2 else None,
            "classes": sorted(list({str(c) for c in y})),
            "sklearn_version": getattr(sys.modules.get("sklearn"), "__version__", "unknown"),
            "saved_at": _dt_all.now().isoformat(timespec="seconds"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if _joblib is not None:
        _joblib.dump(bundle, path)
    else:
        with path.open("wb") as f:
            _pickle.dump(bundle, f)


def main() -> None:
    """エントリポイント: 引数を解釈し、学習・CV または外部評価を実行します。"""
    # 関数内でグローバル値を上書きするため、先に宣言
    global N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF, MAX_FEATURES
    global CLASS_WEIGHT, CV_SPLITS, CV_SHUFFLE, RANDOM_STATE, N_JOBS, VERBOSE

    p = argparse.ArgumentParser(
        description="特徴量CSVでRandomForestを学習し、交差検証や外部テキストでの評価を行います。"
    )
    p.add_argument("--csv", type=Path, default=CSV_PATH, help="入力CSVのパス（Build_RF_dataset.pyの出力）")
    p.add_argument(
        "--stats",
        type=str,
        default=",".join(SELECTED_STATS),
        help="使用する統計量名をカンマ区切りで指定（最大5つ）例: mean,std,iqr,rms,kurtosis",
    )
    p.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    p.add_argument("--max-depth", type=int, default=-1, help="-1 を指定すると None と同等")
    p.add_argument("--min-samples-split", type=int, default=MIN_SAMPLES_SPLIT)
    p.add_argument("--min-samples-leaf", type=int, default=MIN_SAMPLES_LEAF)
    p.add_argument(
        "--max-features",
        type=str,
        default=str(MAX_FEATURES),
        help='例: "sqrt" / "log2" / 0.5（特徴次元の割合）',
    )
    p.add_argument("--class-weight", type=str, default=str(CLASS_WEIGHT) if CLASS_WEIGHT is not None else "")
    p.add_argument("--cv-splits", type=int, default=CV_SPLITS, help="StratifiedKFold の分割数")
    p.add_argument("--cv-shuffle", type=int, default=int(CV_SHUFFLE), help="分割前にシャッフルするか (1=yes,0=no)")
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    p.add_argument("--n-jobs", type=int, default=N_JOBS)
    p.add_argument("--verbose", type=int, default=VERBOSE)
    # save model options
    default_save_path = (MODEL_SAVE_DIR / MODEL_SAVE_NAME) if SAVE_MODEL_BY_DEFAULT else None
    p.add_argument(
        "--save-model",
        type=Path,
        default=default_save_path,
        help="学習済みモデルを保存するパス（.joblib など）。未指定かつ SAVE_MODEL_BY_DEFAULT=True の場合は"
             " models/rf_bundle.joblib に保存します。",
    )
    # 評価時の設定
    p.add_argument("--eval-files", type=Path, nargs="*", help="外部評価に用いる *_set-*.txt を明示指定（複数可・ワイルドカード可）")
    p.add_argument("--eval-input-dir", type=Path, default=Path("Labelled_data"), help="外部評価用テキストのディレクトリ")
    p.add_argument("--eval-glob", type=str, default="*_set-*.txt", help="外部評価ファイルの検索グロブ")
    p.add_argument("--window-ms", type=int, default=250, help="スライディングウィンドウの幅[ms]")
    p.add_argument("--hop-ms", type=int, default=10, help="スライディングウィンドウのホップ[ms]")
    args = p.parse_args()

    # Apply CLI overrides to globals (for simplicity)
    N_ESTIMATORS = int(args.n_estimators)
    MAX_DEPTH = None if int(args.max_depth) == -1 else int(args.max_depth)
    MIN_SAMPLES_SPLIT = int(args.min_samples_split)
    MIN_SAMPLES_LEAF = int(args.min_samples_leaf)
    # Parse max_features which can be float, int, or string
    mf = args.max_features.strip()
    if mf.lower() in {"sqrt", "log2", "none"}:
        MAX_FEATURES = None if mf.lower() == "none" else mf.lower()
    else:
        try:
            if "." in mf:
                MAX_FEATURES = float(mf)
            else:
                MAX_FEATURES = int(mf)
        except ValueError:
            MAX_FEATURES = "sqrt"
    cw = args.class_weight.strip()
    CLASS_WEIGHT = None if cw == "" or cw.lower() == "none" else cw
    CV_SPLITS = int(args.cv_splits)
    CV_SHUFFLE = bool(int(args.cv_shuffle))
    RANDOM_STATE = int(args.random_state)
    N_JOBS = int(args.n_jobs)
    VERBOSE = int(args.verbose)

    stats = [s.strip() for s in args.stats.split(",") if s.strip()]
    if not stats:
        raise SystemExit("--stats には少なくとも1つの統計量名が必要です")
    if len(stats) > 5:
        print("注意: 統計量が6つ以上指定されました。先頭5つのみを使用します。")
        stats = stats[:5]

    X, y, feat_cols = load_dataset(args.csv, stats)

    # ---- Execution summary (datasets and hyperparameters) ----
    try:
        print("\n=== 実行設定サマリ ===")
        print(f"- 学習に使用したデータセット: {args.csv} (行={X.shape[0]}, 特徴次元={X.shape[1]})")
        print(f"- 使用統計量: {', '.join(stats)}")
        print(
            "- ハイパーパラメータ: "
            f"n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, "
            f"min_samples_split={MIN_SAMPLES_SPLIT}, min_samples_leaf={MIN_SAMPLES_LEAF}, "
            f"max_features={MAX_FEATURES}, class_weight={CLASS_WEIGHT}, "
            f"random_state={RANDOM_STATE}, n_jobs={N_JOBS}"
        )
    except Exception:
        pass

    # If external eval is specified, train on full CSV then evaluate
    eval_files: List[Path] = []
    if args.eval_files:
        from glob import glob as _glob

        # Expand potential wildcards for each provided path
        for p in args.eval_files:
            s = str(p)
            matches = list(Path().glob(s)) if any(ch in s for ch in "*?[]") else [Path(s)]
            for m in matches:
                if m.is_file():
                    eval_files.append(m)
    elif DEFAULT_EVAL_PATHS:
        eval_files = _expand_paths_from_strings(DEFAULT_EVAL_PATHS)
    else:
        # Fallback: use directory+glob (useful to quickly point to a folder)
        eval_files = sorted(p for p in args.eval_input_dir.glob(args.eval_glob) if p.is_file())

    # Report evaluation dataset selection
    try:
        if eval_files:
            print("- 精度評価に使用したデータセット(外部評価):")
            for p in eval_files:
                print(f"    - {p}")
            print(f"- 評価に使用したデータの設定: window={args.window_ms} ms, hop={args.hop_ms} ms")
        else:
            print(
                "- 精度評価に使用したデータセット: 交差検証 (StratifiedKFold) "
                f"splits={CV_SPLITS}, shuffle={CV_SHUFFLE}"
            )
    except Exception:
        pass

    if eval_files:
        print(f"[2/4] CSV全体で学習中（行={X.shape[0]}, 特徴次元={X.shape[1]}）…")
        rf = _train_rf_on_full(X, y)
        # Show feature importance ranking from the model trained on full CSV
        if hasattr(rf, "feature_importances_"):
            imp = rf.feature_importances_.astype(float)
            order = np.argsort(imp)[::-1]
            top_k = min(20, len(feat_cols))
            print("\n重要度による上位特徴（全データ学習モデル）:")
            print("   rank  importance       feature")
            for rank, idx in enumerate(order[:top_k], start=1):
                print(f"   {rank:>4d}   {imp[idx]:.6f}   {feat_cols[idx]}")

        # Save model if requested
        if args.save_model is not None:
            _save_model_bundle(args.save_model, rf, feat_cols, stats, X, y)
            print(f"Saved model bundle to: {args.save_model}")

        print("[3/4] 外部ファイルから評価用ウィンドウを作成 …")
        window_sec = float(args.window_ms) / 1000.0
        hop_sec = float(args.hop_ms) / 1000.0
        X_eval, y_eval = build_eval_matrix(eval_files, stats, window_sec, hop_sec)
        if X_eval.shape[0] == 0:
            print("外部ファイルに評価可能なウィンドウが見つかりませんでした。")
            return
        print(f"       評価ウィンドウ数: {X_eval.shape[0]} （特徴次元={X_eval.shape[1]}）")

        # Align feature columns order by selecting the same headers order
        expected_headers = feat_cols
        current_headers = feature_headers_for(stats)
        # current_headers and feat_cols should match by construction
        if len(current_headers) != len(expected_headers):
            print("Warning: feature header length mismatch between train and eval.")
        y_pred = rf.predict(X_eval)
        print("[4/4] 外部評価の結果:")
        acc = accuracy_score(y_eval, y_pred)
        print(f"正解率: {acc:.4f}")
        labels_unique = list(np.unique(np.concatenate([y_eval, y_pred])))
        print("\n分類レポート:\n" + classification_report(y_eval, y_pred, labels=labels_unique))
        cm = confusion_matrix(y_eval, y_pred, labels=labels_unique)
        print("混同行列（行=true, 列=pred）:")
        header = "      " + " ".join(f"{lbl:>15s}" for lbl in labels_unique)
        print(header)
        for i, row in enumerate(cm):
            print(f"{labels_unique[i]:>6s} " + " ".join(f"{v:15d}" for v in row))
    else:
        # No external eval files; run CV as before
        cross_validate_and_report(X, y, feat_cols)
        # If user requested saving a model but外部評価が無い場合、全データで改めて学習して保存
        if args.save_model is not None:
            print(f"[2/2] CSV全体で最終モデルを学習・保存（行={X.shape[0]}, 特徴次元={X.shape[1]}）…")
            rf = _train_rf_on_full(X, y)
            _save_model_bundle(args.save_model, rf, feat_cols, stats, X, y)
            print(f"Saved model bundle to: {args.save_model}")


if __name__ == "__main__":
    main()
