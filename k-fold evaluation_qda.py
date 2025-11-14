from __future__ import annotations

import argparse
import fnmatch
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

# ==== Build_RF_dataset から I/O と特徴量処理を再利用 ====
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


# ================= ユーザー設定（必要に応じて変更） =================
USER_SELECTED_STATS: List[str] = ["mean", "std", "max", "rms", "min","abs_sum"]
USER_OUT_DIR: Path = Path("models") / "multi_eval"
USER_WINDOW_MS: int = 250
USER_HOP_MS: int = 25
USER_FS_HZ: Optional[float] = 200.0  # None で自動推定

# もしファイルで持ちたい場合：明示特徴名、ワイルドカード可
MANUAL_FEATURES_FILE: Optional[Path] = None  # 例 Path("features.txt")


# ===== 手動指定ゾーン: ここに使いたい特徴量名を列挙（空なら従来どおり自動） =====
# 完全名: 例 "acc_x_mean" / "gyro_z_std"
# ワイルドカード可: 例 "acc_*_mean", "gyro_*"
MANUAL_FEATURES: List[str] = [
    # "encoder_angle_mean",
    # "quat_w_mean",
    # "grf_x_rms",
    # "grf_x_mean",
    # "grf_y_rms",
    # "acc_x_mean",
    # "acc_z_mean",
    # "gyro_x_mean",
    # "gyro_x_rms",
    # "gyro_y_rms",
    # "gyro_x_max",
    # "gyro_z_rms",
    # "quat_y_std",
    # "gyro_z_mean",
    # "acc_x_max",
    # "acc_x_min",
    # "acc_x_rms",
    # "quat_y_rms",
    # "grt_z_mean",
    # "grt_y_rms",
    # "grf_y_min",
    # "grt_x_min",
    # "grt_z_rms",
    # "grf_z_mean",
    # "gyro_y_mean",
    # "gyro_y_min",
    # "grf_y_std",
    # "grf_z_std",
    # "gyro_z_min",
    # "grf_x_std",
]

@dataclass
class Dataset:
    """学習/評価用の配列データを保持"""
    X: np.ndarray
    y: np.ndarray
    columns: List[str]
    t_sec: np.ndarray            # ウィンドウ代表時刻（秒）
    src: List[str]               # 由来ファイル名（N と同数）


def format_hp(params: Dict[str, object], keys: Sequence[str]) -> str:
    """表示したいキーだけ key=value の短い文字列に整形"""
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
    """モデルをシリアライズしたサイズ（バイト）を返す。失敗時は -1。"""
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
    """ワイルドカード（* ?）を candidates にマッチさせ、順番を保ってユニークで返す"""
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

    # サンプリング周波数の決定
    if fs_hz is None:
        if series.t_s.size >= 2:
            dt = float(np.median(np.diff(series.t_s)))
            fs_hz = (1.0 / dt) if dt > 0 else 100.0
        else:
            fs_hz = 100.0

    win_samp = max(int(round((window_ms / 1000.0) * fs_hz)), 2)
    hop_samp = max(int(round((hop_ms / 1000.0) * fs_hz)), 1)

    # 使用統計量は used_cols の「末尾一致」で逆推定（例: acc_x_abs_sum -> abs_sum）
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

    # 全ヘッダ（後で列選択用）
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


# ------------------ CV 用ユーティリティ ------------------

_SET_RE = re.compile(r"set-(\d+)")


def _extract_set_id(name: str) -> Optional[str]:
    m = _SET_RE.search(name)
    return m.group(1) if m else None


def find_pairs_by_subject(base_dir: Path, subject_keys: Sequence[str]) -> Dict[str, List[str]]:
    """Labelled_data 配下から、subject ごとに ramp/stair のペアが揃う set-XXXX を列挙"""
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


# ------------------ QDA モデル（スケーリング込み） ------------------

def train_qda_classifier(
    X: np.ndarray,
    y: np.ndarray,
    reg_param: float = 0.1,
    priors: Optional[np.ndarray] = None,
    store_covariance: bool = False,
    scaler: str = "standard",  # 追加: "standard" | "robust" | "none"
) -> Union[QuadraticDiscriminantAnalysis, Pipeline]:
    if X.size == 0:
        raise SystemExit("学習データが空です")

    # QDA 本体
    qda = QuadraticDiscriminantAnalysis(
        reg_param=reg_param,
        priors=priors,
        store_covariance=store_covariance,
    )

    # スケーラ選択
    steps = []
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    # "none" の場合はスケーラなし

    steps.append(("qda", qda))

    if len(steps) == 1:
        # スケーラなし（none）のときは素の QDA を返す
        qda.fit(X, y)
        return qda

    pipe = Pipeline(steps)
    pipe.fit(X, y)
    return pipe


def evaluate_classifier(
    clf: Union[QuadraticDiscriminantAnalysis, Pipeline],
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


# ------------------ メイン処理 ------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="kanoga/kaneishi それぞれで、ramp+stair を1ペアとする被験者ごとの k-fold CV（QDA+スケーリング対応）"
    )
    parser.add_argument("--labelled-dir", type=Path, default=Path("Labelled_data"), help="ラベル付きTXTのルート")
    parser.add_argument("--subjects", type=str, default="kanoga,kaneishi,murata", help="対象被験者（カンマ区切り）")
    parser.add_argument("--use-first-k", type=int, default=11, help="被験者ごとに使用する先頭ペア数（=fold の k）")
    parser.add_argument("--window-ms", type=int, default=USER_WINDOW_MS)
    parser.add_argument("--hop-ms", type=int, default=USER_HOP_MS)
    parser.add_argument("--fs-hz", type=float, default=USER_FS_HZ, help="サンプリング周波数（Hz）。None で自動推定")
    parser.add_argument("--out-dir", type=Path, default=USER_OUT_DIR, help="結果出力ベースディレクトリ")

    # 特徴量選択
    parser.add_argument("--manual-features", type=str, default=None, help="カンマ区切り列名またはワイルドカード（例: acc_*_mean）")
    parser.add_argument("--features-file", type=Path, default=MANUAL_FEATURES_FILE, help="列名を1行ずつ列挙したファイル")

    # QDA の主要ハイパーパラメータ
    parser.add_argument("--reg-param", type=float, default=0.0, help="正則化係数（0.0=正則化なし）")
    parser.add_argument("--store-covariance", action="store_true", help="学習後に共分散を保持（解析用途）")

    # 追加: スケーリングの選択
    parser.add_argument("--scaler", type=str, choices=["standard", "robust", "none"], default="standard",
                        help="前処理スケーラ（QDAはスケーリング推奨）")

    args = parser.parse_args()

    base_dir: Path = args.out_dir / "qda_eval"
    run_dir = base_dir
    _ensure_dir(run_dir)

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    # 特徴量列の決定
    if FEATURE_NAMES:
        auto_used_cols = _infer_used_cols_by_feature_names(FEATURE_NAMES, USER_SELECTED_STATS)
    else:
        auto_used_cols = []

    # 手動指定があれば優先
    manual_patterns: List[str] = []
    # 本ファイル内の静的リストも取り込む
    if MANUAL_FEATURES:
        manual_patterns.extend(MANUAL_FEATURES)
    if args.features_file:
        manual_patterns.extend(_read_features_file(args.features_file))
    if args.manual_features:
        manual_patterns.extend([s.strip() for s in args.manual_features.split(",") if s.strip()])

    if manual_patterns:
        candidates = _all_possible_cols(FEATURE_NAMES, STAT_FUNCS, USER_SELECTED_STATS)
        used_cols = _resolve_feature_patterns(candidates, manual_patterns)
        if not used_cols:
            raise SystemExit("手動指定の特徴量が空です。パターンを見直してください。")
        print(f"[CV] 特徴量列（手動指定）: {len(used_cols)} 列", flush=True)
    else:
        used_cols = auto_used_cols
        print(f"[CV] 特徴量列（自動: FEATURE_NAMES × USER_SELECTED_STATS）: {len(used_cols)} 列", flush=True)

    summary_lines: List[str] = []
    summary_lines.append(f"FEATURES\t{len(used_cols)}\t" + ",".join(used_cols))

    subject_to_sets = find_pairs_by_subject(args.labelled_dir, subjects)

    for subject, set_ids in subject_to_sets.items():
        if not set_ids:
            print(f"[CV:{subject}] 対象セットがありません", flush=True)
            continue
        k = min(args.use_first_k, len(set_ids))
        set_ids = set_ids[:k]
        print(f"[CV:{subject}] folds={k} sets={set_ids}", flush=True)

        fold_accs: List[float] = []
        fold_f1s: List[float] = []
        eval_dir = run_dir / subject
        _ensure_dir(eval_dir)

        for fold_idx in range(1, k + 1):
            test_set = set_ids[fold_idx - 1]
            train_sets = [s for s in set_ids if s != test_set]

            # データ構築
            train_dsets: List[Dataset] = []
            for sid in train_sets:
                ds = build_pair_dataset(
                    args.labelled_dir, subject, sid, used_cols, args.window_ms, args.hop_ms, args.fs_hz if not np.isnan(args.fs_hz) else None
                )
                train_dsets.append(ds)
            train_data = concat_datasets(train_dsets, used_cols)

            test_data = build_pair_dataset(
                args.labelled_dir, subject, test_set, used_cols, args.window_ms, args.hop_ms, args.fs_hz if not np.isnan(args.fs_hz) else None
            )

            # fold 固有のラベル順（安定化）
            labels_order = sorted(list(set(map(str, train_data.y)) | set(map(str, test_data.y))))

            # 学習（スケーリング付き）
            clf = train_qda_classifier(
                train_data.X, train_data.y,
                reg_param=args.reg_param,
                store_covariance=args.store_covariance,
                scaler=args.scaler,
            )
            size_info = f"{model_size_bytes(clf)} bytes"
            hp_str = format_hp(
                {
                    "reg_param": args.reg_param,
                    "scaler": args.scaler,
                    "window_ms": args.window_ms,
                    "hop_ms": args.hop_ms,
                    "fs_hz": args.fs_hz if args.fs_hz is not None else "auto",
                },
                keys=["reg_param", "scaler", "window_ms", "hop_ms", "fs_hz"]
            )

            # 評価
            acc, macro_f1, report, cm, _ = evaluate_classifier(clf, test_data.X, test_data.y, labels_order=labels_order)
            fold_accs.append(acc)
            fold_f1s.append(macro_f1)

            # 誤分類ログ
            y_pred = clf.predict(test_data.X)
            mis_idx = np.where(y_pred != test_data.y)[0]
            mis_out = eval_dir / f"CV_{subject}_fold{fold_idx}_misclassified.csv"
            with mis_out.open("w", encoding="utf-8") as fw:
                fw.write("t_sec,src,true_label,pred_label\n")
                for i in mis_idx:
                    fw.write(f"{test_data.t_sec[i]:.6f},{test_data.src[i]},{test_data.y[i]},{y_pred[i]}\n")
            print(f"  -> 誤分類ログ: {mis_out} | 件数={len(mis_idx)}", flush=True)

            print(f"[CV:{subject}] Fold {fold_idx}: acc={acc:.6f}, macroF1={macro_f1:.6f} | model size={size_info}", flush=True)

            # 詳細保存
            out_txt = eval_dir / f"CV_{subject}_fold{fold_idx}.txt"
            with out_txt.open("w", encoding="utf-8") as f:
                f.write(f"Subject: {subject}\n")
                f.write(f"Fold: {fold_idx}/{k}\n")
                f.write(f"Train sets: {', '.join(train_sets)}\n")
                f.write(f"Test set: {test_set}\n")
                f.write(f"Model size: {size_info}\n")
                f.write(f"Hyper params: {hp_str}\n")
                # QDA モデル情報（Pipeline でも可能な範囲で出力）
                try:
                    qda_obj = clf.named_steps["qda"] if isinstance(clf, Pipeline) else clf
                    f.write(f"Classes: {list(getattr(qda_obj, 'classes_', []))}\n")
                    f.write(f"n_features_in_: {getattr(qda_obj, 'n_features_in_', 'NA')}\n")
                    cov_info = getattr(qda_obj, 'covariance_', None)
                    if cov_info is not None:
                        if isinstance(cov_info, list):
                            f.write(f"covariance_: {len(cov_info)} class matrices\n")
                        else:
                            f.write(f"covariance_.shape: {np.asarray(cov_info).shape}\n")
                except Exception:
                    pass
                f.write(f"Window: {args.window_ms} ms, Hop: {args.hop_ms} ms, fs: {args.fs_hz if args.fs_hz is not None else 'auto'} Hz\n")
                f.write(f"Features ({len(used_cols)}): {', '.join(used_cols)}\n\n")
                f.write(f"Accuracy: {acc:.6f}\n")
                f.write(f"Macro-F1: {macro_f1:.6f}\n\n")
                f.write("Classification report:\n")
                f.write(report + "\n\n")
                f.write("Labels order:\n")
                f.write(", ".join(map(str, labels_order)) + "\n\n")
                f.write("Confusion matrix (row=true, col=pred):\n")
                for row in cm:
                    f.write("\t".join(str(int(v)) for v in row) + "\n")
            print(f"  -> 保存 {out_txt}", flush=True)

            # サマリ（fold単位）
            summary_lines.append(f"CVFOLD\t{subject}\t{fold_idx}\tacc\t{acc:.6f}")
            summary_lines.append(f"CVFOLD\t{subject}\t{fold_idx}\tmacroF1\t{macro_f1:.6f}")

        # 被験者ごとの平均
        if fold_accs:
            mean_acc = float(np.mean(fold_accs))
            mean_f1 = float(np.mean(fold_f1s))
            print(f"[CV:{subject}] 平均 acc={mean_acc:.6f}, macroF1={mean_f1:.6f}", flush=True)
            summary_lines.append(f"CVMEAN\t{subject}\tacc\t{mean_acc:.6f}")
            summary_lines.append(f"CVMEAN\t{subject}\tmacroF1\t{mean_f1:.6f}")

            # 棒グラフ（acc と F1）
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
                    ax.text(b.get_x() + b.get_width()/2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
            _add_labels(bars_acc)
            _add_labels(bars_f1)
            out_png = eval_dir / f"{subject}_acc_f1_bar.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[CV:{subject}] 可視化出力 {out_png}", flush=True)

    # サマリー
    with (run_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("# Run summary (QDA + scaling)\n")
        f.write(
            f"window_ms={args.window_ms}, hop_ms={args.hop_ms}, fs={args.fs_hz if args.fs_hz is not None else 'auto'}\n"
        )
        f.write(f"output_dir: {run_dir}\n\n")
        f.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
