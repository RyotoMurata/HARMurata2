from __future__ import annotations

import argparse
import fnmatch
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


# Build_RF_dataset からウィンドウ生成と特徴量計算を再利用
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


# ===== ユーザー設定（スクリプト先頭で編集） =====
USER_TRAIN_INPUT_SPECS: list[str] = [
    "Featured_data/rf_dataset_kanoga_all.csv",
    "Featured_data/rf_dataset_kaneishi_all.csv",
]
USER_TEST_INPUT_SPECS: list[str] = [
    "Labelled_data/ramp_t6_kanoga_set-0006.txt",
    "Labelled_data/ramp_t6_kaneishi_set-0006.txt",
    "Labelled_data/stair_t8_kanoga_set-0006.txt",
    "Labelled_data/stair_t6_kaneishi_set-0006.txt",
]
USER_FEATURES: list[str] = [
    "acc_x_*",
    "acc_y_*",
    "acc_z_*",
    "gyro_x_*",
    "gyro_y_*",
    "gyro_z_*",
    "quat_w_*",
    "quat_x_*",
    "quat_y_*",
    "quat_z_*",
    "encoder_angle_*",
    "grf_x_*",
    "grf_y_*",
    "grf_z_*",
    "grt_x_*",
    "grt_y_*",
    "grt_z_*",
]
USER_LABEL_COL: str = "label"
USER_EXCLUDE_COLS: list[str] = ["file", "set_id", "t_start_s", "t_end_s", USER_LABEL_COL]
# RandomForest のハイパーパラメータ
USER_N_ESTIMATORS: int = 100
USER_MAX_DEPTH: Optional[int] = 10  # None で無制限
USER_RANDOM_STATE: int = 0
USER_OUT_DIR: Path = Path("models") / "multi_eval"
# テキスト評価用のウィンドウ設定
USER_TEST_WINDOW_MS: int = 250
USER_TEST_HOP_MS: int = 25
USER_TEST_FS_HZ: Optional[float] = None
# Train_RF_from_csv.py と同じ既定の統計量（最大5種）
USER_SELECTED_STATS: List[str] = ["mean", "std", "max", "rms", "min"]


# ===== デフォルト設定 =====
DEFAULT_LABEL_COL = "label"
DEFAULT_EXCLUDE_COLS = ["file", "set_id", "t_start_s", "t_end_s", DEFAULT_LABEL_COL]
DEFAULT_OUT_DIR = Path("models") / "multi_eval"
DEFAULT_TEST_WINDOW_MS = 250
DEFAULT_TEST_HOP_MS = 10
DEFAULT_TEST_FS_HZ: Optional[float] = 200.0  # None なら推定
# ==========================


@dataclass
class Dataset:
    """学習/評価用に使う配列データを保持する簡易クラス"""

    X: np.ndarray
    y: np.ndarray
    columns: List[str]


def _read_csv(path: Path) -> pd.DataFrame:
    """CSVを読み込み（UTF-8を基本、必要に応じてフォールバック）"""
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8", errors="ignore")


def _sanitize_label_series(s: pd.Series) -> pd.Series:
    """ラベル列の体裁を整える（全体がクォートされている場合の剥がし等）"""

    def clean(v):
        if isinstance(v, str):
            t = v.strip()
            if len(t) >= 2 and ((t[0] == t[-1]) and t[0] in ('"', "'")):
                t = t[1:-1]
            t = t.replace('""', '"')
            return t
        return v

    return s.map(clean)


def _detect_numeric_feature_cols(df: pd.DataFrame, exclude_cols: Sequence[str]) -> List[str]:
    """数値列かつ除外対象ではない列名を返す"""
    excl = set(exclude_cols)
    return [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]


def _select_feature_cols(df: pd.DataFrame, include_patterns: Optional[Sequence[str]], exclude_cols: Sequence[str]) -> List[str]:
    """特徴量列を選択（パターン未指定なら自動検出）"""
    if not include_patterns:
        return _detect_numeric_feature_cols(df, exclude_cols)
    cols = [c for c in df.columns if c not in set(exclude_cols)]
    chosen: List[str] = []
    for pat in include_patterns:
        if any(ch in pat for ch in "*?["):
            for c in cols:
                if fnmatch.fnmatch(c, pat) and c not in chosen:
                    chosen.append(c)
        else:
            if pat in cols and pat not in chosen:
                chosen.append(pat)
    chosen = [c for c in chosen if pd.api.types.is_numeric_dtype(df[c])]
    return chosen


def _select_by_stats(columns: Sequence[str], stats: Sequence[str], exclude_cols: Sequence[str]) -> List[str]:
    """列名の接尾辞（統計量名）で特徴量列を選択（HOUSEKEEPING列を除外）。"""
    chosen = list(stats[:5])
    excl = set(exclude_cols)
    out: List[str] = []
    for c in columns:
        if c in excl:
            continue
        for s in chosen:
            if c.endswith(f"_{s}"):
                out.append(c)
                break
    return out

#特徴量を除外して、他の特徴量を使う
def _exclude_columns_by_patterns(cols: Sequence[str], patterns: Sequence[str]) -> List[str]:
    """fnmatchパターン（例 'acc_x_*'）に一致する列を除外した新しい列リストを返す"""
    pats = [p.strip() for p in patterns if p.strip()]
    out = []
    for c in cols:
        if any(fnmatch.fnmatch(c, p) for p in pats):
            continue
        out.append(c)
    return out


def load_dataset_from_csv(csv_paths: Sequence[Path], feature_cols: Optional[Sequence[str]], label_col: str, exclude_cols: Sequence[str]) -> Tuple[Dataset, List[str]]:
    """複数CSVを連結してDatasetを作成。"""
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    used_cols: Optional[List[str]] = None

    if feature_cols is None:
        stats = USER_SELECTED_STATS
        common_cols: Optional[set] = None
        for p in csv_paths:
            df = _read_csv(p)
            cols = set(_select_by_stats(df.columns, stats, exclude_cols))
            common_cols = cols if common_cols is None else (common_cols & cols)
        feature_cols = sorted(common_cols) if common_cols else []

    for p in csv_paths:
        df = _read_csv(p)
        if label_col not in df.columns:
            raise SystemExit(f"ラベル列 '{label_col}' が見つかりません: {p}")
        y = _sanitize_label_series(df[label_col]).to_numpy()
        cols = list(feature_cols) if feature_cols is not None else _select_by_stats(df.columns, USER_SELECTED_STATS, exclude_cols)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"必要な特徴量列が見つかりません {p}: {missing}")
        X = df[cols].to_numpy(dtype=float, copy=False)
        xs.append(X)
        ys.append(y)
        used_cols = cols

    X_all = np.concatenate(xs, axis=0) if xs else np.empty((0, 0), dtype=float)
    y_all = np.concatenate(ys, axis=0) if ys else np.empty((0,), dtype=object)
    if used_cols is None:
        used_cols = list(feature_cols) if feature_cols else []
    return Dataset(X=X_all, y=y_all, columns=used_cols), used_cols


def _infer_stats_from_used_cols(used_cols: Sequence[str]) -> List[str]:
    """使用中の列名から統計量名を推定"""
    if not used_cols:
        return list(BRD_DEFAULT_STATS) if BRD_DEFAULT_STATS else []
    known = set(STAT_FUNCS.keys()) if STAT_FUNCS else set()
    stats: List[str] = []
    for c in used_cols:
        if "_" not in c:
            continue
        base, suf = c.rsplit("_", 1)
        if suf in known:
            if base in FEATURE_NAMES and suf not in stats:
                stats.append(suf)
    if not stats and BRD_DEFAULT_STATS:
        return list(BRD_DEFAULT_STATS)
    return stats


def build_features_from_txt(txt_path: Path, used_cols: Sequence[str], window_ms: int, hop_ms: int, fs_hz: Optional[float]) -> Dataset:
    """*.txt（ラベル付き生ログ）からウィンドウ特徴量を生成"""
    if load_set_series is None or compute_window_features is None or sliding_windows_by_count is None:
        raise SystemExit("Build_RF_dataset.py の関数をインポートできませんでした。スクリプト配置を確認してください。")

    series = load_set_series(txt_path)
    # fs の決定
    if fs_hz is None:
        if series.t_s.size >= 2:
            dt = float(np.median(np.diff(series.t_s)))
            fs_hz = (1.0 / dt) if dt > 0 else 100.0
        else:
            fs_hz = 100.0

    win_samp = max(int(round((window_ms / 1000.0) * fs_hz)), 2)
    hop_samp = max(int(round((hop_ms / 1000.0) * fs_hz)), 1)

    # 学習で使っている統計量名を推定
    stat_names = _infer_stats_from_used_cols(used_cols)
    if not stat_names:
        stat_names = list(BRD_DEFAULT_STATS) if BRD_DEFAULT_STATS else ["mean", "std", "min", "max"]

    # ウィンドウ列挙
    windows = sliding_windows_by_count(series.t_s.size, win_samp, hop_samp)
    rows: List[List[float]] = []
    labels: List[str] = []

    # ヘッダ作成
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

    if not rows:
        return Dataset(X=np.empty((0, len(used_cols))), y=np.empty((0,), dtype=object), columns=list(used_cols))

    df_all = pd.DataFrame(rows, columns=all_headers)
    # 学習時の列順に揃える
    missing = [c for c in used_cols if c not in df_all.columns]
    if missing:
        raise SystemExit(f"テキストから生成した特徴量に不足があります。欠落列: {missing}")
    X = df_all[used_cols].to_numpy(dtype=float, copy=False)
    y = _sanitize_label_series(pd.Series(labels)).to_numpy()
    return Dataset(X=X, y=y, columns=list(used_cols))


def train_rf_classifier(X: np.ndarray, y: np.ndarray, n_estimators: int, max_depth: Optional[int], random_state: int) -> RandomForestClassifier:
    """RandomForest の学習"""
    if X.size == 0:
        raise SystemExit("学習データが空です")
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=random_state)
    clf.fit(X, y)
    return clf


def model_size_bytes(model) -> int:
    """モデルのシリアライズサイズ（バイト）を返す（失敗時は -1）"""
    try:
        data = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        return len(data)
    except Exception:
        return -1


def evaluate_classifier(
    clf: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    labels_order: Optional[Sequence[str]] = None
) -> Tuple[float, float, str, np.ndarray, List[str]]:
    """精度、Macro-F1、分類レポート、混同行列、ラベル順を返す"""
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


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="複数CSVからRFを学習し、複数CSV/テキストで評価するスクリプト")
    parser.add_argument(
        "--train-csvs",
        type=Path,
        nargs="*",
        default=[Path(s) for s in USER_TRAIN_INPUT_SPECS],
        help="学習用CSV。各CSV単独モデル＋全結合モデル。ファイル/ディレクトリ/ワイルドカード/@list 対応",
    )
    parser.add_argument(
        "--test-inputs",
        type=Path,
        nargs="*",
        default=[Path(s) for s in USER_TEST_INPUT_SPECS],
        help="評価用入力（CSVまたはラベル付き*.txt）。ファイル/ディレクトリ/ワイルドカード/@list 対応",
    )
    parser.add_argument("--label-col", type=str, default=USER_LABEL_COL, help="ラベル列名（CSV用）")
    parser.add_argument("--features", type=str, default=",".join(USER_FEATURES), help="使用する特徴量列名またはワイルドカード（例: 'acc_*'）。空なら共通の数値列を自動選択")
    parser.add_argument("--exclude-cols", type=str, default=",".join(USER_EXCLUDE_COLS), help="特徴量から除外する列（CSV用、カンマ区切り）")
    parser.add_argument("--n-estimators", type=int, default=USER_N_ESTIMATORS, help="RandomForest n_estimators")
    parser.add_argument("--max-depth", type=int, default=USER_MAX_DEPTH, help="RandomForest max_depth（Noneで無制限）")
    parser.add_argument("--random-state", type=int, default=USER_RANDOM_STATE, help="乱数シード")
    parser.add_argument("--out-dir", type=Path, default=USER_OUT_DIR, help="評価結果の出力先フォルダ")
    # テキスト評価用のウィンドウ設定
    parser.add_argument("--test-window-ms", type=int, default=USER_TEST_WINDOW_MS, help="テキスト入力のウィンドウ長[ms]")
    parser.add_argument("--test-hop-ms", type=int, default=USER_TEST_HOP_MS, help="テキスト入力のホップ長[ms]")
    parser.add_argument("--test-fs-hz", type=float, default=USER_TEST_FS_HZ, help="テキスト入力のサンプリング周波数[Hz]（省略時は推定）")

    args = parser.parse_args()

    # ---- パス展開ユーティリティ ----
    def _expand_paths(paths: Sequence[Path], allowed_suffixes: Sequence[str]) -> List[Path]:
        """ファイル/ディレクトリ/ワイルドカード/@list を展開し、許可拡張子のみ返す（順序維持・重複排除）"""
        wildcard_chars = set("*?[]")
        out: List[Path] = []
        seen = set()

        def add_path(p: Path):
            try:
                real = p.resolve()
            except Exception:
                return
            if not p.is_file():
                return
            if allowed_suffixes and p.suffix.lower() not in {s.lower() for s in allowed_suffixes}:
                return
            if real in seen:
                return
            seen.add(real)
            out.append(p)

        def add_from_string(s: str):
            # ワイルドカード
            if any(ch in s for ch in wildcard_chars):
                for m in Path().glob(s):
                    add_path(m)
                return
            p = Path(s)
            if p.is_dir():
                # 再帰的に許可拡張子を収集
                for m in p.rglob("*"):
                    if m.is_file() and (not allowed_suffixes or m.suffix.lower() in {x.lower() for x in allowed_suffixes}):
                        add_path(m)
            else:
                add_path(p)

        for p in paths:
            s = str(p)
            if s.startswith("@"):
                # リストファイル対応
                lst = Path(s[1:])
                try:
                    for line in lst.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        add_from_string(line)
                except FileNotFoundError:
                    raise SystemExit(f"パスリストファイルが見つかりません: {lst}")
            else:
                add_from_string(s)
        return out

    # 引数のパスを展開
    train_paths: List[Path] = _expand_paths(list(args.train_csvs), [".csv"])
    test_paths: List[Path] = _expand_paths(list(args.test_inputs), [".csv", ".txt"])
    print(f"[1/5] 入力解決: 学習CSV={len(train_paths)}件, テスト入力={len(test_paths)}件", flush=True)
    if not train_paths:
        raise SystemExit("学習入力が空です。先頭の USER_TRAIN_INPUT_SPECS か --train-csvs を設定してください。")
    if not test_paths:
        raise SystemExit("評価入力が空です。先頭の USER_TEST_INPUT_SPECS か --test-inputs を設定してください。")

    exclude_cols = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
    include_feats = [c.strip() for c in args.features.split(",") if c.strip()] if args.features else []

    # === 1) ハイパラ名フォルダ作成 ===
    # md_str = "None" if args.max_depth is None else str(args.max_depth)
    # run_dir_name = f"ne{args.n_estimators}_md{md_str}_rs{args.random_state}"
    # 変更後（末尾に _ablation を付加）
    md_str = "None" if args.max_depth is None else str(args.max_depth)
    run_dir_name = f"ne{args.n_estimators}_md{md_str}_rs{args.random_state}_permutation"

    run_dir = args.out_dir / run_dir_name
    _ensure_dir(run_dir)
    eval_dir = run_dir / "evals"
    _ensure_dir(eval_dir)
    print(f"[2/5] 出力先: {run_dir}（評価詳細: {eval_dir}）", flush=True)

    # 学習グループ（各CSV単独＋ALL）
    groups: List[Tuple[str, List[Path]]] = []
    for p in train_paths:
        groups.append((p.stem, [p]))
    if len(train_paths) >= 2:
        groups.append(("ALL", train_paths))
    print(f"[3/5] 学習グループ作成: {len(groups)}モデル（個別{len(train_paths)}＋ALL{'+1' if len(train_paths) >= 2 else ''}）", flush=True)
    print(f"[INFO] Hyperparams: n_estimators={args.n_estimators}, max_depth={args.max_depth}, random_state={args.random_state}", flush=True)

    # === 2) サマリー行の先頭にハイパラを記録 ===
    summary_lines: List[str] = []
    summary_lines.append(
        f"PARAMS\tn_estimators={args.n_estimators}\tmax_depth={args.max_depth}\trandom_state={args.random_state}"
    )

    # 特徴量列の事前決定（ユーザ指定があれば最初のCSVから解決）
    base_feature_cols: Optional[List[str]] = None
    if include_feats:
        first_df = _read_csv(train_paths[0])
        base_feature_cols = _select_feature_cols(first_df, include_feats, exclude_cols)
        if not base_feature_cols:
            raise SystemExit("--features で指定されたパターンに一致する列が見つかりません")
        print(f"[4/5] 特徴量（パターン指定）: {len(base_feature_cols)}列", flush=True)
         # ★ 方式B：ワイルドカード後に接尾辞で再フィルタして統一 ★
        base_feature_cols = _select_by_stats(base_feature_cols, USER_SELECTED_STATS, exclude_cols)
        if not base_feature_cols:
            raise SystemExit(
                "パターンには一致しましたが、USER_SELECTED_STATS に合う接尾辞の列がありません。"
                f"（使用統計={', '.join(USER_SELECTED_STATS)}）"
            )
        print(f"     ↳ 接尾辞フィルタ後: {len(base_feature_cols)}列 （{', '.join(USER_SELECTED_STATS)}）", flush=True)  
    else:
        print(f"[4/5] 特徴量（既定サフィックス）: {', '.join(USER_SELECTED_STATS)} を使用して共通列を選択", flush=True)

    total_models = len(groups)
    for gi, (model_name, paths) in enumerate(groups, start=1):
        # 学習データの読み込み
        train_data, used_cols = load_dataset_from_csv(paths, base_feature_cols, args.label_col, exclude_cols)
        print(f"[5/5] 学習 {gi}/{total_models}: {model_name}: X={train_data.X.shape}, y={train_data.y.shape}, 特徴量={len(used_cols)}列", flush=True)
        clf = train_rf_classifier(train_data.X, train_data.y, args.n_estimators, args.max_depth, args.random_state)
        size_b = model_size_bytes(clf)
        size_info = f"{size_b} B" if size_b >= 0 else "N/A"
        print(f"[モデル] {model_name}: サイズ={size_info}", flush=True)
        # 特徴量重要度
        importance_text_lines: List[str] = []
        if hasattr(clf, "feature_importances_"):
            imps = np.asarray(clf.feature_importances_, dtype=float)
            order = np.argsort(imps)[::-1]
            top_k = min(20, len(used_cols))
            print("\n上位特徴量（importance）:")
            print("   rank  importance        feature")
            for rank, idx in enumerate(order[:top_k], start=1):
                line = f"   {rank:>4d}   {imps[idx]:.6f}   {used_cols[idx]}"
                print(line)
                importance_text_lines.append(line.strip())
            print("")
        else:
            print("特徴量重要度は利用できません")

        summary_lines.append(f"MODEL {model_name}\tsize\t{size_b}")

        # # 各テスト入力で評価（CSVまたはTXTを自動判別）
        # for ti, test_p in enumerate(test_paths, start=1):
        #     if test_p.suffix.lower() == ".csv":
        #         test_data, _ = load_dataset_from_csv([test_p], used_cols, args.label_col, exclude_cols)
        #     elif test_p.suffix.lower() == ".txt":
        #         test_data = build_features_from_txt(test_p, used_cols, args.test_window_ms, args.test_hop_ms, args.test_fs_hz)
        #     else:
        #         raise SystemExit(f"未対応の拡張子です: {test_p}")
        #     acc, report, cm, labels_order = evaluate_classifier(clf, test_data.X, test_data.y)
        #     print(f"[評価] {gi}/{total_models} モデル {model_name} | テスト {ti}/{len(test_paths)} {test_p.stem}: acc={acc:.4f}", flush=True)
        #     print(report)
        #     print("混同行列（行=true, 列=pred）:")
        #     print(cm)

        #     # === 3) 評価結果をハイパラフォルダ内に保存（各ファイルにもハイパラ記載） ===
        #     out_txt = eval_dir / f"{model_name}__on__{test_p.stem}.txt"
        #     with out_txt.open("w", encoding="utf-8") as f:
        #         f.write(f"モデル: {model_name}\n")
        #         f.write(f"モデルサイズ: {size_info}\n")
        #         f.write(f"ハイパーパラメータ: n_estimators={args.n_estimators}, max_depth={args.max_depth}, random_state={args.random_state}\n")
        #         f.write(f"特徴量（{len(used_cols)}列）: {', '.join(used_cols)}\n")
        #         f.write(f"学習CSV: {', '.join(str(p) for p in paths)}\n")
        #         f.write(f"テスト入力: {test_p}\n")
        #         if test_p.suffix.lower() == ".txt":
        #             f.write(f"テスト窓: {args.test_window_ms} ms, ホップ: {args.test_hop_ms} ms, fs: {args.test_fs_hz if args.test_fs_hz is not None else '推定'} Hz\n")
        #         f.write(f"Accuracy: {acc:.6f}\n\n")
        #         f.write("Classification report:\n")
        #         f.write(report + "\n\n")
        #         f.write("Labels order:\n")
        #         f.write(", ".join(labels_order) + "\n\n")
        #         f.write("混同行列（行=true, 列=pred）:\n")
        #         for row in cm:
        #             f.write("\t".join(str(int(v)) for v in row) + "\n")
        #         if importance_text_lines:
        #             f.write("\n上位特徴量（importance）:\n")
        #             f.write("   rank  importance        feature\n")
        #             for line in importance_text_lines:
        #                 f.write(line + "\n")
        #         else:
        #             f.write("\n特徴量重要度は利用できません\n")
        #     print(f"  -> 保存: {out_txt}", flush=True)

        #     summary_lines.append(f"EVAL {model_name}\t{test_p.stem}\tacc\t{acc:.6f}")
        # ---- ベースライン評価（全特徴） ----
        baseline_per_test = []  # 後続の差分計算用に保持
        for ti, test_p in enumerate(test_paths, start=1):
            if test_p.suffix.lower() == ".csv":
                test_data, _ = load_dataset_from_csv([test_p], used_cols, args.label_col, exclude_cols)
            elif test_p.suffix.lower() == ".txt":
                test_data = build_features_from_txt(test_p, used_cols, args.test_window_ms, args.test_hop_ms, args.test_fs_hz)
            else:
                raise SystemExit(f"未対応の拡張子です: {test_p}")

            acc, macro_f1, report, cm, labels_order = evaluate_classifier(clf, test_data.X, test_data.y)
            print(f"[評価] {gi}/{total_models} モデル {model_name} | テスト {ti}/{len(test_paths)} {test_p.stem}: "
                f"acc={acc:.4f}, macroF1={macro_f1:.4f}", flush=True)
            print(report)
            print("混同行列（行=true, 列=pred）:")
            print(cm)

            out_txt = eval_dir / f"{model_name}__on__{test_p.stem}.txt"
            with out_txt.open("w", encoding="utf-8") as f:
                f.write(f"モデル: {model_name}\n")
                f.write(f"モデルサイズ: {size_info}\n")
                f.write(f"ハイパーパラメータ: n_estimators={args.n_estimators}, max_depth={args.max_depth}, random_state={args.random_state}\n")
                f.write(f"特徴量（{len(used_cols)}列）: {', '.join(used_cols)}\n")
                f.write(f"学習CSV: {', '.join(str(p) for p in paths)}\n")
                f.write(f"テスト入力: {test_p}\n")
                if test_p.suffix.lower() == ".txt":
                    f.write(f"テスト窓: {args.test_window_ms} ms, ホップ: {args.test_hop_ms} ms, fs: {args.test_fs_hz if args.test_fs_hz is not None else '推定'} Hz\n")
                f.write(f"Accuracy: {acc:.6f}\n")
                f.write(f"Macro-F1: {macro_f1:.6f}\n\n")
                f.write("Classification report:\n")
                f.write(report + "\n\n")
                f.write("Labels order:\n")
                f.write(", ".join(labels_order) + "\n\n")
                f.write("混同行列（行=true, 列=pred）:\n")
                for row in cm:
                    f.write("\t".join(str(int(v)) for v in row) + "\n")
                if importance_text_lines:
                    f.write("\n上位特徴量（importance）:\n")
                    f.write("   rank  importance        feature\n")
                    for line in importance_text_lines:
                        f.write(line + "\n")
                else:
                    f.write("\n特徴量重要度は利用できません\n")
            print(f"  -> 保存: {out_txt}", flush=True)

            summary_lines.append(f"EVAL {model_name}\t{test_p.stem}\tacc\t{acc:.6f}")
            summary_lines.append(f"EVAL {model_name}\t{test_p.stem}\tmacroF1\t{macro_f1:.6f}")

            baseline_per_test.append({
                "test_name": test_p.stem,
                "acc": acc,
                "macro_f1": macro_f1,
                "X": test_data.X,   # 追加
                "y": test_data.y    # 追加
            })
        # ★ この部分を既存のアブレーションループと置き換え ★
        # =========================================================
        print("\n[Permutation Importance] 開始: 各特徴量をシャッフルして精度変化を評価", flush=True)

        perm_rows = []
        # === 各特徴量を1列ずつシャッフルして再評価 ===
        rng = np.random.default_rng(args.random_state)
        for feat_idx, feat_name in enumerate(used_cols):
            print(f"[Perm] シャッフル: {feat_idx+1}/{len(used_cols)} {feat_name}", flush=True)
            for base in baseline_per_test:
                X_perm = base["X"].copy()
                # 1列だけシャッフル（対応関係を破壊）
                rng.shuffle(X_perm[:, feat_idx])
                acc_p, f1_p, _, _, _ = evaluate_classifier(clf, X_perm, base["y"])
                d_acc = acc_p - base["acc"]
                d_f1  = f1_p - base["macro_f1"]
                print(f"   {base['test_name']}: acc={acc_p:.4f} (Δ{d_acc:+.4f}), F1={f1_p:.4f} (Δ{d_f1:+.4f})", flush=True)

                perm_rows.append({
                    "model": model_name,
                    "feature": feat_name,
                    "test": base["test_name"],
                    "acc": acc_p,
                    "macro_f1": f1_p,
                    "delta_acc": d_acc,
                    "delta_macro_f1": d_f1,
                })

        # === 集計・保存 ===
        if perm_rows:
            df_perm = pd.DataFrame(perm_rows)
            out_csv = run_dir / f"permutation_importance_{model_name}.csv"
            df_perm.to_csv(out_csv, index=False, encoding="utf-8")
            print(f"\n[Permutation] 集計CSVを保存: {out_csv}", flush=True)

            # 平均ΔMacro-F1でランキング
            rank = (
                df_perm.groupby("feature")["delta_macro_f1"]
                .mean()
                .sort_values()  # Δがより負（悪化が大きい）ほど重要
            )
            print("\n[Permutation] 特徴量重要度ランキング（平均ΔMacro-F1）")
            for i, (feat, d) in enumerate(rank.items(), start=1):
                print(f"{i:>2d}. {feat:<25s}  mean ΔMacro-F1 = {d:+.4f}")
                # === Summary.txt にも重要度ランキングを追記 ===
            summary_lines.append(f"\n# Permutation Importance Ranking ({model_name})")
            top_n = 10  # 出力する上位特徴量の数（必要に応じて変更）
            for i, (feat, d) in enumerate(rank.items(), start=1):
                if i > top_n:
                    break
                summary_lines.append(f"RANK {i:02d}\t{model_name}\t{feat}\tmeanΔMacroF1\t{d:+.6f}")

        # =========================================================

    # === 2) サマリーをハイパラフォルダに保存（冒頭にハイパラを明記） ===
    with (run_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("# Run summary\n")
        f.write(f"hyperparams: n_estimators={args.n_estimators}, max_depth={args.max_depth}, random_state={args.random_state}\n")
        f.write(f"output_dir: {run_dir}\n\n")
        f.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
