from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ======== ユーザー編集用デフォルト (ここを書き換えるだけでOK) ========
# 入力ファイル（Labelled_data で作成した1ファイル）
# DEFAULT_INPUT: Path = Path("Labelled_data\ramp_t1_set-0001.txt")

# Windows のバックスラッシュによるエスケープ（"\r" 等）を避けるために安全な結合で再定義
DEFAULT_INPUT = (Path("Labelled_data") / "stair_t5_kaneishi_set-0005.txt")

# ウィンドウ長[ms] と ホップ長[ms]
DEFAULT_WINDOW_MS: int = 1000
DEFAULT_HOP_MS: int = 100

# 計算する統計量（複数選択可）
# 利用可能: mean, std, min, max, range, median, iqr, rms, skewness, kurtosis, zcr, abs_integral
DEFAULT_STATS: List[str] = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "median",
    "iqr",
    "rms",
    "skewness",
    "kurtosis",
    "zcr",
    "abs_integral",
]

# 出力先フォルダ（None の場合は入力ファイルと同じフォルダ）
DEFAULT_OUTPUT_DIR: Optional[Path] = Path("Featured_data")
# ===================================================================


# 出力ファイル名テンプレート（先頭で指定可能）
# {stem}: 入力ファイル名（拡張子なし）
# {w}, {h}: 窓幅/ホップ長 [ms]
DEFAULT_OUTPUT_NAME_FORMAT: str = "{stem}+Featured_w{w}ms_h{h}ms.txt"

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
    "power",  # derived: encoder_angle * grt_x
]


def parse_timestamp(line: str) -> Optional[datetime]:
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
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None


def parse_values_and_label(line: str) -> Optional[Tuple[List[float], Optional[str]]]:
    """
    先頭のタイムスタンプの後に続く数値列(>=17)と末尾の "ラベル" を抽出。
    ラベルは無い/うまく切り出せない場合 None。
    """
    try:
        idx = line.index("]")
    except ValueError:
        return None
    rest = line[idx + 1 :].strip()
    # 元ログ由来のノイズ除去
    rest = (
        rest.replace("uart:~$", " ")
        .replace("uart:$", " ")
        .replace("uart:", " ")
        .strip()
    )
    label: Optional[str] = None
    # 末尾の "..." を取り出す（含まれていれば）
    qpos = rest.find('"')
    if qpos >= 0:
        label_part = rest[qpos:].strip()
        rest = rest[:qpos].strip()
        if label_part.startswith('"') and label_part.endswith('"'):
            label = label_part.strip('"')
        else:
            # 引用符が分割されていても先頭だけあれば良しとする
            try:
                label = label_part.split('"')[1]
            except Exception:
                label = None

    vals: List[float] = []
    for tok in rest.split():
        try:
            vals.append(float(tok))
        except ValueError:
            return None
    if len(vals) < 17:
        return None
    return vals[:17], label


@dataclass
class LabeledSeries:
    t_s: np.ndarray  # 相対時刻[s]
    data: np.ndarray  # 形状: (N, 18) with derived 'power' appended
    label: Optional[str]


def load_labeled_series(path: Path) -> LabeledSeries:
    t0: Optional[datetime] = None
    times: List[float] = []
    rows: List[List[float]] = []
    first_label: Optional[str] = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ts = parse_timestamp(line)
            if ts is None:
                continue
            parsed = parse_values_and_label(line)
            if parsed is None:
                continue
            vals, lbl = parsed
            if t0 is None:
                t0 = ts
            times.append((ts - t0).total_seconds())
            rows.append(vals)
            if first_label is None and lbl:
                first_label = lbl
    if not rows:
        raise SystemExit(f"No valid rows in {path}")
    base = np.asarray(rows, dtype=float)
    # derived power column from encoder_angle (col 10) and grt_x (col 14)
    try:
        power_col = (base[:, 10] * base[:, 14]).reshape(-1, 1)
        data = np.concatenate([base, power_col], axis=1)
    except Exception:
        data = base
    return LabeledSeries(np.asarray(times, dtype=float), data, first_label)


# 統計量関数群（arr は 1次元配列, t は同じ長さの時刻[s]）
def stat_mean(arr: np.ndarray, t: np.ndarray) -> float:
    return float(np.mean(arr))


def stat_std(arr: np.ndarray, t: np.ndarray) -> float:
    return float(np.std(arr, ddof=0))


def stat_min(arr: np.ndarray, t: np.ndarray) -> float:
    return float(np.min(arr))


def stat_max(arr: np.ndarray, t: np.ndarray) -> float:
    return float(np.max(arr))


def stat_range(arr: np.ndarray, t: np.ndarray) -> float:
    return float(np.max(arr) - np.min(arr))


def stat_median(arr: np.ndarray, t: np.ndarray) -> float:
    return float(np.median(arr))


def stat_iqr(arr: np.ndarray, t: np.ndarray) -> float:
    q75 = np.percentile(arr, 75)
    q25 = np.percentile(arr, 25)
    return float(q75 - q25)


def stat_rms(arr: np.ndarray, t: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(arr))))


def stat_skewness(arr: np.ndarray, t: np.ndarray) -> float:
    mu = np.mean(arr)
    sd = np.std(arr, ddof=0)
    if sd == 0:
        return 0.0
    m3 = np.mean((arr - mu) ** 3)
    return float(m3 / (sd ** 3))


def stat_kurtosis(arr: np.ndarray, t: np.ndarray) -> float:
    mu = np.mean(arr)
    sd = np.std(arr, ddof=0)
    if sd == 0:
        return 0.0
    m4 = np.mean((arr - mu) ** 4)
    return float(m4 / (sd ** 4))  # Pearson (excessは未使用)


def stat_zcr(arr: np.ndarray, t: np.ndarray) -> float:
    # 符号が変わる回数（ゼロ通過）をカウント
    s = np.sign(arr)
    s[s == 0] = 1  # 0 を正として扱い不要な増加を回避
    return float(np.sum(s[1:] * s[:-1] < 0))


def stat_abs_integral(arr: np.ndarray, t: np.ndarray) -> float:
    # 台形則で ∫|x| dt を近似
    if len(arr) < 2:
        return 0.0
    # NumPy 2.0+ では np.trapz は非推奨。np.trapezoid が推奨。
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(np.abs(arr), t))
    else:
        return float(np.trapz(np.abs(arr), t))


STAT_FUNCS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
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


def sliding_windows(t: np.ndarray, window_sec: float, hop_sec: float) -> List[Tuple[int, int]]:
    """
    時刻配列 t（秒, 昇順）から、[start_idx, end_idx]（endは含む）を返す。
    """
    if t.size == 0:
        return []
    starts: List[int] = []
    ends: List[int] = []
    n = t.size
    t_start = t[0]
    t_end_total = t[-1]
    current = t_start
    i0 = 0
    i1 = 0
    while current + window_sec <= t_end_total + 1e-9:
        # start index
        while i0 < n and t[i0] < current - 1e-12:
            i0 += 1
        # end index: t <= current + window_sec
        wend = current + window_sec + 1e-12
        i1 = i0
        while i1 < n and t[i1] <= wend:
            i1 += 1
        if i1 - i0 >= 2:  # 最低2サンプル
            starts.append(i0)
            ends.append(i1 - 1)
        current += hop_sec
    return list(zip(starts, ends))


def compute_features(series: LabeledSeries, stat_names: Sequence[str], window_sec: float, hop_sec: float) -> Tuple[List[str], np.ndarray]:
    # 検証: 統計量名
    for name in stat_names:
        if name not in STAT_FUNCS:
            raise SystemExit(f"Unknown stat: {name}. Available: {', '.join(sorted(STAT_FUNCS))}")

    idx_pairs = sliding_windows(series.t_s, window_sec, hop_sec)
    if not idx_pairs:
        raise SystemExit("No windows generated. Check window/hop sizes and input length.")

    # ヘッダ（列名）作成
    headers: List[str] = ["t_start_s", "t_end_s"]
    for ch in FEATURE_NAMES:
        for st in stat_names:
            headers.append(f"{ch}_{st}")
    # ラベル列（存在する場合）
    label_exists = series.label is not None
    if label_exists:
        headers.append("label")

    out: List[List[float]] = []
    for (a, b) in idx_pairs:
        t_sub = series.t_s[a : b + 1]
        row: List[float] = [float(t_sub[0]), float(t_sub[-1])]
        for col in range(series.data.shape[1]):
            x = series.data[a : b + 1, col]
            for st in stat_names:
                val = STAT_FUNCS[st](x, t_sub)
                row.append(float(val))
        if label_exists:
            # 同一ファイルからのウィンドウであれば同一ラベル想定
            # 文字列列は後で別枠で扱うため placeholder を使わず、ここでは追加をスキップし
            # 別配列に文字列として持つこともできるが、簡便のため最後に文字列結合で出力
            pass
        out.append(row)

    out_arr = np.asarray(out, dtype=float)
    return headers, out_arr


def save_features(headers: List[str], values: np.ndarray, dst: Path, label: Optional[str] = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # テキスト（CSV）として保存。末尾にラベル列が必要なら付与。
    with dst.open("w", encoding="utf-8", newline="") as f:
        # ヘッダー
        f.write(",".join(headers) + "\n")
        for i in range(values.shape[0]):
            row = ",".join(f"{v:.10g}" for v in values[i])
            if label is not None:
                row = row + "," + f'"{label}"'
            f.write(row + "\n")


def main():
    p = argparse.ArgumentParser(description="Labelled_data の1ファイルからスライディングウィンドウ特徴量を計算")
    p.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT, help="入力ファイル (既定: DEFAULT_INPUT)")
    p.add_argument("--window-ms", type=int, default=DEFAULT_WINDOW_MS, help="ウィンドウ長 [ms]")
    p.add_argument("--hop-ms", type=int, default=DEFAULT_HOP_MS, help="ホップ長 [ms]")
    p.add_argument(
        "--stats",
        type=str,
        default=",".join(DEFAULT_STATS),
        help="カンマ区切りの統計量名 (例: mean,std,iqr,rms). 利用可能: "
        + ", ".join(sorted(STAT_FUNCS.keys())),
    )
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR, help="出力ディレクトリ (None で入力と同じ)")
    args = p.parse_args()

    in_path: Path = args.input
    series = load_labeled_series(in_path)

    stat_names = [s.strip() for s in args.stats.split(",") if s.strip()]
    window_sec = float(args.window_ms) / 1000.0
    hop_sec = float(args.hop_ms) / 1000.0
    headers, values = compute_features(series, stat_names, window_sec, hop_sec)

    out_dir = args.outdir if args.outdir is not None else in_path.parent
    # 出力ファイル名: テンプレートから生成
    out_name = DEFAULT_OUTPUT_NAME_FORMAT.format(
        stem=in_path.stem,
        w=int(args.window_ms),
        h=int(args.hop_ms),
    )
    out_path = out_dir / out_name
    save_features(headers, values, out_path, label=series.label)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
