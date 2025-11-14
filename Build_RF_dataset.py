from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ===== User-editable defaults =====
DEFAULT_INPUT_DIR = Path("Labelled_data")
DEFAULT_GLOB = "*_set-*.txt"
# 明示的に個別ファイルを指定したい場合はここに列挙（空なら下の input-dir/glob を使用）
# 例: ["Labelled_data/stair_t1_set-0001.txt", "Labelled_data/stair_t2_set-0002.txt"]
DEFAULT_INPUT_PATHS: List[str] = ["Labelled_data/ramp_t1_kaneishi_set-0001.txt", "Labelled_data/ramp_t2_kaneishi_set-0002.txt", "Labelled_data/ramp_t3_kaneishi_set-0003.txt"
                                  , "Labelled_data/ramp_t4_kaneishi_set-0004.txt", "Labelled_data/ramp_t5_kaneishi_set-0005.txt"]
DEFAULT_OUTPUT_CSV = Path("Featured_data") / "rf_dataset_kaneishi_ramp_h250.csv"
DEFAULT_WINDOW_MS: int = 250
DEFAULT_HOP_MS: int = 10
# Optional default sampling frequency [Hz]. If None, will estimate per file
DEFAULT_FS_HZ: Optional[float] = None
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
# ==================================


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


def parse_values_label_setid(line: str) -> Optional[Tuple[List[float], Optional[str], Optional[str]]]:
    """Parse numeric values (>=17), the quoted label, and SetID from a line.

    Returns (values[:17], label or None, set_id or None).
    """
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

    # Extract quoted segments
    quoted = re.findall(r'"([^"]+)"', rest)
    label: Optional[str] = None
    set_id: Optional[str] = None
    for q in quoted:
        if q.startswith("SetID="):
            set_id = q.split("=", 1)[1]
        else:
            if label is None:
                label = q

    # Remove the quoted parts to parse numbers safely
    if quoted:
        # Remove everything from the first quote onward
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
    return vals[:17], label, set_id


@dataclass
class SetSeries:
    file: Path
    set_id: Optional[str]
    t_s: np.ndarray  # seconds from first valid timestamp
    data: np.ndarray  # (N, 17)
    labels: List[Optional[str]]  # per-row labels


def load_set_series(path: Path) -> SetSeries:
    t0: Optional[datetime] = None
    times: List[float] = []
    rows: List[List[float]] = []
    labels: List[Optional[str]] = []
    set_id_seen: Optional[str] = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ts = parse_timestamp(line)
            if ts is None:
                continue
            parsed = parse_values_label_setid(line)
            if parsed is None:
                continue
            vals, lbl, sid = parsed
            if t0 is None:
                t0 = ts
            times.append((ts - t0).total_seconds())
            rows.append(vals)
            labels.append(lbl)
            if set_id_seen is None and sid:
                set_id_seen = sid
    if not rows:
        raise SystemExit(f"No valid rows in {path}")
    return SetSeries(
        file=path,
        set_id=set_id_seen,
        t_s=np.asarray(times, dtype=float),
        data=np.asarray(rows, dtype=float),
        labels=labels,
    )


# ---- stats ----
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
    return float(m4 / (sd ** 4))


def stat_zcr(arr: np.ndarray, t: np.ndarray) -> float:
    s = np.sign(arr)
    s[s == 0] = 1
    return float(np.sum(s[1:] * s[:-1] < 0))


def stat_abs_integral(arr: np.ndarray, t: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(np.abs(arr), t))
    else:
        return float(np.trapz(np.abs(arr), t))

#周波数特徴の追加
def stat_ssc(arr: np.ndarray, t: np.ndarray) -> float:
    """Slope Sign Changes count (EMGで一般的な定義)。微小ノイズ抑制のための閾値を内蔵。"""
    n = arr.size
    if n < 3:
        return 0.0
    # NaN安全化
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)

    # しきい値（微小ノイズ抑制用）: データスケールに合わせる
    # 例: 標準偏差の 1e-6 倍（完全0だと無視、単位に依存しない弱い閾値）
    sd = float(np.std(arr, ddof=0))
    thr = 1e-6 * sd

    a = arr[1:-1] - arr[:-2]
    b = arr[1:-1] - arr[2:]
    # (a * b) > thr なら傾きが符号反転したとみなす
    return float(np.sum((a * b) > thr))


def stat_abs_sum(arr: np.ndarray, t: np.ndarray) -> float:
    """Sum of absolute values in the window (time t is unused)."""
    return float(np.sum(np.abs(arr)))



STAT_FUNCS: Dict[str, callable] = {
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
    #周波数特徴
    "ssc": stat_ssc,  
    "abs_sum": stat_abs_sum,        # ← これを追加
}


def sliding_windows_by_count(n_samples: int, window_samples: int, hop_samples: int) -> List[Tuple[int, int]]:
    """Return list of (start_idx, end_idx) windows using sample counts.

    Windows are length `window_samples` and advance by `hop_samples`.
    The last start index is at most n_samples - window_samples.
    """
    if n_samples <= 0 or window_samples <= 0 or hop_samples <= 0:
        return []
    starts: List[int] = []
    ends: List[int] = []
    i = 0
    last_start = n_samples - window_samples
    while i <= last_start:
        a = i
        b = i + window_samples - 1
        # ensure at least 2 samples per window for integration-based stats
        if b >= a + 1:
            starts.append(a)
            ends.append(b)
        i += hop_samples
    return list(zip(starts, ends))


def label_at_window_median(labels: Sequence[Optional[str]]) -> Optional[str]:
    """Return the label at the median position in the window.

    If the median element is unlabeled (None), search outward to the nearest
    labeled element (by index/time). If none labeled, return None.
    For even-length windows, the lower-middle index is used as the median.
    """
    n = len(labels)
    if n == 0:
        return None
    mid = n // 2  # lower middle for even n
    if labels[mid] is not None:
        return labels[mid]
    # search outward from the median
    for off in range(1, n):
        for cand in (mid - off, mid + off):
            if 0 <= cand < n and labels[cand] is not None:
                return labels[cand]
    return None


def compute_window_features(
    data: np.ndarray,
    t: np.ndarray,
    stat_names: Sequence[str],
) -> List[float]:
    row: List[float] = []
    for col in range(data.shape[1]):
        x = data[:, col]
        for st in stat_names:
            row.append(float(STAT_FUNCS[st](x, t)))
    return row


def build_dataset_for_file(
    series: SetSeries,
    stat_names: Sequence[str],
    window_samples: int,
    hop_samples: int,
    *,
    progress_every: Optional[int] = None,
) -> Tuple[List[str], List[List[str]]]:
    # headers
    headers: List[str] = [
        "file",
        "set_id",
        "t_start_s",
        "t_end_s",
    ]
    for ch in FEATURE_NAMES:
        for st in stat_names:
            headers.append(f"{ch}_{st}")
    headers.append("label")

    rows_out: List[List[str]] = []
    windows = sliding_windows_by_count(series.t_s.size, window_samples, hop_samples)
    total_w = len(windows)
    for idx, (a, b) in enumerate(windows, start=1):
        t_sub = series.t_s[a : b + 1]
        d_sub = series.data[a : b + 1, :]
        lbl_sub = series.labels[a : b + 1]
        lbl = label_at_window_median(lbl_sub)
        if lbl is None:
            continue  # skip ambiguous or unlabeled windows
        feats = compute_window_features(d_sub, t_sub, stat_names)
        row: List[str] = [
            series.file.stem,
            series.set_id or "",
            f"{float(t_sub[0]):.6f}",
            f"{float(t_sub[-1]):.6f}",
        ]
        row.extend(f"{v:.10g}" for v in feats)
        row.append(f'"{lbl}"')
        rows_out.append(row)
        if progress_every and total_w > 0 and (idx % progress_every == 0 or idx == total_w):
            pct = int(round(100 * idx / total_w))
            print(f"    progress: {idx}/{total_w} ({pct}%)", flush=True)

    return headers, rows_out


def save_csv(headers: List[str], rows: List[List[str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")


def collect_files(input_dir: Path, glob_pat: str) -> List[Path]:
    return sorted(p for p in input_dir.glob(glob_pat) if p.is_file())


def expand_input_paths(paths: Sequence[Path]) -> List[Path]:
    """Expand wildcard patterns in given paths and filter existing files.

    Supports patterns like Labelled_data/*_set-000{1..5}.txt if expanded by shell,
    and raw '*' or '?' passed as-is by expanding via glob from CWD.
    """
    out: List[Path] = []
    wildcard_chars = set("*?[]")
    for p in paths:
        s = str(p)
        pth = Path(s)
        if any(ch in s for ch in wildcard_chars):
            out.extend(Path().glob(s))
        else:
            out.append(pth)
    # keep order; only files
    out2: List[Path] = []
    seen = set()
    for p in out:
        try:
            real = p.resolve()
        except Exception:
            continue
        if not p.is_file():
            continue
        if real in seen:
            continue
        seen.add(real)
        out2.append(p)
    return out2


def main() -> None:
    ap = argparse.ArgumentParser(description="Build RF training dataset from set-labeled files with sliding windows.")
    ap.add_argument("inputs", type=Path, nargs="*", help="Specific *_set-*.txt files to include (overrides --input-dir/--glob)" )
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Folder with *_set-*.txt files")
    ap.add_argument("--glob", type=str, default=DEFAULT_GLOB, help="Glob pattern to match input files")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV path")
    ap.add_argument("--window-ms", type=int, default=DEFAULT_WINDOW_MS, help="Window size in ms")
    ap.add_argument("--hop-ms", type=int, default=DEFAULT_HOP_MS, help="Hop size in ms")
    ap.add_argument("--fs-hz", type=float, default=DEFAULT_FS_HZ, help="Sampling frequency in Hz. If omitted, estimate per file from timestamps.")
    ap.add_argument(
        "--stats",
        type=str,
        default=",".join(DEFAULT_STATS),
        help="Comma-separated stats (e.g., mean,std,iqr). Available: "
        + ", ".join(sorted(STAT_FUNCS.keys())),
    )
    # majority threshold option removed; now use median label per window
    args = ap.parse_args()

    files: List[Path]
    if args.inputs:
        files = expand_input_paths(list(args.inputs))
    elif DEFAULT_INPUT_PATHS:
        files = expand_input_paths(list(DEFAULT_INPUT_PATHS))
    else:
        files = collect_files(args.input_dir, args.glob)
    if not files:
        if args.inputs:
            raise SystemExit("No input files matched the specified paths/patterns.")
        elif DEFAULT_INPUT_PATHS:
            raise SystemExit("No input files matched DEFAULT_INPUT_PATHS.")
        else:
            raise SystemExit(f"No input files found under {args.input_dir} matching {args.glob}")

    stat_names = [s.strip() for s in args.stats.split(",") if s.strip()]
    for name in stat_names:
        if name not in STAT_FUNCS:
            raise SystemExit(f"Unknown stat: {name}. Available: {', '.join(sorted(STAT_FUNCS))}")

    all_headers: Optional[List[str]] = None
    all_rows: List[List[str]] = []
    print(f"Found {len(files)} file(s) to process.")
    for i, fp in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Loading {fp} ...")
        series = load_set_series(fp)
        try:
            dur = float(series.t_s[-1]) if series.t_s.size else 0.0
        except Exception:
            dur = 0.0
        # determine sampling frequency
        fs_hz: Optional[float] = args.fs_hz
        if fs_hz is None:
            # estimate from timestamps per file
            if series.t_s.size >= 2:
                dt = np.median(np.diff(series.t_s))
                fs_hz = float(1.0 / dt) if dt > 0 else None
            else:
                fs_hz = None
        # compute window/hop in samples (fallbacks ensure valid integers)
        if fs_hz is None:
            print("    warning: --fs-hz not provided and estimation failed; assuming 100 Hz", flush=True)
            fs_hz = 200.0 #サンプリング周波数の設定
        win_samp = max(int(round((args.window_ms / 1000.0) * fs_hz)), 2)
        hop_samp = max(int(round((args.hop_ms / 1000.0) * fs_hz)), 1)
        print(
            f"    loaded: samples={series.t_s.size}, duration={dur:.2f}s, set_id={series.set_id or ''}, fs~{fs_hz:.3f} Hz, win={win_samp} samp, hop={hop_samp} samp"
        )
        print("    building windows and features ...")
        headers, rows = build_dataset_for_file(series, stat_names, win_samp, hop_samp, progress_every=200)
        if all_headers is None:
            all_headers = headers
        all_rows.extend(rows)
        print(f"    done: produced {len(rows)} row(s)")

    if all_headers is None:
        raise SystemExit("No windows produced; check inputs and parameters.")
    save_csv(all_headers, all_rows, args.output)
    print(f"Saved: {args.output}  (rows={len(all_rows)})")


if __name__ == "__main__":
    main()
