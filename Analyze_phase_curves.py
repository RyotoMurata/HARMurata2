from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm, rcParams as _rc

# ===== Display (JP font fallback) =====
_JP_FONT_CANDIDATES = [
    "Yu Gothic","Yu Gothic UI","Meiryo","MS Gothic",
    "Noto Sans CJK JP","IPAGothic","IPAexGothic",
]
for _name in _JP_FONT_CANDIDATES:
    try:
        _path = _fm.findfont(_name, fallback_to_default=False)
        if _path:
            _rc["font.family"] = _name
            break
    except Exception:
        pass
_rc["axes.unicode_minus"] = False

# ===== Data channels (17 columns) =====
FEATURE_NAMES: List[str] = [
    "acc_x","acc_y","acc_z",
    "gyro_x","gyro_y","gyro_z",
    "quat_w","quat_x","quat_y","quat_z",
    "encoder_angle",
    "grf_x","grf_y","grf_z",
    "grt_x","grt_y","grt_z",
]
NAME_TO_INDEX: Dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}
FEATURE_UNITS: Dict[str, str] = {
    "acc_x":"g","acc_y":"g","acc_z":"g",
    "gyro_x":"deg/s","gyro_y":"deg/s","gyro_z":"deg/s",
    "quat_w":"","quat_x":"","quat_y":"","quat_z":"",
    "encoder_angle":"deg",
    "grf_x":"N","grf_y":"N","grf_z":"N",
    "grt_x":"N·m","grt_y":"N·m","grt_z":"N·m",
}

# ===== I/O parsing =====
from datetime import datetime
import re

def parse_timestamp(line: str) -> Optional[datetime]:
    line = line.strip()
    if not line.startswith("["):
        return None
    try:
        idx = line.index("]")
    except ValueError:
        return None
    ts_str = line[1:idx].strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f","%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None

def parse_values_label_setid(line: str) -> Optional[Tuple[List[float], Optional[str], Optional[str]]]:
    try:
        idx = line.index("]")
    except ValueError:
        return None
    rest = line[idx + 1:].strip()
    rest = rest.replace("uart:~$"," ").replace("uart:$"," ").replace("uart:"," ").strip()

    quoted = re.findall(r'"([^"]+)"', rest)
    label: Optional[str] = None
    set_id: Optional[str] = None
    for q in quoted:
        if q.startswith("SetID="):
            set_id = q.split("=",1)[1]
        else:
            if label is None:
                label = q
    if quoted:
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
    t_s: np.ndarray
    data: np.ndarray
    labels: List[Optional[str]]

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
    return SetSeries(path, set_id_seen, np.asarray(times, float), np.asarray(rows, float), labels)

# ===== Subject handling =====
def extract_subject(fp: Path) -> str:
    """例: ramp_t10_kaneishi_set-0010.txt -> 'kaneishi'"""
    m = re.search(r"_([A-Za-z]+)_set-\d+", fp.name)
    return (m.group(1).lower() if m else "unknown")

# ===== Phase segmentation =====
class Phase:
    STANCE = "stance"
    SWING  = "swing"

@dataclass
class Segment:
    label: Optional[str]
    phase: str
    t: np.ndarray
    X: np.ndarray
    subject: str

def _segment_phase_in_region(t: np.ndarray,
                             grf_z: np.ndarray,
                             X: np.ndarray,
                             label: Optional[str],
                             low_thr: float,
                             high_thr: float,
                             subject: str) -> List[Segment]:
    segs: List[Segment] = []
    if t.size == 0:
        return segs
    in_stance = bool(grf_z[0] > high_thr)
    start_idx = 0
    for i in range(1, t.size):
        if in_stance:
            if grf_z[i] <= low_thr and grf_z[i-1] > low_thr:
                if i - 1 > start_idx:
                    segs.append(Segment(label, Phase.STANCE, t[start_idx:i], X[start_idx:i,:], subject))
                in_stance = False
                start_idx = i
        else:
            if grf_z[i] >= high_thr and grf_z[i-1] < high_thr:
                if i - 1 > start_idx:
                    segs.append(Segment(label, Phase.SWING, t[start_idx:i], X[start_idx:i,:], subject))
                in_stance = True
                start_idx = i
    if start_idx < t.size - 1:
        segs.append(Segment(label, Phase.STANCE if in_stance else Phase.SWING, t[start_idx:], X[start_idx:,:], subject))
    return segs

def segment_by_label(series: SetSeries, low_thr: float, high_thr: float, subject: str) -> List[Segment]:
    t = series.t_s
    X = series.data
    labels = series.labels
    grf_z = X[:, NAME_TO_INDEX["grf_z"]]
    segs: List[Segment] = []
    if t.size == 0:
        return segs
    start = 0
    curr = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != curr:
            segs.extend(_segment_phase_in_region(t[start:i], grf_z[start:i], X[start:i,:], curr, low_thr, high_thr, subject))
            start = i
            curr = labels[i]
    segs.extend(_segment_phase_in_region(t[start:], grf_z[start:], X[start:,:], curr, low_thr, high_thr, subject))
    return segs

# ===== Normalization & helpers =====
def _canon(lbl: Optional[str]) -> str:
    return (lbl or "").strip().lower().replace(" ", "-").replace("_", "-")

def normalize_and_resample(seg: Segment, n_points: int) -> Tuple[str, str, str, np.ndarray, np.ndarray]:
    t = seg.t
    if t.size < 2:
        raise ValueError("segment too short")
    t0 = float(t[0]); t1 = float(t[-1])
    if t1 <= t0:
        raise ValueError("non-increasing segment time")
    tau = (t - t0) / (t1 - t0)
    xi = np.linspace(0.0, 1.0, n_points)
    Y = np.empty((n_points, seg.X.shape[1]), dtype=float)
    for j in range(seg.X.shape[1]):
        Y[:, j] = np.interp(xi, tau, seg.X[:, j])
    return (seg.subject, _canon(seg.label), seg.phase, xi * 100.0, Y)

# 95%CI 用の簡易 t 値（0.975点）
def _t975(n: int) -> float:
    if n <= 1: return 1.0
    table = {2:12.71,3:4.30,4:3.18,5:2.78,6:2.57,7:2.45,8:2.36,9:2.31,10:2.26,
             12:2.18,15:2.13,20:2.09,25:2.06,30:2.04}
    if n in table: return table[n]
    if n < 30:
        keys = sorted(table.keys())
        lo = max(k for k in keys if k < n); hi = min(k for k in keys if k > n)
        return table[lo] + (table[hi]-table[lo])*(n-lo)/(hi-lo)
    return 1.96

# ===== Aggregation (with multiple band stats) =====
def aggregate_by_label_phase(samples: List[Tuple[str, str, np.ndarray, np.ndarray]]) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    """
    samples: List[(label, phase, x_pct(P,), Y(P,17))]
    return: dict[(label, phase)] -> stats dict (mean, min/max, std, se, p25/p75, p10/p90, ci95, median±MAD, count)
    """
    buckets: Dict[Tuple[str, str], List[np.ndarray]] = {}
    x_ref: Dict[Tuple[str, str], np.ndarray] = {}
    for label, phase, x_pct, Y in samples:
        key = (_canon(label), phase)
        buckets.setdefault(key, []).append(Y)
        if key not in x_ref:
            x_ref[key] = x_pct

    out: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for key, arrs in buckets.items():
        stack = np.stack(arrs, axis=0)  # (N, P, 17)
        n = stack.shape[0]
        mean = np.nanmean(stack, axis=0)
        std  = np.nanstd(stack, axis=0, ddof=1 if n > 1 else 0)
        se   = std / np.sqrt(n) if n > 0 else std
        p25  = np.nanpercentile(stack, 25, axis=0)
        p75  = np.nanpercentile(stack, 75, axis=0)
        p10  = np.nanpercentile(stack, 10, axis=0)
        p90  = np.nanpercentile(stack, 90, axis=0)
        tval = _t975(n)
        ci95_low  = mean - tval * se
        ci95_high = mean + tval * se
        median = np.nanmedian(stack, axis=0)
        mad    = np.nanmedian(np.abs(stack - median), axis=0)
        mad_lo = median - 1.4826 * mad
        mad_hi = median + 1.4826 * mad

        out[key] = {
            "x_percent": x_ref[key],
            "mean": mean,
            "min":  np.nanmin(stack, axis=0),
            "max":  np.nanmax(stack, axis=0),
            "std":  std,
            "se":   se,
            "p25":  p25, "p75": p75,
            "p10":  p10, "p90": p90,
            "ci95_low": ci95_low, "ci95_high": ci95_high,
            "median": median, "mad_lo": mad_lo, "mad_hi": mad_hi,
            "count": np.array([n], dtype=int),
        }
    return out

# ===== Plotting =====
STYLE_BY_LABEL: Dict[str, Tuple[str, Optional[str]]] = {
    "stair-ascent":  ("-.", "tab:red"),
    "stair-descent": ("--", "tab:brown"),
    "ramp-ascent":   ("-.", "tab:green"),
    "ramp-descent":  (":",  "tab:purple"),
    "stop":          ("-",  "tab:gray"),
    "level-walk":    ("-",  None),
}
LABEL_ORDER: List[str] = [
    "stair-ascent","stair-descent","ramp-ascent","ramp-descent","level-walk","stop",
]

def choose_style_for_label(label: str) -> Tuple[str, Optional[str]]:
    k = _canon(label)
    return STYLE_BY_LABEL.get(k, ("-", None))

def plot_feature_curves(agg: Dict[Tuple[str, str], Dict[str, np.ndarray]],
                        feature: str,
                        phase_filter: Optional[str],
                        save_dir: Optional[Path],
                        labels_wanted: Optional[List[str]] = None,
                        band: str = "iqr") -> None:
    if feature not in NAME_TO_INDEX:
        raise SystemExit(f"Unknown feature: {feature}")
    j = NAME_TO_INDEX[feature]
    unit = FEATURE_UNITS.get(feature, "")

    if (labels_wanted is None or len(labels_wanted) == 0 or
        (len(labels_wanted) == 1 and labels_wanted[0].lower() == "all")):
        labels_seq = LABEL_ORDER[:]
    else:
        labels_seq = [_canon(lbl) for lbl in labels_wanted if lbl.strip()]

    phases = [phase_filter] if phase_filter else sorted({k[1] for k in agg.keys()})
    title_band = {
        "minmax":"min–max","iqr":"IQR (25–75%)","p10-90":"p10–p90",
        "std":"±1σ","se":"±SE","ci95":"95% CI","mad":"median±1.4826·MAD"
    }.get(band, "IQR (25–75%)")

    for ph in phases:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        any_plotted = False
        for lbl in labels_seq:
            key = (lbl, ph)
            if key not in agg:
                continue
            entry = agg[key]
            x = entry["x_percent"]
            y_mean = entry["mean"][:, j]

            # 帯の上下
            if band == "minmax":
                lo, hi = entry["min"][:, j], entry["max"][:, j]
            elif band == "iqr":
                lo, hi = entry["p25"][:, j], entry["p75"][:, j]
            elif band == "p10-90":
                lo, hi = entry["p10"][:, j], entry["p90"][:, j]
            elif band == "std":
                lo, hi = y_mean - entry["std"][:, j], y_mean + entry["std"][:, j]
            elif band == "se":
                lo, hi = y_mean - entry["se"][:, j],  y_mean + entry["se"][:, j]
            elif band == "ci95":
                lo, hi = entry["ci95_low"][:, j], entry["ci95_high"][:, j]
            elif band == "mad":
                lo, hi = entry["mad_lo"][:, j], entry["mad_hi"][:, j]
            else:  # fallback
                lo, hi = entry["p25"][:, j], entry["p75"][:, j]

            ls, color = choose_style_for_label(lbl)
            ax.plot(x, y_mean, linestyle=ls, color=color, label=f"{lbl} (n={int(entry['count'][0])})")
            ax.fill_between(x, lo, hi, color=color, alpha=0.15)
            any_plotted = True

        ax.set_xlabel("歩行周期 [%]")
        ylabel = feature + (f" [{unit}]" if unit else "")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{feature}: {ph}  |  band={title_band}")
        ax.grid(True, alpha=0.3)
        if any_plotted:
            ax.legend(ncol=2, fontsize=9)
        ax.set_xlim(0, 100)
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            out = save_dir / f"{feature}_{ph}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"saved: {out}")
        else:
            fig.tight_layout()
            plt.show()

# ===== CLI =====
def expand_input_paths(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    wildcard_chars = set("*?[]")
    for p in paths:
        s = str(p); pth = Path(s)
        if any(ch in s for ch in wildcard_chars):
            out.extend(Path().glob(s))
        else:
            out.append(pth)
    out2: List[Path] = []
    seen = set()
    for p in out:
        try:
            rp = p.resolve()
        except Exception:
            continue
        if not p.is_file():
            continue
        if rp in seen:
            continue
        seen.add(rp)
        out2.append(p)
    return out2

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-subject mean curves by label/phase with robust bands. (default: per-subject output to save_dir/<subject>/)"
    )
    ap.add_argument("inputs", type=Path, nargs="*", help="Labelled *_set-*.txt files or globs")
    ap.add_argument("--file", dest="single_file", type=Path, help="Analyze only this single labelled text file")
    ap.add_argument("--low-thr", type=float, default=-60.0, help="Swing start threshold on grf_z (below => swing)")
    ap.add_argument("--high-thr", type=float, default=-59.0, help="Stance start threshold on grf_z (above => stance)")
    ap.add_argument("--points", type=int, default=101, help="Normalized points per segment")
    ap.add_argument("--min-seg-ms", type=float, default=150.0, help="Minimum segment duration [ms]")
    ap.add_argument("--phase", type=str, choices=["stance","swing","both"], default="both", help="Plot stance, swing, or both")
    ap.add_argument("--features", type=str, default="all", help="[Ignored] Always plots all 17 features")
    ap.add_argument("--save-dir", type=Path, default=Path("plots")/"phase_curves"/"per_subject", help="Base directory to save figures")
    ap.add_argument("--labels", type=str, default="all", help="Comma separated labels or 'all'")
    ap.add_argument("--band", type=str, default="iqr",
                    choices=["iqr","p10-90","std","se","ci95","mad","minmax"],
                    help="帯の種類（既定: iqr）")
    ap.add_argument("--also-all", action="store_true", help="被験者ごと出力に加え、全被験者まとめの図も保存")

    args = ap.parse_args()

    # Resolve file list
    if args.single_file is not None:
        if not args.single_file.is_file():
            raise SystemExit(f"--file not found: {args.single_file}")
        files = [args.single_file]
    else:
        files = expand_input_paths(args.inputs or [Path("Labelled_data") / "*_set-*.txt"])
    if not files:
        raise SystemExit("No input files.")
    print(f"Found {len(files)} file(s)")

    if args.low_thr >= args.high_thr:
        raise SystemExit("--low-thr must be lower than --high-thr")

    feat_list = FEATURE_NAMES

    # Collect segments
    all_segments: List[Segment] = []
    for i, fp in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] loading {fp}")
        series = load_set_series(fp)
        subject = extract_subject(fp)
        segs = segment_by_label(series, args.low_thr, args.high_thr, subject)
        keep: List[Segment] = []
        min_len_s = args.min_seg_ms / 1000.0
        for s in segs:
            if s.t[-1] - s.t[0] >= min_len_s:
                keep.append(s)
        print(f"    segments: total={len(segs)}, kept={len(keep)} (subject={subject})")
        all_segments.extend(keep)

    if not all_segments:
        raise SystemExit("No segments after thresholding/filters. Adjust --low-thr/--high-thr.")

    # Normalize & resample
    samples: List[Tuple[str, str, str, np.ndarray, np.ndarray]] = []
    for seg in all_segments:
        try:
            samples.append(normalize_and_resample(seg, args.points))
        except Exception:
            pass

    # Label filter/ordering
    if args.labels.strip().lower() == "all":
        labels_wanted = LABEL_ORDER[:]
    else:
        labels_wanted = [_canon(s) for s in args.labels.split(",") if s.strip()]

    phase_filter: Optional[str] = None if args.phase == "both" else args.phase

    # ===== per-subject output (default) =====
    save_base: Optional[Path] = args.save_dir
    subjects = sorted({s for (s, _, _, _, _) in samples})
    print(f"Subjects detected: {subjects}")
    for subj in subjects:
        sub_samples_LP = [(l, p, x, Y) for (s, l, p, x, Y) in samples if s == subj]  # (label, phase, x, Y)
        agg = aggregate_by_label_phase(sub_samples_LP)
        print(f"[{subj}] Aggregated into {len(agg)} (label, phase) groups")
        for feat in feat_list:
            out_dir = (save_base / subj) if save_base else None
            plot_feature_curves(agg, feat, phase_filter, out_dir, labels_wanted=labels_wanted, band=args.band)

    # ===== optional: also-all (everyone combined) =====
    if args.also_all:
        agg_all = aggregate_by_label_phase([(l, p, x, Y) for (_, l, p, x, Y) in samples])
        print(f"[ALL] Aggregated into {len(agg_all)} (label, phase) groups")
        for feat in feat_list:
            out_dir = (save_base / "_ALL") if save_base else None
            plot_feature_curves(agg_all, feat, phase_filter, out_dir, labels_wanted=labels_wanted, band=args.band)

if __name__ == "__main__":
    main()
