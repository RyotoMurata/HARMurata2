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
    "Yu Gothic",
    "Yu Gothic UI",
    "Meiryo",
    "MS Gothic",
    "Noto Sans CJK JP",
    "IPAGothic",
    "IPAexGothic",
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


# ===== I/O parsing (compatible with Build_RF_dataset.py) =====
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


# ===== Features metadata =====
# Follow the same 17-channel layout used elsewhere in this repo
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
NAME_TO_INDEX: Dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}
FEATURE_UNITS: Dict[str, str] = {
    "acc_x": "g", "acc_y": "g", "acc_z": "g",
    "gyro_x": "deg/s", "gyro_y": "deg/s", "gyro_z": "deg/s",
    "quat_w": "", "quat_x": "", "quat_y": "", "quat_z": "",
    "encoder_angle": "deg",
    "grf_x": "N", "grf_y": "N", "grf_z": "N",
    "grt_x": "N·m", "grt_y": "N·m", "grt_z": "N·m",
}

STYLE_BY_LABEL: Dict[str, Tuple[str, Optional[str]]] = {
    "stair-ascent":  ("-.", "tab:red"),
    "stair-descent": ("--", "tab:brown"),
    "ramp-ascent":   ("-.", "tab:green"),
    "ramp-descent":  (":",  "tab:purple"),
    "stop":          ("-",  "tab:gray"),
    "level-walk":    ("-",  None),
}


def choose_style_for_label(label: Optional[str]) -> Tuple[str, Optional[str]]:
    if not label:
        return ("-", None)
    k = label.strip().lower().replace(" ", "-").replace("_", "-")
    return STYLE_BY_LABEL.get(k, ("-", None))


def plot_feature_by_label(series: SetSeries, feature: str, save_dir: Optional[Path], show: bool) -> Path:
    if feature not in NAME_TO_INDEX:
        raise SystemExit(f"Unknown feature: {feature}")
    idx = NAME_TO_INDEX[feature]
    unit = FEATURE_UNITS.get(feature, "")
    t = series.t_s
    y = series.data[:, idx]
    labels = series.labels

    # Find contiguous label ranges
    ranges: List[Tuple[int, int, Optional[str]]] = []  # (start, end_exclusive, label)
    if len(labels) > 0:
        start = 0
        curr = labels[0]
        for i in range(1, len(labels)):
            if labels[i] != curr:
                ranges.append((start, i, curr))
                start = i
                curr = labels[i]
        ranges.append((start, len(labels), curr))

    fig, ax = plt.subplots(figsize=(10, 4))
    seen_labels: Dict[str, bool] = {}
    for a, b, lbl in ranges:
        ls, color = choose_style_for_label(lbl)
        lab = lbl if (lbl and lbl not in seen_labels) else None
        ax.plot(t[a:b], y[a:b], linestyle=ls, color=color, label=lab)
        if lbl:
            seen_labels[lbl] = True

    ax.set_xlabel("時間 [s]")
    ylabel = feature + (f" [{unit}]" if unit else "")
    ax.set_ylabel(ylabel)
    title_sid = f" (SetID={series.set_id})" if series.set_id else ""
    ax.set_title(f"{feature} by label: {series.file.stem}{title_sid}")
    ax.grid(True, alpha=0.3)
    if seen_labels:
        ax.legend(ncol=2, fontsize=9)

    fig.tight_layout()
    if save_dir is None and not show:
        save_dir = Path("plots") / "by_label"
    out_path = None
    if save_dir is not None:
        # Save under subfolder per feature
        sub = save_dir / feature
        sub.mkdir(parents=True, exist_ok=True)
        out_path = sub / f"{series.file.stem}_{feature}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"saved: {out_path}")
    if show:
        plt.show()
    return out_path or Path()


# ===== CLI =====

def expand_input_paths(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    wildcard_chars = set("*?[]")
    for p in paths:
        s = str(p)
        pth = Path(s)
        if any(ch in s for ch in wildcard_chars):
            out.extend(Path().glob(s))
        else:
            out.append(pth)
    # only existing files; keep order
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
    ap = argparse.ArgumentParser(description="Plot selected features over full duration, colored by label segments.")
    ap.add_argument("inputs", type=Path, nargs="*", help="Labelled *_set-*.txt files or globs")
    ap.add_argument("--file", dest="single_file", type=Path, help="Analyze only this single labelled text file (overrides positional inputs)")
    ap.add_argument("--save-dir", type=Path, default=Path("plots") / "by_label", help="Base directory to save figures (per-feature subfolders)")
    ap.add_argument("--show", action="store_true", help="Show figure interactively instead of only saving")
    ap.add_argument("--features", type=str, default="all", help="Comma-separated feature names or 'all'")

    args = ap.parse_args()

    # Resolve files
    if args.single_file is not None:
        if not args.single_file.is_file():
            raise SystemExit(f"--file not found: {args.single_file}")
        files = [args.single_file]
    else:
        files = expand_input_paths(args.inputs or [Path("Labelled_data") / "*_set-*.txt"])  # default: all labelled files
    if not files:
        raise SystemExit("No input files.")
    print(f"Found {len(files)} file(s)")

    # Resolve features
    if args.features.strip().lower() == "all":
        feat_list = FEATURE_NAMES
    else:
        feat_list = [s.strip() for s in args.features.split(",") if s.strip()]
        for f in feat_list:
            if f not in NAME_TO_INDEX:
                raise SystemExit(f"Unknown feature: {f}")

    for i, fp in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] loading {fp}")
        series = load_set_series(fp)
        for feat in feat_list:
            plot_feature_by_label(series, feature=feat, save_dir=args.save_dir, show=args.show)


if __name__ == "__main__":
    main()
