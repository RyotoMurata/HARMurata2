from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class Config:
    # Settings
    time_basis: str = "absolute"  # "absolute" or "relative"
    offset_ms: int = 0  # sensor-annotation offset correction in ms
    snap_tolerance_ms: int = 5  # epsilon for snapping
    transition_margin_ms: int = 200  # shrink edges by this margin
    min_duration_ms: int = 150  # minimum duration
    include_transition: bool = False  # export transition rows as a label
    exclude_unknown_in_labeled_only: bool = True
    label_normalization: Dict[str, str] = None
    priority: List[str] = None  # not used in prototype; last-wins policy used

    def __post_init__(self):
        if self.label_normalization is None:
            self.label_normalization = {
                "stair ascent": "stair_up",
                "upstair": "stair_up",
                "upstairs": "stair_up",
                "stair_up": "stair_up",
                "stair descent": "stair_down",
                "downstair": "stair_down",
                "downstairs": "stair_down",
                "stair_down": "stair_down",
            }
        if self.priority is None:
            self.priority = []


@dataclass
class Segment:
    start: datetime
    end: datetime
    label: str
    src: Optional[str] = None  # which annotation file

    def duration_ms(self) -> int:
        return int((self.end - self.start).total_seconds() * 1000)


def parse_annotation_file(path: Path, cfg: Config) -> Tuple[List[Segment], Dict[str, Any]]:
    """
    Expected format example:
        2024/12/13
        13:55:00.610 - 13:55:12.550 UpStair
        13:55:16.994 - 13:55:30.677 DownStair
    First line may be a date (YYYY/MM/DD). Times are absolute times for that date.
    """
    text = path.read_text(encoding="utf-8").strip().splitlines()
    meta: Dict[str, Any] = {"short_segments_removed": 0, "overlaps": 0}

    base_date: Optional[datetime] = None
    segments: List[Segment] = []

    # Try detect a date on the first non-empty line
    i = 0
    while i < len(text) and not text[i].strip():
        i += 1
    if i < len(text):
        head = text[i].strip()
        try:
            base_date = datetime.strptime(head, "%Y/%m/%d")
            i += 1
        except ValueError:
            base_date = None

    for line in text[i:]:
        line = line.strip()
        if not line:
            continue
        # e.g., 13:55:00.610 - 13:55:12.550 UpStair
        try:
            parts = line.split()
            # expect: [start_time, '-', end_time, label words...]
            dash_idx = parts.index('-') if '-' in parts else 1
            start_s = parts[0]
            end_s = parts[dash_idx + 1]
            label_raw = ' '.join(parts[dash_idx + 2:]).strip()

            # Normalize label
            norm_key = label_raw.replace('_', ' ').lower()
            label = cfg.label_normalization.get(norm_key, label_raw.lower())

            # Build datetime
            if base_date:
                start_dt = datetime.strptime(
                    base_date.strftime("%Y-%m-%d ") + start_s, "%Y-%m-%d %H:%M:%S.%f"
                )
                end_dt = datetime.strptime(
                    base_date.strftime("%Y-%m-%d ") + end_s, "%Y-%m-%d %H:%M:%S.%f"
                )
            else:
                # Fallback: try parse absolute datetime directly
                start_dt = datetime.strptime(start_s, "%H:%M:%S.%f")
                end_dt = datetime.strptime(end_s, "%H:%M:%S.%f")

            # Offset correction
            if cfg.offset_ms:
                delta = timedelta(milliseconds=cfg.offset_ms)
                start_dt += delta
                end_dt += delta

            seg = Segment(start=start_dt, end=end_dt, label=label, src=str(path))
            if seg.duration_ms() < cfg.min_duration_ms:
                meta["short_segments_removed"] += 1
                continue
            segments.append(seg)
        except Exception:
            # Skip unparseable lines in prototype
            continue

    # Sort and basic overlap accounting (last-wins applied later during assignment)
    segments.sort(key=lambda s: (s.start, s.end))
    # Count overlaps for reporting
    for j in range(1, len(segments)):
        if segments[j].start < segments[j - 1].end:
            meta["overlaps"] += 1

    return segments, meta


def shrink_for_transition(seg: Segment, margin_ms: int) -> Tuple[Optional[Segment], Optional[Segment], Optional[Segment]]:
    """
    Returns (left_transition, core, right_transition) segments according to margin.
    If margin removes the whole segment, core becomes None.
    """
    if margin_ms <= 0:
        return None, seg, None
    margin = timedelta(milliseconds=margin_ms)
    left_end = min(seg.end, seg.start + margin)
    right_start = max(seg.start, seg.end - margin)
    core_start = left_end
    core_end = right_start
    left = Segment(seg.start, left_end, label="transition", src=seg.src) if left_end > seg.start else None
    core = Segment(core_start, core_end, label=seg.label, src=seg.src) if core_end > core_start else None
    right = Segment(right_start, seg.end, label="transition", src=seg.src) if seg.end > right_start else None
    return left, core, right


def parse_sensor_files(data_dir: Path, pattern: str = "stair_t*.txt") -> List[Dict[str, Any]]:
    """
    Parse text sensor logs with lines like:
        [YYYY-MM-DD HH:MM:SS.mmm] <optional tokens> <float> <float> ...
    Returns a list of dicts with timestamp, values..., and source_file.
    """
    rows: List[Dict[str, Any]] = []
    for path in sorted(data_dir.glob(pattern)):
        try:
            for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                if not raw.startswith('['):
                    continue
                # Extract timestamp inside brackets
                try:
                    ts_end = raw.index(']')
                except ValueError:
                    continue
                ts_str = raw[1:ts_end].strip()
                rest = raw[ts_end + 1 :].strip()
                # Strip any tokens like 'uart:~$'
                rest = rest.replace('uart:~$', '').strip()
                # Split into floats; some lines may have non-numeric tokens; skip those
                nums: List[float] = []
                ok = True
                for tok in rest.split():
                    try:
                        nums.append(float(tok))
                    except ValueError:
                        ok = False
                        break
                if not ok or not nums:
                    continue
                # Parse timestamp
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                except ValueError:
                    # try without milliseconds
                    try:
                        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        continue
                row: Dict[str, Any] = {"timestamp": ts, "source_file": path.name}
                for i, val in enumerate(nums):
                    row[f"v{i}"] = val
                rows.append(row)
        except FileNotFoundError:
            continue
    # Sort by timestamp
    rows.sort(key=lambda r: r["timestamp"]) 
    return rows


def assign_labels(rows: List[Dict[str, Any]], segments: List[Segment], cfg: Config) -> None:
    """Assign labels to rows in-place using a last-wins policy."""
    # Prepare segments with transition margins
    prepared: List[Segment] = []
    for seg in segments:
        left, core, right = shrink_for_transition(seg, cfg.transition_margin_ms)
        if left and cfg.include_transition:
            prepared.append(left)
        if core:
            prepared.append(core)
        if right and cfg.include_transition:
            prepared.append(right)
    prepared.sort(key=lambda s: (s.start, s.end))

    # Two-pointer sweep for efficiency
    j = 0
    nseg = len(prepared)
    for row in rows:
        row["label"] = "unknown"
        t = row["timestamp"]
        # Advance j while segments end before t with tolerance
        while j < nseg and prepared[j].end + timedelta(milliseconds=cfg.snap_tolerance_ms) < t:
            j += 1
        # Check applicable segments from current j backwards (last-wins)
        k = j
        while k < nseg and prepared[k].start - timedelta(milliseconds=cfg.snap_tolerance_ms) <= t <= prepared[k].end + timedelta(milliseconds=cfg.snap_tolerance_ms):
            row["label"] = prepared[k].label
            k += 1


def add_derived_features(rows: List[Dict[str, Any]]) -> None:
    """Add derived feature columns in-place.

    - Power: encoder_angle * grt_x, assuming channel order consistent with
      Compare_labelled.py (v10 = encoder_angle, v14 = grt_x).
    """
    for r in rows:
        try:
            ea = r.get("v10")
            gx = r.get("v14")
            if ea is not None and gx is not None:
                r["Power"] = float(ea) * float(gx)
        except Exception:
            # If conversion fails, skip adding Power for this row
            pass

def write_outputs(output_dir: Path, rows: List[Dict[str, Any]], segments: List[Segment], cfg: Config, meta: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # segments.csv
    seg_csv = output_dir / "segments.csv"
    with seg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_time", "end_time", "label", "duration_ms", "src"])
        for s in segments:
            w.writerow([s.start.isoformat(), s.end.isoformat(), s.label, s.duration_ms(), s.src or ""])

    # labeled_only.csv
    # Add derived feature columns (e.g., Power)
    add_derived_features(rows)
    labeled_csv = output_dir / "labeled_only.csv"
    # Determine numeric columns dynamically
    num_cols = sorted([k for k in rows[0].keys() if k.startswith("v")], key=lambda x: int(x[1:])) if rows else []
    # Include derived columns explicitly if present
    derived_cols: List[str] = []
    if rows and any("Power" in r for r in rows):
        derived_cols.append("Power")
    headers = ["timestamp", "label", "source_file"] + num_cols + derived_cols
    with labeled_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            if cfg.exclude_unknown_in_labeled_only and (r.get("label") in (None, "unknown", "transition")):
                continue
            base = [r["timestamp"].isoformat(), r.get("label", ""), r.get("source_file", "")]
            vals = [r.get(c, "") for c in num_cols]
            derived = [r.get(c, "") for c in derived_cols]
            w.writerow(base + vals + derived)

    # labeling_report.txt
    report = output_dir / "labeling_report.txt"
    # Class distribution by label
    counts: Dict[str, int] = {}
    for r in rows:
        lbl = r.get("label", "unknown")
        counts[lbl] = counts.get(lbl, 0) + 1
    total = len(rows)
    labeled = total - counts.get("unknown", 0) - (0 if cfg.include_transition else counts.get("transition", 0))
    coverage = (labeled / total * 100.0) if total else 0.0
    with report.open("w", encoding="utf-8") as f:
        f.write("Labeling Report\n")
        f.write("================\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Labeled samples: {labeled}\n")
        f.write(f"Coverage: {coverage:.2f}%\n\n")
        f.write("Class distribution (samples)\n")
        for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"  {k}: {v}\n")
        f.write("\nWarnings/Notes\n")
        f.write(f"  Short segments removed: {meta.get('short_segments_removed', 0)}\n")
        f.write(f"  Overlaps detected (last-wins): {meta.get('overlaps', 0)}\n")

    # config.yaml (simple JSON-style YAML for prototype) and run_meta.json
    cfg_yaml = output_dir / "config.yaml"
    try:
        import yaml  # type: ignore
        with cfg_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(cfg), f, allow_unicode=True, sort_keys=False)
    except Exception:
        # Fallback: write JSON with .yaml extension
        with cfg_yaml.open("w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    run_meta = {
        "generated_at": datetime.now().isoformat(),
        "segments": len(segments),
        "total_rows": total,
        "counts": counts,
        "inputs": {
            "data_dir": str(output_dir.parent),
        },
        "notes": meta,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Prototype labeling pipeline")
    p.add_argument("--annotation", type=str, default="1213/20241213ラベリング.txt", help="Annotation text file path")
    p.add_argument("--data_dir", type=str, default="1213", help="Directory with sensor text files")
    p.add_argument("--pattern", type=str, default="stair_t*.txt", help="Glob pattern for sensor files")
    p.add_argument("--output_dir", type=str, default="1213/labeled_output", help="Output directory")
    # Settings
    p.add_argument("--offset_ms", type=int, default=0)
    p.add_argument("--snap_ms", type=int, default=5)
    p.add_argument("--transition_ms", type=int, default=200)
    p.add_argument("--min_duration_ms", type=int, default=150)
    p.add_argument("--include_transition", action="store_true")
    args = p.parse_args(argv)

    cfg = Config(
        time_basis="absolute",
        offset_ms=args.offset_ms,
        snap_tolerance_ms=args.snap_ms,
        transition_margin_ms=args.transition_ms,
        min_duration_ms=args.min_duration_ms,
        include_transition=args.include_transition,
    )

    annotation_path = Path(args.annotation)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    segments, meta = parse_annotation_file(annotation_path, cfg)
    # Apply transition margin and (optional) keep transition labeled segments for segments.csv
    segs_for_csv: List[Segment] = []
    for s in segments:
        left, core, right = shrink_for_transition(s, cfg.transition_margin_ms)
        if left and cfg.include_transition:
            segs_for_csv.append(left)
        if core:
            segs_for_csv.append(core)
        if right and cfg.include_transition:
            segs_for_csv.append(right)

    rows = parse_sensor_files(data_dir, pattern=args.pattern)
    assign_labels(rows, segments, cfg)
    write_outputs(output_dir, rows, segs_for_csv or segments, cfg, meta)


if __name__ == "__main__":
    main()
