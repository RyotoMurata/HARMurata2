import argparse
import os
import re
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Append Excel-recorded data to labelled set-XXXX files by aligning times,"
            " or explore labelled files by subject.\n"
            "Append: read 'Metadata Time' start, convert to absolute, align nearest, fill gaps.\n"
            "Explore: list/summarize files filtered by subject name."
        )
    )
    # Make positional args optional to enable explore-only mode
    p.add_argument("label_file", nargs="?", help="Path to labelled set-XXXX .txt file")
    p.add_argument("excel_file", nargs="?", help="Path to Excel .xls/.xlsx containing data")
    p.add_argument(
        "--sheets",
        nargs="*",
        default=None,
        help=(
            "Sensor sheet names to import (default: all sheets that have 'Time (s)').\n"
            "Examples: 'Accelerometer' 'Gyroscope' 'Linear Acceleration' 'Gravity'"
        ),
    )
    p.add_argument(
        "--meta-sheet",
        default="Metadata Time",
        help="Sheet name containing start/end times (default: 'Metadata Time')",
    )
    p.add_argument(
        "--start-event",
        default="START",
        help="Event name that marks measurement start (default: START)",
    )
    p.add_argument(
        "--time-col",
        default="Time (s)",
        help="Column name for relative time in sensor sheets (default: 'Time (s)')",
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Output file path. If omitted, writes alongside input as '<name>_excel.txt'"
        ),
    )
    p.add_argument(
        "--tolerance-ms",
        type=int,
        default=None,
        help=(
            "Optional tolerance in milliseconds for nearest alignment. If set, rows whose nearest"
            " sensor time is farther than this tolerance remain NaN before filling."
        ),
    )
    p.add_argument(
        "--no-bfill-leading",
        action="store_true",
        help="Do not backfill leading NaNs in appended columns (default: backfill leading)",
    )
    # Explore mode options
    p.add_argument(
        "--subject",
        default=None,
        help="Explore mode: subject substring to filter filenames (e.g., 'murata')",
    )
    p.add_argument(
        "--label-dir",
        default="Labelled_data",
        help="Explore mode: directory of labelled files (default: Labelled_data)",
    )
    p.add_argument(
        "--pattern",
        default=None,
        help="Explore mode: additional filename filter (e.g., 'ramp' or 'stair')",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Explore mode: only list matching files (no content read)",
    )
    p.add_argument(
        "--head",
        type=int,
        default=0,
        help="Explore mode: print first N lines of each matching file",
    )
    p.add_argument(
        "--tail",
        type=int,
        default=0,
        help="Explore mode: print last N lines of each matching file",
    )
    return p.parse_args()


_QUOTE_RE = re.compile(r'"([^\"]*)"')


def _parse_labelled_line(line: str) -> Tuple[pd.Timestamp, str, str, str]:
    """
    Parse a labelled data line of the form:
    [YYYY-MM-DD HH:MM:SS.mmm] <numbers...> "<label>" "<SetID=XXXX>"

    Returns: (timestamp, numeric_part_str, label_str, setid_str)
    The numeric_part_str is preserved without trailing spaces changes.
    """
    line = line.rstrip("\n\r")
    if not line.startswith("["):
        raise ValueError("Line does not start with '[': " + line[:50])
    rb = line.find("]")
    if rb < 0:
        raise ValueError("No closing ']' for timestamp: " + line[:80])
    ts_str = line[1:rb]
    rest = line[rb + 1 :].lstrip()
    # Extract the two quoted fields (label and setid)
    qs = _QUOTE_RE.findall(rest)
    if len(qs) < 2:
        raise ValueError("Expected two quoted fields in line: " + line[:120])
    label, setid = qs[-2], qs[-1]
    # numeric part is everything before the first quote occurrence
    first_quote_idx = rest.find('"')
    numeric_part = rest[:first_quote_idx].strip()
    # Parse timestamp to pandas Timestamp (naive)
    # Accept formats with or without milliseconds
    try:
        ts = pd.to_datetime(ts_str)
        # Ensure tz-naive for consistent matching with Excel local time
        if ts.tzinfo is not None:
            # Drop timezone while keeping local wall time
            ts = ts.tz_localize(None)
    except Exception:
        raise
    return ts, numeric_part, label, setid


def _format_timestamp(ts: pd.Timestamp) -> str:
    # Format like existing files: [YYYY-MM-DD HH:MM:SS.mmm]
    # If no ms, still print 3-digit ms
    s = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return f"[{s}]"


def read_labelled_file(path: str) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    times: List[pd.Timestamp] = []
    numeric_parts: List[str] = []
    labels: List[str] = []
    setids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ts, num_str, label, setid = _parse_labelled_line(line)
            times.append(ts)
            numeric_parts.append(num_str)
            labels.append(label)
            setids.append(setid)
    df = pd.DataFrame({"time": pd.to_datetime(times)})
    return df, numeric_parts, labels, setids


def read_metadata_start_time(xls: pd.ExcelFile, meta_sheet: str, start_event: str) -> pd.Timestamp:
    dfm = xls.parse(meta_sheet)
    # Try common column names
    # Expected columns: 'event', 'experiment time', 'system time', 'system time text'
    cols = {c.lower(): c for c in dfm.columns}
    if "event" not in cols:
        raise ValueError(f"'{meta_sheet}' sheet must contain an 'event' column")
    # Prefer precise 'system time text' when present, else epoch 'system time'
    sys_text_col = cols.get("system time text")
    sys_epoch_col = cols.get("system time")
    event_col = cols["event"]
    # First START row
    start_rows = dfm[dfm[event_col].astype(str).str.upper() == start_event.upper()]
    if start_rows.empty:
        raise ValueError(f"No '{start_event}' row found in '{meta_sheet}'")
    row0 = start_rows.iloc[0]
    if sys_text_col and pd.notna(row0[sys_text_col]):
        start_ts = pd.to_datetime(row0[sys_text_col])
    elif sys_epoch_col and pd.notna(row0[sys_epoch_col]):
        # Epoch seconds
        start_ts = pd.to_datetime(float(row0[sys_epoch_col]), unit="s", utc=True)
    else:
        raise ValueError(
            f"'{meta_sheet}' must contain 'system time text' or 'system time' for start event"
        )
    # Convert to tz-naive local time-like (drop tz) to match labelled files
    if start_ts.tzinfo is not None:
        # Drop timezone while keeping local wall time
        start_ts = start_ts.tz_localize(None)
    return start_ts


def read_sensor_sheet_abs_times(
    xls: pd.ExcelFile, sheet: str, time_col: str, start_ts: pd.Timestamp
) -> pd.DataFrame:
    df = xls.parse(sheet)
    # Must contain relative time column
    if time_col not in df.columns:
        # Try case-insensitive match
        matches = [c for c in df.columns if c.strip().lower() == time_col.strip().lower()]
        if matches:
            time_col = matches[0]
        else:
            raise ValueError(f"Sheet '{sheet}' has no time column '{time_col}'. Columns: {list(df.columns)}")
    rel_s = pd.to_numeric(df[time_col], errors="coerce").astype(float)
    abs_time = start_ts + pd.to_timedelta(rel_s, unit="s")
    df = df.drop(columns=[time_col])
    # Sanitize column names to include sheet prefix and short axis names
    new_cols = {}
    axis_re = re.compile(r"\b([xyzXYZ])\b")
    clean_re = re.compile(r"[^A-Za-z0-9_]+")
    for c in df.columns:
        cc = str(c)
        m = axis_re.search(cc)
        if m is not None:
            suffix = m.group(1).lower()
        else:
            suffix = clean_re.sub("_", cc).strip("_")
        name = f"{sheet}:{suffix}"
        new_cols[c] = name
    df = df.rename(columns=new_cols)
    df.insert(0, "time", abs_time)
    # Ensure sorted by time for asof merge
    df = df.sort_values("time").reset_index(drop=True)
    return df


def build_appended_columns(
    labelled_times: pd.Series,
    sensor_dfs: Dict[str, pd.DataFrame],
    tolerance_ms: int | None,
    do_bfill_leading: bool,
) -> pd.DataFrame:
    aligned = pd.DataFrame({"time": labelled_times})
    base_times = aligned[["time"]].sort_values("time").reset_index(drop=True)
    for sheet, sdf in sensor_dfs.items():
        # Merge asof for nearest match per column group
        # We merge once per sheet to pull all its columns
        join_cols = [c for c in sdf.columns if c != "time"]
        # pandas.merge_asof requires both sides sorted by 'on'
        merged = pd.merge_asof(
            base_times,
            sdf.sort_values("time"),
            on="time",
            direction="nearest",
            tolerance=(pd.Timedelta(milliseconds=tolerance_ms) if tolerance_ms else None),
        )
        # Append new columns to aligned
        for c in join_cols:
            aligned[c] = merged[c]
    # Forward-fill gaps per column
    aligned = aligned.sort_values("time").reset_index(drop=True)
    new_only = aligned.drop(columns=["time"]) if "time" in aligned.columns else aligned
    new_only_ff = new_only.ffill()
    if do_bfill_leading:
        new_only_ff = new_only_ff.bfill()
    # Reattach time
    out = pd.concat([aligned[["time"]], new_only_ff], axis=1)
    return out


def write_augmented_file(
    out_path: str,
    times: pd.Series,
    numeric_parts: List[str],
    labels: List[str],
    setids: List[str],
    appended: pd.DataFrame,
):
    # Columns to append in deterministic order
    append_cols = [c for c in appended.columns if c != "time"]
    with open(out_path, "w", encoding="utf-8") as w:
        for i in range(len(times)):
            ts = times.iloc[i]
            base = f"{_format_timestamp(ts)} {numeric_parts[i]}"
            # Values to append
            vals = []
            for c in append_cols:
                v = appended.at[i, c]
                if pd.isna(v):
                    # As a last resort, write empty numeric 0.0 to keep shape
                    vals.append("0.0")
                else:
                    # Compact but precise float formatting
                    if isinstance(v, (float, np.floating)):
                        vals.append(f"{float(v):.6f}")
                    else:
                        vals.append(str(v))
            extra = (" " + " ".join(vals)) if vals else ""
            w.write(f"{base}{extra} \"{labels[i]}\" \"{setids[i]}\"\n")


# ===== Explore mode helpers =====
from collections import Counter
from typing import Iterable


def _iter_label_files(root: str, subject: str, pattern: str | None) -> Iterable[str]:
    subj = subject.lower()
    patt = (pattern.lower() if pattern else None)
    if not os.path.isdir(root):
        return []
    for name in sorted(os.listdir(root)):
        if not name.lower().endswith('.txt'):
            continue
        low = name.lower()
        if subj not in low:
            continue
        if patt and patt not in low:
            continue
        yield os.path.join(root, name)


def _summarize_label_file(path: str) -> dict:
    n_lines = 0
    tmin = None
    tmax = None
    labels_ctr = Counter()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    ts, _, label, _ = _parse_labelled_line(s)
                except Exception:
                    continue
                n_lines += 1
                if tmin is None or ts < tmin:
                    tmin = ts
                if tmax is None or ts > tmax:
                    tmax = ts
                labels_ctr[label] += 1
    except FileNotFoundError:
        pass
    return {"lines": n_lines, "start": tmin, "end": tmax, "labels": labels_ctr}


def _print_head_tail(path: str, head: int, tail: int):
    if head > 0:
        print(f"== HEAD({head}) {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= head:
                    break
                print(line.rstrip('\n'))
    if tail > 0:
        print(f"== TAIL({tail}) {path}")
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines[-tail:]:
            print(line.rstrip('\n'))


def main():
    args = parse_args()
    # Explore mode branch
    if args.subject:
        files = list(_iter_label_files(args.label_dir, args.subject, args.pattern))
        if not files:
            print("No files matched.")
            return
        print(
            f"Matched {len(files)} files under {args.label_dir} for subject='{args.subject}' pattern='{args.pattern}'"
        )
        for pth in files:
            print("-", pth)
        if args.list and not args.head and not args.tail:
            return
        print("\nSummary:")
        from collections import Counter
        total_lines = 0
        overall_labels = Counter()
        overall_start = None
        overall_end = None
        for pth in files:
            info = _summarize_label_file(pth)
            total_lines += info["lines"]
            overall_labels.update(info["labels"]) 
            s = info["start"]
            e = info["end"]
            if s is not None and (overall_start is None or s < overall_start):
                overall_start = s
            if e is not None and (overall_end is None or e > overall_end):
                overall_end = e
            print(
                f"  {os.path.basename(pth)}: lines={info['lines']} start={info['start']} end={info['end']} labels={dict(info['labels'])}"
            )
        print("\nOverall:")
        print(
            f"  lines={total_lines} start={overall_start} end={overall_end} labels={dict(overall_labels)}"
        )
        if args.head or args.tail:
            print()
            for pth in files:
                _print_head_tail(pth, args.head, args.tail)
        return

    # Append mode
    label_file = args.label_file
    excel_file = args.excel_file
    if not label_file or not excel_file:
        raise SystemExit(
            "Append mode requires positional arguments: <label_file> <excel_file>. "
            "Or use --subject to run explore mode."
        )
    out_path = (
        args.output
        if args.output
        else os.path.splitext(label_file)[0] + "_excel.txt"
    )

    # 1) Read labelled file timestamps and structure
    labelled_df, numeric_parts, labels, setids = read_labelled_file(label_file)

    # 2) Open Excel and read start time from metadata
    xls = pd.ExcelFile(excel_file)
    start_ts = read_metadata_start_time(xls, args.meta_sheet, args.start_event)

    # 3) Decide sheets to import
    sheet_names = args.sheets
    if not sheet_names:
        # Heuristic: sheets that include a 'Time (s)' column
        cand = []
        for name in xls.sheet_names:
            try:
                head = xls.parse(name, nrows=1)
            except Exception:
                continue
            if any(str(c).strip().lower() == args.time_col.strip().lower() for c in head.columns):
                cand.append(name)
        sheet_names = cand
    if not sheet_names:
        raise ValueError("No sensor sheets found to import (no sheets with 'Time (s)')")

    # 4) Read each sensor sheet and build absolute time series
    sensor_dfs: Dict[str, pd.DataFrame] = {}
    for name in sheet_names:
        sdf = read_sensor_sheet_abs_times(xls, name, args.time_col, start_ts)
        sensor_dfs[name] = sdf

    # 5) Align to nearest labelled times and forward-fill gaps
    appended = build_appended_columns(
        labelled_times=labelled_df["time"],
        sensor_dfs=sensor_dfs,
        tolerance_ms=args.tolerance_ms,
        do_bfill_leading=(not args.no_bfill_leading),
    )

    # 6) Write augmented file
    write_augmented_file(
        out_path=out_path,
        times=labelled_df["time"],
        numeric_parts=numeric_parts,
        labels=labels,
        setids=setids,
        appended=appended,
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
