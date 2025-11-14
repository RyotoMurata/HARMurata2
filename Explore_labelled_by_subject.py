import argparse
import os
import re
from collections import Counter
from typing import Iterable, Tuple

import pandas as pd


# Reuse the parser from Add_excel_to_labelled if available
try:
    from Add_excel_to_labelled import _parse_labelled_line  # type: ignore
except Exception:
    _Q = re.compile(r'"([^\"]*)"')

    def _parse_labelled_line(line: str) -> Tuple[pd.Timestamp, str, str, str]:
        line = line.rstrip("\n\r")
        if not line.startswith("["):
            raise ValueError("Bad line")
        rb = line.find("]")
        ts_str = line[1:rb]
        rest = line[rb + 1 :].lstrip()
        qs = _Q.findall(rest)
        label, setid = qs[-2], qs[-1]
        first_quote_idx = rest.find('"')
        numeric_part = rest[:first_quote_idx].strip()
        ts = pd.to_datetime(ts_str)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts, numeric_part, label, setid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Explore labelled set-XXXX files by subject name (e.g., murata, kaneishi)."
    )
    p.add_argument("subject", help="Subject name to match in filenames (case-insensitive)")
    p.add_argument(
        "--dir",
        default="Labelled_data",
        help="Directory containing labelled files (default: Labelled_data)",
    )
    p.add_argument(
        "--pattern",
        default=None,
        help="Optional extra filename pattern to filter (e.g., 'ramp' or 'stair')",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Only list matching files without reading contents",
    )
    p.add_argument(
        "--head",
        type=int,
        default=0,
        help="Print first N lines of each matching file",
    )
    p.add_argument(
        "--tail",
        type=int,
        default=0,
        help="Print last N lines of each matching file",
    )
    return p.parse_args()


def iter_files(root: str, subject: str, pattern: str | None) -> Iterable[str]:
    subj = subject.lower()
    patt = (pattern.lower() if pattern else None)
    for name in sorted(os.listdir(root)):
        if not name.lower().endswith(".txt"):
            continue
        low = name.lower()
        if subj not in low:
            continue
        if patt and patt not in low:
            continue
        yield os.path.join(root, name)


def summarize_file(path: str) -> dict:
    n_lines = 0
    tmin = None
    tmax = None
    labels = Counter()
    with open(path, "r", encoding="utf-8") as f:
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
            labels[label] += 1
    return {
        "lines": n_lines,
        "start": tmin,
        "end": tmax,
        "labels": labels,
    }


def print_head_tail(path: str, head: int, tail: int):
    if head > 0:
        print(f"== HEAD({head}) {path}")
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= head:
                    break
                print(line.rstrip("\n"))
    if tail > 0:
        print(f"== TAIL({tail}) {path}")
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-tail:]:
            print(line.rstrip("\n"))


def main():
    args = parse_args()
    root = args.dir
    files = list(iter_files(root, args.subject, args.pattern))
    if not files:
        print("No files matched.")
        return

    print(f"Matched {len(files)} files under {root} for subject='{args.subject}' pattern='{args.pattern}'")
    for p in files:
        print("-", p)
    if args.list and not args.head and not args.tail:
        return

    print("\nSummary:")
    total_lines = 0
    overall_labels = Counter()
    overall_start = None
    overall_end = None
    for p in files:
        info = summarize_file(p)
        total_lines += info["lines"]
        overall_labels.update(info["labels"]) 
        s = info["start"]
        e = info["end"]
        if s is not None and (overall_start is None or s < overall_start):
            overall_start = s
        if e is not None and (overall_end is None or e > overall_end):
            overall_end = e
        print(f"  {os.path.basename(p)}: lines={info['lines']} start={info['start']} end={info['end']} labels={dict(info['labels'])}")

    print("\nOverall:")
    print(f"  lines={total_lines} start={overall_start} end={overall_end} labels={dict(overall_labels)}")

    if args.head or args.tail:
        print()
        for p in files:
            print_head_tail(p, args.head, args.tail)


if __name__ == "__main__":
    main()

