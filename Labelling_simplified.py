from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

"""
Annotation-driven segment extractor.

Reads an annotation file, locates directed-nearest windows in raw .txt data
by timestamp, and writes one labeled file per annotation with a sequential
SetID appended.

Python 3.9+, standard library only.
"""

# ===== User-editable defaults (you can change these at the top) =====
# Path to the annotation file to use by default
DEFAULT_ANNOTATIONS = Path("Annotation_ramp_sasaki.txt") #被験者を変えるときは、出力ファイル名も変えないと上書きされる
# Folders where raw data files are stored (searched in order)
DEFAULT_DATA_DIRS = [Path("1113")]
# Output folder for labeled segments (if None, uses sibling labeled_output next to source file)
DEFAULT_OUTPUT_DIR = Path("Labelled_data")
# ====================================================================


# 許容フォーマット:
# 1) [file](HH:MM:SS.mmm-HH:MM:SS.mmm)"Label"
# 2) (HH:MM:SS.mmm-HH:MM:SS.mmm)"Label"   ← [file] を省略
ANNOT_LINE_RE = re.compile(
    r'^\s*(?:\[(?P<file>[^\]]+)\]\s*)?\(\s*(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s*-\s*(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})\s*\)\s*"(?P<label>[^"]+)"\s*$'
)


@dataclass
class AnnItem:
    """One annotation item parsed from the text file."""

    file_key: Optional[str]
    start_str: str
    end_str: str
    label: str


def parse_annotations(text: str) -> List[AnnItem]:
    """Parse annotation text into a list of AnnItem.

    Accepts either [file](HH:MM:SS.mmm-HH:MM:SS.mmm)"Label" or
    (HH:MM:SS.mmm-HH:MM:SS.mmm)"Label".
    Lines not matching the format are ignored.
    """
    items: List[AnnItem] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = ANNOT_LINE_RE.match(line)
        if not m:
            continue
        file_key = m.group("file") if m.groupdict().get("file") else None
        items.append(
            AnnItem(
                file_key=file_key,
                start_str=m.group("start"),
                end_str=m.group("end"),
                label=m.group("label"),
            )
        )
    return items


def parse_annotation_sets(text: str) -> List[List[AnnItem]]:
    """Parse annotations grouped into SET blocks.

    Recognizes lines like "//SET 1//" as set separators. Other lines starting
    with "//" are ignored as comments. If annotations appear before any SET
    header, they are grouped into the first implicit set.
    Returns a list of sets, each being a list of AnnItem.
    """
    sets: List[List[AnnItem]] = []
    current: List[AnnItem] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # SET header
        if line.startswith("//"):
            if re.match(r"^//\s*SET\b", line, flags=re.IGNORECASE):
                if current:
                    sets.append(current)
                    current = []
                else:
                    # start a new empty set
                    current = []
                continue
            # other comments ignored
            continue
        m = ANNOT_LINE_RE.match(line)
        if not m:
            continue
        file_key = m.group("file") if m.groupdict().get("file") else None
        current.append(
            AnnItem(
                file_key=file_key,
                start_str=m.group("start"),
                end_str=m.group("end"),
                label=m.group("label"),
            )
        )
    if current:
        sets.append(current)
    return sets


def find_data_file(file_key: str, search_dirs: Iterable[Path]) -> Optional[Path]:
    """Resolve a data file path by key within the search directories.

    Accepts bare key or key with .txt; also matches files whose stem equals key.
    Returns the first occurrence found respecting directory order.
    """
    candidates: List[Path] = []
    names = [file_key, f"{file_key}.txt"]
    for base in search_dirs:
        for name in names:
            p = base / name
            if p.exists() and p.is_file():
                candidates.append(p)
        for p in base.glob("*.txt"):
            if p.stem == file_key:
                candidates.append(p)
    # de-duplicate preserving order
    seen = set()
    uniq: List[Path] = []
    for p in candidates:
        real = p.resolve()
        if real in seen:
            continue
        seen.add(real)
        uniq.append(p)
    return uniq[0] if uniq else None


def parse_data_timestamp(line: str) -> Optional[datetime]:
    """Parse leading timestamp like [YYYY-MM-DD HH:MM:SS.mmm] or without millis."""
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


def parse_numeric_values(line: str) -> Optional[List[float]]:
    """Extract numeric payload after the timestamp; require at least 17 floats.

    Removes known noise tokens 'uart:~$' and 'uart:$'. If any token fails to
    parse as float, the line is rejected (returns None).
    """
    try:
        idx = line.index("]")
    except ValueError:
        return None
    rest = line[idx + 1 :].strip()
    rest = rest.replace("uart:~$", " ").replace("uart:$", " ").strip()
    vals: List[float] = []
    for tok in rest.split():
        try:
            vals.append(float(tok))
        except ValueError:
            return None
    if len(vals) < 17:
        return None
    return vals


def _find_directed_indices_partial(
    src: Path, t_start_time: datetime, t_end_time: datetime
) -> Tuple[Optional[int], Optional[int]]:
    """Internal helper: find start/end row indices under directed-nearest rules.

    - Start row: ts > start with minimal (ts - start)
    - End row:   ts < end with minimal (end - ts)
    Returns (row_start or None, row_end or None). Counts only rows with valid
    numeric payload. Single pass streaming.
    """
    best_start_idx: int = -1
    best_end_idx: int = -1
    best_start_diff: Optional[float] = None
    best_end_diff: Optional[float] = None
    row_idx = -1

    with src.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ts = parse_data_timestamp(line)
            if ts is None:
                continue
            # Only count lines that have numeric payload
            if parse_numeric_values(line) is None:
                continue
            row_idx += 1

            current_date = ts.date()
            t_start = datetime.combine(current_date, t_start_time.time())
            t_end = datetime.combine(current_date, t_end_time.time())

            if ts > t_start:
                d = (ts - t_start).total_seconds()
                if best_start_diff is None or d < best_start_diff:
                    best_start_diff = d
                    best_start_idx = row_idx

            if ts < t_end:
                d = (t_end - ts).total_seconds()
                if best_end_diff is None or d < best_end_diff:
                    best_end_diff = d
                    best_end_idx = row_idx

    s_idx: Optional[int] = None if best_start_idx < 0 else best_start_idx
    e_idx: Optional[int] = None if best_end_idx < 0 else best_end_idx
    return s_idx, e_idx


def find_directed_indices(
    src: Path, t_start_time: datetime, t_end_time: datetime
) -> Optional[Tuple[int, int]]:
    """Public API: find (row_start, row_end) using directed-nearest rules.

    Returns None if start or end cannot be found, or if row_start > row_end.
    """
    s_idx, e_idx = _find_directed_indices_partial(src, t_start_time, t_end_time)
    if s_idx is None or e_idx is None:
        return None
    if s_idx > e_idx:
        return None
    return s_idx, e_idx


def find_best_file_for_times(
    data_dirs: Iterable[Path], t_start: datetime, t_end: datetime
) -> Optional[Tuple[Path, int, int]]:
    """Scan .txt files and choose the file with the shortest valid window.

    Score = (row_end - row_start), lower is better. Only considers files where
    both directed-nearest indices exist and row_start <= row_end.
    """
    best: Optional[Tuple[Path, int, int, int]] = None  # (path, s_idx, e_idx, span)
    seen: set = set()
    for base in data_dirs:
        if not base.exists():
            continue
        for p in sorted(base.glob("*.txt")):
            real = p.resolve()
            if real in seen:
                continue
            seen.add(real)
            res = find_directed_indices(p, t_start, t_end)
            if not res:
                continue
            s_idx, e_idx = res
            span = e_idx - s_idx
            if best is None or span < best[3]:
                best = (p, s_idx, e_idx, span)
    if best is None:
        return None
    return best[0], best[1], best[2]


def find_best_file_for_set(
    data_dirs: Iterable[Path], intervals: List[Tuple[datetime, datetime]]
) -> Optional[Tuple[Path, List[Tuple[int, int]]]]:
    """Choose a single data file that supports all intervals (directed indices).

    Returns the file path and the list of (row_start,row_end) for each interval,
    preserving the order of `intervals`. Scoring minimizes total span length.
    """
    best: Optional[Tuple[Path, List[Tuple[int, int]], int]] = None
    seen: set = set()
    for base in data_dirs:
        if not base.exists():
            continue
        for p in sorted(base.glob("*.txt")):
            real = p.resolve()
            if real in seen:
                continue
            seen.add(real)
            spans: List[Tuple[int, int]] = []
            ok = True
            total_span = 0
            for t_start, t_end in intervals:
                res = find_directed_indices(p, t_start, t_end)
                if not res:
                    ok = False
                    break
                s_idx, e_idx = res
                spans.append((s_idx, e_idx))
                total_span += (e_idx - s_idx)
            if not ok:
                continue
            if best is None or total_span < best[2]:
                best = (p, spans, total_span)
    if best is None:
        return None
    return best[0], best[1]


def export_set_segments(
    src: Path,
    out_dir: Path,
    windows: List[Tuple[int, int, str]],
    set_id_str: str,
) -> Path:
    """Export multiple windows from one source file into a single set file.

    Each window is (row_start,row_end,label). Rows outside all windows are not
    written. Lines within a window receive "Label" and "SetID=XXXX" suffixes.
    Output filename: <src.stem>_set-XXXX.txt
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}_sasaki_set-{set_id_str}.txt" #出力ファイルの命名

    # Sort by start index to ensure chronological order
    windows_sorted = sorted(windows, key=lambda w: (w[0], w[1]))

    row_idx = -1
    w_i = 0
    current = windows_sorted[w_i] if windows_sorted else None

    with src.open("r", encoding="utf-8", errors="ignore") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        written_lines = 0
        for line in fin:
            ts = parse_data_timestamp(line)
            if ts is None:
                continue
            if parse_numeric_values(line) is None:
                continue
            row_idx += 1
            # Advance window pointer while current window has passed
            while current is not None and row_idx > current[1]:
                w_i += 1
                current = windows_sorted[w_i] if w_i < len(windows_sorted) else None
            if current is None:
                break
            if current[0] <= row_idx <= current[1]:
                label = current[2]
                # Compute derived feature Power = encoder_angle(v10) * grt_x(v14)
                vals = parse_numeric_values(line) or []
                power: Optional[float] = None
                if len(vals) >= 15:
                    try:
                        power = float(vals[10]) * float(vals[14])
                    except Exception:
                        power = None
                base = line.rstrip("\r\n")
                if power is not None:
                    # Append as an extra numeric column before quoted labels
                    base = f"{base} {power:.10g}"
                fout.write(base + f' "{label}" "SetID={set_id_str}"\n')
                written_lines += 1
    print(f"     出力行数: {written_lines}", flush=True)
    return out_path


def run(
    annotation_file: Path,
    data_dirs: List[Path],
    output_dir: Optional[Path] = DEFAULT_OUTPUT_DIR,
) -> List[Path]:
    """Process annotations grouped by sets and export one file per set.

    Returns the list of written file paths.
    """
    print(f"[1/4] 読み込み中: アノテーション {annotation_file}", flush=True)
    ann_text = annotation_file.read_text(encoding="utf-8")
    sets = parse_annotation_sets(ann_text)
    if not sets:
        raise SystemExit(
            'No valid annotations found. Expected format: [file](HH:MM:SS.mmm-HH:MM:SS.mmm)"Label"'
        )
    total_items = sum(len(s) for s in sets)
    print(f"[2/4] 解析完了: セット数={len(sets)}, 総アイテム数={total_items}", flush=True)

    written: List[Path] = []
    next_set_id = 1

    for set_idx, items in enumerate(sets, start=1):
        print(f"[3/4] SET {set_idx}/{len(sets)}: アイテム数={len(items)}", flush=True)
        # Determine target data file for this set
        explicit_key: Optional[str] = next((it.file_key for it in items if it.file_key), None)

        # Build times list
        times: List[Tuple[datetime, datetime]] = [
            (
                datetime.strptime(it.start_str, "%H:%M:%S.%f"),
                datetime.strptime(it.end_str, "%H:%M:%S.%f"),
            )
            for it in items
        ]

        if explicit_key:
            print(f"  -> 明示ファイル指定あり: [{explicit_key}] を探索", flush=True)
            data_path = find_data_file(explicit_key, data_dirs)
            if not data_path:
                print(
                    f"Skip: data file for [{explicit_key}] not found in: {', '.join(map(str, data_dirs))}"
                )
                continue
            print(f"  -> 使用ファイル: {data_path}", flush=True)
            windows: List[Tuple[int, int, str]] = []
            for i_it, (it, (t_start, t_end)) in enumerate(zip(items, times), start=1):
                print(f"     - 区間 {i_it}/{len(items)}: {it.start_str}-{it.end_str} … 走査中", flush=True)
                s_idx_opt, e_idx_opt = _find_directed_indices_partial(data_path, t_start, t_end)
                if s_idx_opt is None:
                    print(f"       -> Skip: ts>{it.start_str} の行なし ({data_path.name})", flush=True)
                    continue
                if e_idx_opt is None:
                    print(f"       -> Skip: ts<{it.end_str} の行なし ({data_path.name})", flush=True)
                    continue
                if s_idx_opt > e_idx_opt:
                    print(f"       -> Skip: 指向ウィンドウの範囲が不正 ({data_path.name})", flush=True)
                    continue
                print(f"       -> OK: rows [{s_idx_opt}, {e_idx_opt}] label='{it.label}'", flush=True)
                windows.append((s_idx_opt, e_idx_opt, it.label))
            if not windows:
                # Nothing valid in this set
                continue
            data_sel = data_path
        else:
            # 候補ファイル数の把握（進捗表示用）
            candidate_files = []
            for base in data_dirs:
                if base.exists():
                    candidate_files.extend(sorted(p for p in base.glob("*.txt") if p.is_file()))
            print(f"  -> 最適ファイル探索: 候補 {len(candidate_files)} 件", flush=True)
            best = find_best_file_for_set(data_dirs, times)
            if not best:
                print(f"Skip: no suitable data file found for all intervals in SET {set_idx}")
                continue
            data_sel, spans = best
            print(f"  -> 使用ファイル: {data_sel}（全区間対応）", flush=True)
            windows = [(s, e, items[i].label) for i, (s, e) in enumerate(spans)]

        out_dir = output_dir if output_dir is not None else (data_sel.parent / "labeled_output")
        set_id_str = f"{next_set_id:04d}"
        print(f"[4/4] 書き出し: {data_sel.name} から {len(windows)} 区間 → SetID={set_id_str}", flush=True)
        out_path = export_set_segments(data_sel, out_dir, windows, set_id_str)
        print(f"  -> Wrote: {out_path}", flush=True)
        written.append(out_path)
        next_set_id += 1

    return written


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "ラベル付け: アノテーションは (start-end)\"Label\" または [file](start-end)\"Label\" を受け付けます。"
        )
    )
    parser.add_argument(
        "-a",
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help=f"Path to annotation text file (default: {DEFAULT_ANNOTATIONS})",
    )
    parser.add_argument(
        "-d",
        "--data-dirs",
        type=str,
        default=";".join(str(p.resolve()) for p in DEFAULT_DATA_DIRS),
        help="Semicolon-separated directories to search for data files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Output folder for labeled segments. Use 'none' to write next to each source file in a sibling 'labeled_output' folder."
        ),
    )
    args = parser.parse_args()

    data_dirs = [Path(p) for p in args.data_dirs.split(";") if p]
    out_dir: Optional[Path]
    if isinstance(args.output_dir, str) and args.output_dir.lower() == "none":
        out_dir = None
    else:
        out_dir = Path(args.output_dir)
    run(args.annotations, data_dirs, out_dir)


if __name__ == "__main__":
    main()
