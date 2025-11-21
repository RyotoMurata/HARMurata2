import argparse
import os
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import pandas as pd

# ===== ★ CONFIG: 実行ボタン用の固定設定（ここを書き換えるだけ） =====
USE_SCRIPT_OVERRIDES = True     # True: 下の SUBJECT/OUT_DIR を引数より優先
SUBJECT = "sasaki"              # None or "" ならフィルタ無し。例: "murata"
OUT_DIR = "Labelled_data/merged"  # None or "" ならラベルと同じ場所に出力
# ====================================================================

# Reuse core utilities from Add_excel_to_labelled
from Add_excel_to_labelled import (
    read_metadata_start_time,
    read_sensor_sheet_abs_times,
    read_labelled_file,
    build_appended_columns,
    write_augmented_file,
    _parse_labelled_line,
)


@dataclass
class LabelSpan:
    path: str
    start: pd.Timestamp
    end: pd.Timestamp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Scan an Excel directory (e.g., 1029), match each Excel file to the best-overlapping"
            " labelled dataset in Labelled_data, and append Excel sensor data to it."
        )
    )
    p.add_argument("--excel-dir", default="1113", help="Directory with Excel files (.xls/.xlsx)")
    p.add_argument("--label-dir", default="Labelled_data", help="Directory with labelled .txt files")
    p.add_argument(
        "--subject",
        default=None,
        help="Optional subject substring to restrict labelled candidates (e.g., murata)",
    )
    p.add_argument(
        "--sheets",
        nargs="*",
        default=None,
        help="Sensor sheet names to import (default: all sheets with 'Time (s)')",
    )
    p.add_argument("--time-col", default="Time (s)", help="Relative time column in sensor sheets")
    p.add_argument("--meta-sheet", default="Metadata Time", help="Metadata sheet name")
    p.add_argument("--start-event", default="START", help="Start event marker in metadata")
    p.add_argument(
        "--tolerance-ms",
        type=int,
        default=None,
        help="Nearest match tolerance in milliseconds (optional)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: same as label-dir)",
    )
    p.add_argument(
        "--out-suffix",
        default=None,
        help="Suffix appended to labelled basename (default: '__' + excel stem)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned mappings without writing")
    p.add_argument("--limit", type=int, default=0, help="Limit number of Excel files processed")
    # ★ ログレベル
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    return p.parse_args()


def setup_logging(level_str: str) -> None:
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def list_excels(root: str) -> List[str]:
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        low = name.lower()
        if low.endswith(".xls") or low.endswith(".xlsx"):
            out.append(os.path.join(root, name))
    return out


def scan_label_spans(label_dir: str, subject: Optional[str]) -> List[LabelSpan]:
    out: List[LabelSpan] = []
    subj = subject.lower() if subject else None
    try:
        names = sorted(os.listdir(label_dir))
    except FileNotFoundError:
        logging.error("Label dir not found: %s", label_dir)
        return out
    for name in names:
        if not name.lower().endswith(".txt"):
            continue
        if subj and subj not in name.lower():
            continue
        path = os.path.join(label_dir, name)
        start, end = None, None
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        ts, _, _, _ = _parse_labelled_line(s)
                    except Exception:
                        continue
                    if start is None:
                        start = ts
                    end = ts
        except Exception as e:
            logging.warning("Skip label file (read error): %s (%s)", path, e)
            continue
        if start is not None and end is not None:
            out.append(LabelSpan(path=path, start=start, end=end))
    return out


def excel_time_window(xls_path: str, meta_sheet: str, start_event: str, time_col: str) -> Tuple[pd.Timestamp, pd.Timestamp, List[str]]:
    xls = pd.ExcelFile(xls_path)
    start_ts = read_metadata_start_time(xls, meta_sheet, start_event)
    # choose sensor sheets with time_col
    sheets = []
    for name in xls.sheet_names:
        try:
            head = xls.parse(name, nrows=1)
        except Exception:
            continue
        if any(str(c).strip().lower() == time_col.strip().lower() for c in head.columns):
            sheets.append(name)
    if not sheets:
        raise ValueError(f"No sensor sheets with '{time_col}' in {xls_path}")
    # compute end as max abs time across sheets
    max_abs = start_ts
    for s in sheets:
        df = xls.parse(s, usecols=[time_col])
        rel = pd.to_numeric(df[time_col], errors="coerce").astype(float)
        if not rel.empty and pd.notna(rel.max()):
            end_ts = start_ts + pd.to_timedelta(float(rel.max()), unit="s")
            if end_ts > max_abs:
                max_abs = end_ts
    return start_ts, max_abs, sheets


def overlap_seconds(a0: pd.Timestamp, a1: pd.Timestamp, b0: pd.Timestamp, b1: pd.Timestamp) -> float:
    lo = max(a0, b0)
    hi = min(a1, b1)
    if hi <= lo:
        return 0.0
    return (hi - lo).total_seconds()


def choose_best_label(ex_start: pd.Timestamp, ex_end: pd.Timestamp, spans: List[LabelSpan]) -> Optional[LabelSpan]:
    best = None
    best_sec = 0.0
    for sp in spans:
        sec = overlap_seconds(ex_start, ex_end, sp.start, sp.end)
        if sec > best_sec:
            best_sec = sec
            best = sp
    return best


def ensure_out_path(label_path: str, out_dir: Optional[str], out_suffix: Optional[str], excel_path: str) -> str:
    base = os.path.splitext(os.path.basename(label_path))[0]
    if out_suffix:
        suffix = out_suffix
    else:
        excel_stem = os.path.splitext(os.path.basename(excel_path))[0]
        suffix = f"__{excel_stem}"
    outname = f"{base}{suffix}.txt"
    root = out_dir if out_dir else os.path.dirname(label_path)
    # 出力ディレクトリを自動生成
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, outname)


def append_excel_to_label(
    label_path: str,
    excel_path: str,
    sheets: Optional[List[str]],
    time_col: str,
    meta_sheet: str,
    start_event: str,
    tolerance_ms: Optional[int],
    out_path: str,
):
    logger = logging.getLogger("append")
    t0 = time.time()
    # Read labelled
    logger.debug("Reading labelled file: %s", label_path)
    labelled_df, numeric_parts, labels, setids = read_labelled_file(label_path)
    # Open Excel and start time
    logger.debug("Opening Excel: %s", excel_path)
    xls = pd.ExcelFile(excel_path)
    start_ts = read_metadata_start_time(xls, meta_sheet, start_event)
    logger.debug("Start timestamp from metadata: %s", start_ts)
    # Decide sheets
    if not sheets:
        cand = []
        for name in xls.sheet_names:
            try:
                head = xls.parse(name, nrows=1)
            except Exception:
                continue
            if any(str(c).strip().lower() == time_col.strip().lower() for c in head.columns):
                cand.append(name)
        sheets = cand
    logger.info("Using sheets (%d): %s", len(sheets), ", ".join(sheets))
    # Build sensor dfs
    sensor_dfs: Dict[str, pd.DataFrame] = {}
    for name in sheets:
        logger.debug("Reading sensor sheet to abs time: %s", name)
        sdf = read_sensor_sheet_abs_times(xls, name, time_col, start_ts)
        logger.debug("  rows=%d cols=%d", len(sdf), len(sdf.columns))
        sensor_dfs[name] = sdf
    # Align and fill
    logger.info("Aligning to labelled times (tolerance_ms=%s, bfill_leading=True)", str(tolerance_ms))
    appended = build_appended_columns(
        labelled_times=labelled_df["time"],
        sensor_dfs=sensor_dfs,
        tolerance_ms=tolerance_ms,
        do_bfill_leading=True,
    )
    # Write
    logger.info("Writing output: %s", out_path)
    write_augmented_file(
        out_path=out_path,
        times=labelled_df["time"],
        numeric_parts=numeric_parts,
        labels=labels,
        setids=setids,
        appended=appended,
    )
    logger.info("Done (%.2fs)", time.time() - t0)


def main():
    args = parse_args()
    setup_logging(args.log_level)

    # スクリプト設定の適用（引数より優先させる場合）
    if USE_SCRIPT_OVERRIDES:
        if SUBJECT is not None:
            args.subject = (SUBJECT if str(SUBJECT).strip() != "" else None)
        if OUT_DIR is not None:
            args.out_dir = (OUT_DIR if str(OUT_DIR).strip() != "" else None)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    t_start = time.time()
    logging.info("=== Batch append start ===")
    logging.info("Config: subject=%r, out_dir=%r, excel_dir=%r, label_dir=%r, meta_sheet=%r, start_event=%r, time_col=%r",
                 args.subject, args.out_dir, args.excel_dir, args.label_dir, args.meta_sheet, args.start_event, args.time_col)

    excels = list_excels(args.excel_dir)
    if args.limit and len(excels) > args.limit:
        excels = excels[: args.limit]
    if not excels:
        logging.warning("No Excel files found in %s", args.excel_dir)
        return
    logging.info("Found %d Excel file(s).", len(excels))

    spans = scan_label_spans(args.label_dir, args.subject)
    if not spans:
        logging.warning("No labelled files found for given criteria (label_dir=%s, subject=%r).", args.label_dir, args.subject)
        return
    logging.info("Found %d labelled candidate file(s).", len(spans))

    plan: List[Tuple[str, Optional[LabelSpan], pd.Timestamp, pd.Timestamp]] = []

    # 計画作成
    logging.info("Building match plan...")
    for x in excels:
        try:
            ex_start, ex_end, _ = excel_time_window(x, args.meta_sheet, args.start_event, args.time_col)
        except Exception as e:
            logging.error("[SKIP] %s: failed to read time window: %s", x, e)
            continue
        match = choose_best_label(ex_start, ex_end, spans)
        plan.append((x, match, ex_start, ex_end))

    # 実行
    processed = 0
    matched = 0
    skipped = 0
    errors = 0
    N = len(plan)

    for i, (x, sp, exs, exe) in enumerate(plan, 1):
        if sp is None:
            logging.warning("[NO MATCH] [%d/%d] %s [%s ~ %s]", i, N, os.path.basename(x), exs, exe)
            skipped += 1
            continue

        out_path = ensure_out_path(sp.path, args.out_dir, args.out_suffix, x)
        logging.info(
            "[%d/%d] MATCH Excel=%s [%s ~ %s] -> Label=%s [%s ~ %s]\n         -> %s",
            i, N,
            os.path.basename(x), exs, exe,
            os.path.basename(sp.path), sp.start, sp.end,
            out_path,
        )

        if args.dry_run:
            matched += 1
            continue

        try:
            append_excel_to_label(
                label_path=sp.path,
                excel_path=x,
                sheets=args.sheets,
                time_col=args.time_col,
                meta_sheet=args.meta_sheet,
                start_event=args.start_event,
                tolerance_ms=args.tolerance_ms,
                out_path=out_path,
            )
            logging.info("  Wrote: %s", out_path)
            processed += 1
            matched += 1
        except Exception as e:
            logging.error("  [ERROR] %s -> %s: %s", x, sp.path, e)
            errors += 1

    # サマリ
    elapsed = time.time() - t_start
    logging.info("=== Summary ===")
    logging.info("Total planned: %d", N)
    logging.info("Matched: %d, Skipped(no match): %d, Processed(written): %d, Errors: %d", matched, skipped, processed, errors)
    logging.info("Elapsed: %.2fs", elapsed)
    logging.info("=== Batch append end ===")


if __name__ == "__main__":
    main()
