from __future__ import annotations

"""
SPM per phase using grf_z hysteresis thresholds (log/CSV compatible).

- Loads labelled log-format files (Labelled_data/*_set-*.txt) via Analyze_phase_curves.load_set_series
  or CSV/TSV files which contain at least the 17 base channels and a label column.
- Segments stance/swing within each contiguous label region using hysteresis on grf_z.
- Normalizes each segment to 0..100% and resamples to T points.
- Aggregates within-set per condition and phase.
- Runs SPM(1D) paired t-test between two conditions within each phase (default: level vs ramp_ascent).
- Outputs figures and CSV summaries per subject and channel (and phase).

Dependencies: numpy, pandas, matplotlib, scipy, spm1d
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Comparison pair you can edit here ----
COND_A = "ramp_ascent"  # 例: "level" に変更
COND_B = "stair_ascent"   # 例: "ramp_ascent" や "stair_ascent" に変更


# Reuse loaders and metadata from Analyze_phase_curves
try:
    from Analyze_phase_curves import (
        FEATURE_NAMES,
        NAME_TO_INDEX,
        load_set_series as load_log_series,
        segment_by_label,
        normalize_and_resample,
        Phase as _PhaseClass,
    )
except Exception as e:  # fallback minimal metadata
    FEATURE_NAMES = [
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
    NAME_TO_INDEX = {n: i for i, n in enumerate(FEATURE_NAMES)}
    load_log_series = None  # type: ignore
    segment_by_label = None  # type: ignore
    normalize_and_resample = None  # type: ignore
    class _PhaseClass:
        STANCE = "stance"
        SWING = "swing"


PhaseNames = (_PhaseClass.STANCE, _PhaseClass.SWING)


# ---------- Utilities ----------

def _infer_subject_from_filename(p: Path) -> Optional[str]:
    stem = p.stem
    i = stem.find("_set-")
    if i <= 0:
        return None
    head = stem[:i]
    parts = head.split("_")
    if len(parts) >= 3:
        return parts[2]
    return None


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


# ---------- CSV reader (flexible) ----------

_COL_ALIASES: Dict[str, List[str]] = {
    "label": ["condition", "cond"],
    "time": ["time_s", "t", "timestamp", "sec", "seconds"],
    "set_id": ["session", "session_id", "set", "rec_id"],
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    newcols: Dict[str, str] = {}
    for std, aliases in _COL_ALIASES.items():
        if std in cols:
            newcols[cols[std]] = std
            continue
        for a in aliases:
            if a in cols:
                newcols[cols[a]] = std
                break
    return df.rename(columns=newcols)


def _read_csv_like(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, List[Optional[str]], Optional[str]]]:
    """Attempt to read a CSV/TSV with 17 channels + label and optional time column.

    Returns (t_s, data[N,17], labels[list], set_id) or None on failure.
    """
    try:
        df = pd.read_csv(path, engine="python")
    except Exception:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            try:
                df = pd.read_csv(path, delim_whitespace=True, engine="python")
            except Exception:
                return None
    df = _standardize_columns(df)
    # Require all channels present
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        return None
    # Time column optional; if absent, use index as sample time (uniform spacing)
    if "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    else:
        t = np.arange(len(df), dtype=float)
        t = t - (t[0] if t.size > 0 else 0.0)
    data = df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    labels: List[Optional[str]] = df["label"].astype(str).tolist() if "label" in df.columns else [None] * len(df)
    set_id: Optional[str] = None
    if "set_id" in df.columns:
        sid = str(df["set_id"].iloc[0])
        set_id = sid
    else:
        set_id = path.parent.name or path.stem
    return t, data, labels, set_id


# ---------- Label normalization and conditions ----------

def _normalize_condition(s: str) -> Optional[str]:
    s0 = s.strip().lower().replace("-", "_").replace(" ", "_")
    if s0 in {"level", "flat", "lvl", "level_walk", "levelwalk"}:
        return "level"
    if s0 in {"ramp_ascent", "ramp-ascent", "ascent", "ramp_up", "rampup"}:
        return "ramp_ascent"
    if s0 in {"stair_ascent", "stairs_ascent", "stair_up", "stairs_up", "stairs"}:
        return "stair_ascent"
    # ★ ここを追加：stop を正規化
    if s0 in {"stop", "stand", "standing", "stationary", "halt"}:
        return "stop"
    return None



# ---------- SPM helpers ----------

def run_spm_paired(Y_a: np.ndarray, Y_b: np.ndarray, alpha: float):
    import spm1d
    ttest = spm1d.stats.ttest_paired(Y_a, Y_b)
    res = ttest.inference(alpha, two_tailed=True, interp=True)
    return ttest, res


def compute_dz(Y_a: np.ndarray, Y_b: np.ndarray) -> np.ndarray:
    """Pointwise effect size dz over time:
       dz(t) = mean_over_sets( Y_b - Y_a ) / std_over_sets( Y_b - Y_a )
    """
    D = Y_b - Y_a                  # (N_sets, T)
    m = np.nanmean(D, axis=0)      # (T,)
    s = np.nanstd(D, axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        dz = np.where(s > 0, m / s, 0.0)
    return dz



def _contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    if mask.size == 0:
        return runs
    in_run = False
    a = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True
            a = i
        elif not v and in_run:
            in_run = False
            runs.append((a, i - 1))
    if in_run:
        runs.append((a, mask.size - 1))
    return runs


def summarize_clusters(res, t_axis: np.ndarray, dz_subj: np.ndarray, z_curve: np.ndarray) -> pd.DataFrame:
    import pandas as pd
    supra = np.abs(z_curve) > float(getattr(res, 'zstar', getattr(res, 'z_star', np.nan)))
    runs = _contiguous_runs(supra)
    rows: List[Dict[str, object]] = []
    for k, (a, b) in enumerate(runs, start=1):
        start_pct = float(t_axis[a])
        end_pct = float(t_axis[b])
        peak_t = float(np.nanmax(np.abs(z_curve[a:b+1])))
        dz_mean = float(np.nanmean(dz_subj[a:b+1])) if dz_subj.size else math.nan
        # p-values per cluster may be available as res.cluster_p or a sequence on res
        pvals = getattr(res, 'p', None)
        if pvals is None:
            pvals = getattr(res, 'p_set', None)
        if pvals is None:
            p_corr = math.nan
        else:
            try:
                p_corr = float(list(pvals)[k-1])
            except Exception:
                p_corr = math.nan
        rows.append({
            "n_clusters": len(runs),
            "cluster_id": k,
            "start_pct": start_pct,
            "end_pct": end_pct,
            "peak_t": peak_t,
            "p_corrected": p_corr,
            "dz_mean_in_cluster": dz_mean,
        })
    return pd.DataFrame(rows)

def plot_results(t_axis: np.ndarray,
                 Y_a: np.ndarray,
                 Y_b: np.ndarray,
                 spm,
                 res,
                 out_png: Path,
                 subj_label: str,
                 chan_label: str,
                 phase_label: str,
                 labelA: str,
                 labelB: str,
                 ylabel: str = "value (unit)") -> None:
    z = getattr(spm, 'z', None)
    if z is None:
        z = getattr(res, 'z', None)
    if z is None:
        raise RuntimeError("Cannot find SPM{t} curve (z).")
    z = np.asarray(z, dtype=float)
    zstar = getattr(res, 'zstar', None)
    if zstar is None:
        zstar = getattr(res, 'z_star', None)
    if zstar is None:
        raise RuntimeError("Cannot find threshold z* in SPM inference result.")
    supra = np.abs(z) > float(zstar)
    runs = _contiguous_runs(supra)

    def _mean_ci95(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = Y.shape[0]
        m = np.nanmean(Y, axis=0)
        s = np.nanstd(Y, axis=0, ddof=1)
        half = 1.96 * s / math.sqrt(max(n, 1)) if n > 0 else np.zeros_like(m)
        return m, half

    mA, hA = _mean_ci95(Y_a)
    mB, hB = _mean_ci95(Y_b)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})
    ax1.plot(t_axis, mA, color="tab:blue", label=labelA)
    ax1.fill_between(t_axis, mA - hA, mA + hA, color="tab:blue", alpha=0.2)
    ax1.plot(t_axis, mB, color="tab:orange", label=labelB)
    ax1.fill_between(t_axis, mB - hB, mB + hB, color="tab:orange", alpha=0.2)
    ax1.set_ylabel(ylabel)
    ax1.set_title(f"{subj_label} | {chan_label} | {phase_label} : {labelA} vs {labelB}")
    ax1.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax1.legend(loc="best")

    ax2.plot(t_axis, z, color="k", label="SPM{t}")
    ax2.axhline(float(zstar), linestyle="--", color="r", linewidth=1.2, label=f"z*={float(zstar):.3f}")
    ax2.axhline(-float(zstar), linestyle="--", color="r", linewidth=1.2)
    for (a, b) in runs:
        ax2.axvspan(t_axis[a], t_axis[b], color="gold", alpha=0.3)
    ax2.set_xlabel("% gait cycle (phase-normalized)")
    ax2.set_ylabel("SPM{t}")
    ax2.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax2.legend(loc="best")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- Core pipeline ----------

@dataclass
class SegSample:
    subject: str
    set_id: str
    condition: str
    phase: str
    channel: str
    values: np.ndarray  # (T,)


def _iter_segments_from_log(path: Path, low_thr: float, high_thr: float, T: int, min_seg_ms: float,
                            subject_fallback: Optional[str]) -> List[Tuple[str, str, str, str, np.ndarray]]:
    """Yield (subject, set_id, condition, phase, Y(T,17)) per segment from a log-format file."""
    if load_log_series is None or segment_by_label is None or normalize_and_resample is None:
        raise SystemExit("Analyze_phase_curves loader/segment functions not available.")
    series = load_log_series(path)
    subj = _infer_subject_from_filename(path) or subject_fallback or "unknown"
    set_id = series.set_id if getattr(series, "set_id", None) else path.stem
    segs = segment_by_label(series, low_thr, high_thr)
    keep: List[Tuple[str, str, str, str, np.ndarray]] = []
    min_len_s = min_seg_ms / 1000.0
    for seg in segs:
        if seg.label is None:
            continue
        cond = _normalize_condition(str(seg.label))
        if cond is None:
            continue
        if seg.t[-1] - seg.t[0] < min_len_s:
            continue
        _, phase, _, Y = normalize_and_resample(seg, T)
        keep.append((subj, set_id, cond, phase, Y))
    return keep


def _iter_segments_from_csv(path: Path, low_thr: float, high_thr: float, T: int, min_seg_ms: float,
                            subject_fallback: Optional[str]) -> List[Tuple[str, str, str, str, np.ndarray]]:
    got = _read_csv_like(path)
    if got is None:
        return []
    t, data, labels, set_id = got
    subj = _infer_subject_from_filename(path) or subject_fallback or "unknown"
    # Segment by contiguous label and then by hysteresis on grf_z
    # Build a temporary series-like structure
    class _Series:
        pass
    ser = _Series()
    ser.t_s = t
    ser.data = data
    ser.labels = labels
    ser.set_id = set_id
    # Implement minimal local segmentation using Analyze_phase_curves if available
    if segment_by_label is None or normalize_and_resample is None:
        raise SystemExit("segment_by_label/normalize_and_resample not available; please ensure Analyze_phase_curves.py is present.")
    segs = segment_by_label(ser, low_thr, high_thr)  # type: ignore[arg-type]
    keep: List[Tuple[str, str, str, str, np.ndarray]] = []
    min_len_s = min_seg_ms / 1000.0
    for seg in segs:
        if seg.label is None:
            continue
        cond = _normalize_condition(str(seg.label))
        if cond is None:
            continue
        if seg.t[-1] - seg.t[0] < min_len_s:
            continue
        _, phase, _, Y = normalize_and_resample(seg, T)
        keep.append((subj, set_id, cond, phase, Y))
    return keep


def collect_segment_samples(files: Sequence[Path], low_thr: float, high_thr: float, T: int, min_seg_ms: float,
                            subject: Optional[str]) -> List[SegSample]:
    samples: List[SegSample] = []
    for p in files:
        ext = p.suffix.lower()
        rows: List[Tuple[str, str, str, str, np.ndarray]]
        if ext in {".txt", ".log"}:
            rows = _iter_segments_from_log(p, low_thr, high_thr, T, min_seg_ms, subject)
        else:
            rows = _iter_segments_from_csv(p, low_thr, high_thr, T, min_seg_ms, subject)
        for subj, set_id, cond, phase, Y in rows:
            # Emit one sample per channel
            for j, ch in enumerate(FEATURE_NAMES):
                samples.append(SegSample(subject=subj, set_id=str(set_id), condition=cond, phase=phase,
                                         channel=ch, values=Y[:, j].astype(float)))
    # Filter by subject if specified
    if subject:
        samples = [s for s in samples if s.subject.lower() == subject.lower()]
    return samples


def aggregate_within_set(samples: List[SegSample], agg: str = "mean") -> Dict[Tuple[str, str, str, str], np.ndarray]:
    """Aggregate multiple segments within a set to one waveform per (subject,set,condition,phase,channel).

    Returns dict keyed by (subject,set_id,condition,phase,channel) -> values(T,)
    """
    from collections import defaultdict
    buckets: Dict[Tuple[str, str, str, str, str], List[np.ndarray]] = defaultdict(list)
    for s in samples:
        buckets[(s.subject, s.set_id, s.condition, s.phase, s.channel)].append(s.values)
    out: Dict[Tuple[str, str, str, str], np.ndarray] = {}
    for (subj, set_id, cond, phase, ch), arrs in buckets.items():
        A = np.stack(arrs, axis=0)
        if agg == "median":
            v = np.nanmedian(A, axis=0)
        else:
            v = np.nanmean(A, axis=0)
        out[(subj, set_id, cond, phase)] = v  # channel collapsed later per selection
    # Note: this returns only last channel per key; we'll not use this helper directly per channel
    return out


def aggregate_within_set_per_channel(samples: List[SegSample], agg: str = "mean") -> Dict[Tuple[str, str, str, str, str], np.ndarray]:
    from collections import defaultdict
    buckets: Dict[Tuple[str, str, str, str, str], List[np.ndarray]] = defaultdict(list)
    for s in samples:
        buckets[(s.subject, s.set_id, s.condition, s.phase, s.channel)].append(s.values)
    out: Dict[Tuple[str, str, str, str, str], np.ndarray] = {}
    for key, arrs in buckets.items():
        A = np.stack(arrs, axis=0)
        if agg == "median":
            v = np.nanmedian(A, axis=0)
        else:
            v = np.nanmean(A, axis=0)
        out[key] = v
    return out


def paired_sets_for_conditions(agg: Dict[Tuple[str, str, str, str, str], np.ndarray], subj: str, channel: str, phase: str,
                               condA: str = "level", condB: str = "ramp_ascent") -> Tuple[List[str], np.ndarray, np.ndarray]:
    # Find all sets for which both conditions exist
    from collections import defaultdict
    per_set: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for (s, set_id, cond, ph, ch), v in agg.items():
        if s != subj or ch != channel or ph != phase:
            continue
        per_set[set_id][cond] = v
    set_ids = [sid for sid, d in per_set.items() if (condA in d and condB in d)]
    set_ids = sorted(set_ids)
    A = np.stack([per_set[sid][condA] for sid in set_ids], axis=0) if set_ids else np.empty((0,))
    B = np.stack([per_set[sid][condB] for sid in set_ids], axis=0) if set_ids else np.empty((0,))
    return set_ids, A, B


def main() -> None:
    ap = argparse.ArgumentParser(description="SPM per phase using grf_z hysteresis thresholds (log/CSV compatible)")
    ap.add_argument("inputs", type=Path, nargs="*", help="Input files or globs (.txt/.csv/.tsv)")
    ap.add_argument("--root", type=Path, default=Path("Labelled_data"), help="Root directory to search if inputs omitted")
    ap.add_argument("--subject", type=str, default=None, help="Subject ID (if omitted, infer from filenames and run per subject)")
    ap.add_argument("--low-thr", type=float, default=-60.0, help="Swing start threshold on grf_z (below => swing)")
    ap.add_argument("--high-thr", type=float, default=-59.0, help="Stance start threshold on grf_z (above => stance)")
    ap.add_argument("--points", type=int, default=101, help="Resampled points per segment [0..100]")
    ap.add_argument("--min-seg-ms", type=float, default=150.0, help="Minimum segment duration [ms]")
    ap.add_argument("--aggregate", choices=["mean", "median"], default="mean")
    ap.add_argument("--phase", choices=["stance", "swing", "both"], default="both")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--channel", type=str, default="all", help="Channel name or index, or 'all' for all 17")
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    args = ap.parse_args()

    # 比較する条件（スクリプト先頭で定義した定数を使う）
    condA, condB = COND_A, COND_B

    # Resolve files
    files = expand_input_paths(args.inputs) if args.inputs else list(args.root.rglob("*"))
    files = [p for p in files if p.is_file() and p.suffix.lower() in {".txt", ".csv", ".tsv", ".log"}]
    if not files:
        raise SystemExit("No input files found.")

    # Determine subjects
    if args.subject:
        subjects = [args.subject]
    else:
        subs: List[str] = []
        for p in files:
            s = _infer_subject_from_filename(p)
            if s and s not in subs:
                subs.append(s)
        if not subs:
            subs = ["unknown"]
        subjects = sorted(subs)

    # Channel selection
    ch_arg = (args.channel or "all").strip().lower()
    if ch_arg in {"all", "*"}:
        ch_list = [(i, nm) for i, nm in enumerate(FEATURE_NAMES)]
    else:
        try:
            idx = int(ch_arg)
            if idx < 0 or idx >= len(FEATURE_NAMES):
                raise ValueError
            ch_list = [(idx, FEATURE_NAMES[idx])]
        except Exception:
            # by name
            nm = None
            for i, n in enumerate(FEATURE_NAMES):
                if n.lower() == ch_arg:
                    nm = n
                    idx = i
                    break
            if nm is None:
                raise SystemExit(f"Unknown channel '{args.channel}'. Known: {FEATURE_NAMES}")
            ch_list = [(idx, nm)]

    # Collect all segment samples
    all_samples = collect_segment_samples(files, args.low_thr, args.high_thr, args.points, args.min_seg_ms, None)
    if not all_samples:
        raise SystemExit("No segments after thresholding/filters. Adjust --low-thr/--high-thr.")

    # Aggregate within set per channel
    agg = aggregate_within_set_per_channel(all_samples, agg=args.aggregate)

    # For each subject, channel, and phase -> pair sets and run SPM
    phases_to_run = [args.phase] if args.phase in {"stance", "swing"} else ["stance", "swing"]
    T = args.points
    t_axis = np.linspace(0.0, 100.0, T)
    # ランキング格納: key=(subject, phase) -> List[metrics per channel]
    rankings: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for subj in subjects:
        for ch_idx, ch_name in ch_list:
            for ph in phases_to_run:
                # ここで比較条件を適用
                set_ids, Y_A, Y_B = paired_sets_for_conditions(
                    agg, subj, ch_name, ph, condA=condA, condB=condB
                )
                print(f"[INFO] Subject={subj} Channel={ch_name} Phase={ph} paired_sets={len(set_ids)} ({condA} vs {condB})")
                if len(set_ids) < 3:
                    print("  [WARN] Not enough paired sets (N<3). Skipping SPM.")
                    continue
                # Run SPM paired t-test
                try:
                    ttest, res = run_spm_paired(Y_A, Y_B, alpha=args.alpha)
                except Exception as e:
                    print(f"  [ERR] SPM failed: {e}")
                    continue
                h0 = bool(getattr(res, 'h0reject', False))
                spm_z = getattr(ttest, 'z', None)
                if spm_z is None:
                    spm_z = getattr(res, 'z', None)
                spm_z = np.asarray(spm_z, dtype=float)
                dz = compute_dz(Y_A, Y_B)  # （condB - condA）の符号で解釈
                # 追加（この2行を dz 計算の直後に入れる）
                mean_abs_dz_val = float(np.nanmean(np.abs(dz)))
                df_sum = summarize_clusters(res, t_axis, dz, spm_z)
                if df_sum.empty:
                    df_sum = pd.DataFrame([{
                        "subject": subj,
                        "channel": ch_name,
                        "phase": ph,
                        "T": T,
                        "alpha": args.alpha,
                        "h0reject": h0,
                        "n_clusters": 0,
                        "cluster_id": None,
                        "start_pct": None,
                        "end_pct": None,
                        "peak_t": None,
                        "p_corrected": None,
                        "dz_mean_in_cluster": None,
                        "mean_abs_dz": mean_abs_dz_val,
                    }])
                else:
                    df_sum.insert(0, "subject", subj)
                    df_sum.insert(1, "channel", ch_name)
                    df_sum.insert(2, "phase", ph)
                    df_sum["T"] = T
                    df_sum["alpha"] = args.alpha
                    df_sum["h0reject"] = bool(getattr(res, 'h0reject', False))
                    # 追加（非空ケースでも列を付与）
                    df_sum["mean_abs_dz"] = mean_abs_dz_val

                # 出力先: results/<subject>/<phase>/
                outdir = args.outdir / subj / ph
                outdir.mkdir(parents=True, exist_ok=True)

                # ファイル名に条件名を含める
                out_png = outdir / f"spm_phase_{subj}_{ch_name}_{ph}_{condA}_vs_{condB}.png"
                plot_results(t_axis, Y_A, Y_B, ttest, res, out_png, subj, ch_name, ph, condA, condB)

                out_csv = outdir / f"spm_summary_{subj}_{ch_name}_{ph}_{condA}_vs_{condB}.csv"
                df_sum.to_csv(out_csv, index=False)
                print(f"  [OK] Saved figure -> {out_png}")
                print(f"  [OK] Saved summary -> {out_csv}")

                # ---- ランキング用メトリクスを追加 ----
                if (df_sum["n_clusters"] > 0).any():
                    n_sig_clusters = int(df_sum["n_clusters"].iloc[0])
                    best_p = float(df_sum["p_corrected"].dropna().min()) if df_sum["p_corrected"].notna().any() else 1.0
                    max_peak_t = float(np.nanmax(np.abs(df_sum["peak_t"].to_numpy())))
                    total_sig_span = float(np.nansum((df_sum["end_pct"] - df_sum["start_pct"]).to_numpy()))
                    if "dz_mean_in_cluster" in df_sum.columns:
                        max_abs_dz = float(np.nanmax(np.abs(df_sum["dz_mean_in_cluster"].to_numpy())))
                    else:
                        max_abs_dz = float("nan")
                    has_sig = True
                else:
                    n_sig_clusters = 0
                    best_p = 1.0
                    max_peak_t = 0.0
                    total_sig_span = 0.0
                    max_abs_dz = float("nan")
                    has_sig = False

                key = (subj, ph)
                rankings.setdefault(key, []).append({
                    "subject": subj,
                    "phase": ph,
                    "channel": ch_name,
                    "condA": condA,
                    "condB": condB,
                    "has_significance": has_sig,
                    "n_sig_clusters": n_sig_clusters,
                    "best_p_corrected": best_p,
                    "max_abs_t": max_peak_t,
                    "total_sig_span_pct": total_sig_span,
                    "max_abs_dz_mean_in_clusters": max_abs_dz,
                    # 追加
                    "mean_abs_dz": mean_abs_dz_val,
                })
        # ここは `for ch_idx, ch_name in ch_list:` を抜けた直後（= 全チャネル処理後）
        for ph in phases_to_run:
            key = (subj, ph)
            if key in rankings and rankings[key]:
                df_rank = pd.DataFrame(rankings[key])
                # ソート: 有意あり→有意領域総延長↓→クラスター数↓→p値↑→|t|↓
                df_rank.sort_values(
                    by=["has_significance", "total_sig_span_pct", "n_sig_clusters", "best_p_corrected", "max_abs_t"],
                    ascending=[False,               False,               False,            True,               False],
                    inplace=True
                )
                rank_outdir = args.outdir / subj / ph
                rank_outdir.mkdir(parents=True, exist_ok=True)
                rank_csv = rank_outdir / f"ranking_{subj}_{ph}_{condA}_vs_{condB}.csv"
                df_rank.to_csv(rank_csv, index=False)
                print(f"[OK] Saved ranking -> {rank_csv}")

        # 既存: df_rank.sort_values(...) の直後あたりに追加
        desired_cols = [
            "subject","phase","channel","condA","condB",
            "has_significance","n_sig_clusters","best_p_corrected",
            "max_abs_t","total_sig_span_pct",
            "max_abs_dz_mean_in_clusters","mean_abs_dz"  # ←ここで順序を保証
        ]
        # 存在する列だけを選んで並べ替え（将来列が増えても安全）
        df_rank = df_rank[[c for c in desired_cols if c in df_rank.columns] + 
                        [c for c in df_rank.columns if c not in desired_cols]]





if __name__ == "__main__":
    main()

