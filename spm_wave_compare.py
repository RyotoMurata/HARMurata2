from __future__ import annotations

"""
CLI: Compare level vs ramp_ascent waveforms via SPM(1D) paired t-test for a single subject.

- Discovers Labelled_data/** files for the specified subject
- Resamples trials to T points (0–100%)
- Aggregates within subject per set, then pairs level vs ramp_ascent across sets
- Runs spm1d paired t-test, summarizes significant clusters
- Saves figure and CSV summary

Dependencies: numpy, pandas, matplotlib, scipy, spm1d
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===== Optional channel metadata / log loader =====
try:
    import Build_RF_dataset as BRD  # provides FEATURE_NAMES and load_set_series for log-format files
    CHANNEL_NAMES: List[str] = list(getattr(BRD, "FEATURE_NAMES", []))
except Exception:
    BRD = None  # type: ignore
    CHANNEL_NAMES = [
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

UNITS_BY_CHANNEL: Dict[str, str] = {
    "acc_x": "g", "acc_y": "g", "acc_z": "g",
    "gyro_x": "deg/s", "gyro_y": "deg/s", "gyro_z": "deg/s",
    "quat_w": "", "quat_x": "", "quat_y": "", "quat_z": "",
    "encoder_angle": "deg",
    "grf_x": "N", "grf_y": "N", "grf_z": "N",
    "grt_x": "N·m", "grt_y": "N·m", "grt_z": "N·m",
}


def _infer_subject_from_filename(p: Path) -> Optional[str]:
    """Infer subject token from filename like 'ramp_t4_kaneishi_set-0004.txt'.

    Returns subject token between the second underscore and '_set-'.
    """
    stem = p.stem
    # Try to find '_set-'
    i = stem.find("_set-")
    if i <= 0:
        return None
    head = stem[:i]
    parts = head.split("_")
    if len(parts) >= 3:
        return parts[2]
    return None


# ============ Data discovery ============

def discover_files(root: Path, subject: str) -> List[Path]:
    """Recursively find candidate data files under root for the subject.

    Accepts .txt/.csv/.tsv files. Actual subject/condition filtering is done later.

    Example:
        discover_files(Path('Labelled_data'), 'S01')
    """
    exts = {".txt", ".csv", ".tsv"}
    paths: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            # quick heuristic: prefer files which name contains subject key
            if subject and (subject.lower() in p.name.lower() or subject.lower() in str(p.parent.name).lower()):
                paths.append(p)
            else:
                # also include; downstream filter will drop non-target rows
                paths.append(p)
    return paths


# ============ Loading ============

_COL_ALIASES: Dict[str, List[str]] = {
    "subject_id": ["subject", "subj", "subjectid", "sid"],
    "label": ["condition", "cond"],
    "trial_id": ["trial", "trialid", "trial_no", "trial_index"],
    "time_pct": ["time_percent", "pct", "percent", "phase_pct"],
    "value": ["y", "signal", "measure", "amp", "val"],
    "phase": ["gait_phase", "phase_name"],
    "set_id": ["session", "session_id", "set", "rec_id"],
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    newcols: Dict[str, str] = {}
    for std, aliases in _COL_ALIASES.items():
        if std in cols:
            newcols[cols[std]] = std
            continue
        found = None
        for a in aliases:
            if a in cols:
                found = cols[a]
                break
        if found is not None:
            newcols[found] = std
    return df.rename(columns=newcols)


def _read_one(path: Path) -> pd.DataFrame:
    # Try flexible CSV reading
    try:
        df = pd.read_csv(path, engine="python")
    except Exception:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            # whitespace-delimited
            df = pd.read_csv(path, delim_whitespace=True, engine="python")
    df = _standardize_columns(df)

    # Attach source and set_id if missing
    df["source"] = str(path)
    if "set_id" not in df.columns:
        # infer from parent dir name (e.g., Set-0001) or stem
        parent = path.parent.name
        stem = path.stem
        sid = parent if parent else stem
        df["set_id"] = sid
    return df


def load_and_concat(paths: Sequence[Path]) -> pd.DataFrame:
    """Load multiple files and concatenate rows.

    Returns a long DataFrame with standardized columns where possible.
    """
    frames: List[pd.DataFrame] = []
    for p in paths:
        try:
            frames.append(_read_one(p))
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


# ============ Log-format support (Labelled_data/*.txt) ============

def _resolve_channel(channel: Optional[str]) -> Tuple[int, str]:
    """Resolve channel argument to (index, name). Defaults to first channel if None.

    Accepts integer string, 0-based index, or channel name (case-insensitive).
    """
    if not CHANNEL_NAMES:
        raise SystemExit("[ERR] CHANNEL_NAMES is empty. Cannot resolve channel.")
    if channel is None:
        return 0, CHANNEL_NAMES[0]
    # try int
    try:
        idx = int(channel)
        if idx < 0 or idx >= len(CHANNEL_NAMES):
            raise ValueError
        return idx, CHANNEL_NAMES[idx]
    except Exception:
        pass
    # try name
    key = channel.strip().lower()
    for i, nm in enumerate(CHANNEL_NAMES):
        if nm.lower() == key:
            return i, nm
    raise SystemExit(f"[ERR] Unknown channel '{channel}'. Known: {CHANNEL_NAMES}")


def _iter_label_segments(labels: List[Optional[str]]) -> List[Tuple[int, int, Optional[str]]]:
    """Return segments (start_idx, end_idx, raw_label) of contiguous identical labels.

    raw_label can be None. Inclusive end index.
    """
    segs: List[Tuple[int, int, Optional[str]]] = []
    if not labels:
        return segs
    a = 0
    cur = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != cur:
            segs.append((a, i - 1, cur))
            a = i
            cur = labels[i]
    segs.append((a, len(labels) - 1, cur))
    return segs


def load_logs_concat_for_channel(paths: Sequence[Path], subject: str, channel_idx: int) -> pd.DataFrame:
    """Parse log-style labelled files using Build_RF_dataset.load_set_series and emit long DataFrame.

    Columns: subject_id, label, trial_id, time, value, set_id, source
    """
    if BRD is None or not hasattr(BRD, "load_set_series"):
        raise SystemExit("[ERR] Build_RF_dataset.load_set_series not available; cannot parse log-style files.")

    rows: List[Dict[str, object]] = []
    for p in paths:
        try:
            series = BRD.load_set_series(p)  # type: ignore[attr-defined]
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
            continue
        # Determine subject label per file
        file_subject = _infer_subject_from_filename(p) or (subject or "unknown")
        t = series.t_s
        data = series.data
        lbls = series.labels
        set_id = series.set_id if getattr(series, "set_id", None) else p.stem

        segs = _iter_label_segments(lbls)
        trial_no = 0
        for a, b, raw_lbl in segs:
            if raw_lbl is None:
                continue
            cond = _normalize_condition(str(raw_lbl))
            if cond not in {"level", "ramp_ascent"}:
                continue
            trial_no += 1
            trial_id = f"{trial_no:03d}"
            # slice segment
            for i in range(a, b + 1):
                rows.append({
                    "subject_id": file_subject,
                    "label": raw_lbl,
                    "trial_id": trial_id,
                    "time": float(t[i]),
                    "value": float(data[i, channel_idx]) if (0 <= channel_idx < data.shape[1]) else float("nan"),
                    "set_id": set_id,
                    "source": str(p),
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

# ============ Filtering ============

def _normalize_condition(s: str) -> Optional[str]:
    s0 = s.strip().lower().replace("-", "_").replace(" ", "_")
    # level synonyms
    if s0 in {"level", "flat", "lvl", "level_walk", "levelwalk"}:
        return "level"
    # ramp ascent synonyms
    if s0 in {"ramp_ascent", "ramp-ascent", "ascent", "ramp_up", "rampup"}:
        return "ramp_ascent"
    return None


def filter_subject_and_conditions(df: pd.DataFrame, subject: str) -> pd.DataFrame:
    """Keep only rows for subject and conditions {level, ramp_ascent}.

    Raises if required columns are missing.
    """
    req = {"subject_id", "label", "trial_id", "time_pct", "value"}
    # if time_pct missing but time present, we'll handle later in resampling
    if "time_pct" not in df.columns and "time" not in df.columns:
        missing = req - set(df.columns)
        if missing:
            raise SystemExit(f"[ERR] Missing required columns: {sorted(missing)}")

    if "subject_id" not in df.columns:
        raise SystemExit("[ERR] Column 'subject_id' is required (case-insensitive aliases supported).")
    if "label" not in df.columns:
        raise SystemExit("[ERR] Column 'label' (condition) is required.")
    if "trial_id" not in df.columns:
        raise SystemExit("[ERR] Column 'trial_id' is required.")
    if "value" not in df.columns:
        raise SystemExit("[ERR] Column 'value' is required.")

    df = df.copy()
    # normalize conditions
    df["condition"] = df["label"].astype(str).map(_normalize_condition)
    df = df[df["condition"].isin(["level", "ramp_ascent"])]
    # subject match (case-insensitive)
    df = df[df["subject_id"].astype(str).str.lower() == subject.lower()]
    return df


# ============ Resampling & aggregation ============

def _resample_one(x_pct: np.ndarray, y: np.ndarray, T: int) -> np.ndarray:
    xi = np.linspace(0.0, 100.0, T)
    # Clean and sort unique
    x = np.asarray(x_pct, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        return np.full(T, np.nan)
    # Normalize if not in [0,100]
    x0, x1 = float(np.nanmin(x)), float(np.nanmax(x))
    if x0 < -1e-6 or x1 > 101.0 or (abs(x0) > 1e-6 or abs(x1 - 100.0) > 1e-6):
        if x1 - x0 > 0:
            x = (x - x0) / (x1 - x0) * 100.0
    # Ensure increasing
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    # Remove duplicate x for interp
    xu, uniq_idx = np.unique(x, return_index=True)
    yu = y[uniq_idx]
    yi = np.interp(xi, xu, yu)
    return yi


def resample_to_T(df_sub: pd.DataFrame, T: int, unwrap_angle: bool = False) -> pd.DataFrame:
    """Resample each (set_id, condition, trial_id[, phase]) waveform to T points [0..100].

    If unwrap_angle=True, interprets 'value' as degrees, converts to radians, unwraps, then keeps unwrapped radians,
    defers conversion back to degrees to plotting stage.
    Returns a tidy DataFrame with one row per resampled waveform and columns:
        subject_id, set_id, condition, trial_id, phase (if present), values: np.ndarray shape (T,)
    """
    has_phase = "phase" in df_sub.columns
    groups = ["subject_id", "set_id", "condition", "trial_id"] + (["phase"] if has_phase else [])
    rows: List[Dict[str, object]] = []
    for keys, g in df_sub.groupby(groups, sort=False):
        g = g.sort_values(by=["time_pct"] if "time_pct" in g.columns else g.columns[0])
        if "time_pct" in g.columns:
            x = g["time_pct"].to_numpy(dtype=float)
        else:
            # fallback: try time column then normalize
            if "time" in g.columns:
                x = g["time"].to_numpy(dtype=float)
            else:
                raise SystemExit("[ERR] Neither 'time_pct' nor 'time' present for resampling.")
        y = g["value"].to_numpy(dtype=float)
        if unwrap_angle:
            # assume input is degrees; convert to radians and unwrap
            yr = np.deg2rad(y)
            yr = np.unwrap(yr)
            y = yr
        yi = _resample_one(x, y, T)
        rec: Dict[str, object] = {}
        if isinstance(keys, tuple):
            for k, v in zip(groups, keys):
                rec[k] = v
        else:
            rec.update({groups[0]: keys})
        rec["values"] = yi
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate_within_subject(df_T: pd.DataFrame, agg: str = "mean", phase: Optional[str] = None
                             ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Aggregate across trials within each set for the subject, per condition.

    Returns:
        Y_level: (N, T), Y_ramp: (N, T), subjects: list of set_id used (paired intersection)
    """
    if phase is not None and "phase" in df_T.columns:
        df_T = df_T[df_T["phase"].astype(str).str.lower() == phase.lower()]

    # Aggregate trials within each set_id x condition
    agg_func = np.nanmean if agg == "mean" else np.nanmedian
    per_set: Dict[Tuple[str, str], np.ndarray] = {}
    for (set_id, cond), grp in df_T.groupby(["set_id", "condition"]):
        if grp.empty:
            continue
        vals = np.vstack(grp["values"].to_list())  # (n_trials, T)
        y = agg_func(vals, axis=0)
        per_set[(set_id, cond)] = y

    # Paired intersection by set_id
    set_ids = sorted({sid for (sid, c) in per_set.keys() if (sid, "level") in per_set and (sid, "ramp_ascent") in per_set})
    Y_level = np.vstack([per_set[(sid, "level")] for sid in set_ids]) if set_ids else np.empty((0, 0))
    Y_ramp = np.vstack([per_set[(sid, "ramp_ascent")] for sid in set_ids]) if set_ids else np.empty((0, 0))
    return Y_level, Y_ramp, set_ids


# ============ SPM & effect size ============

def run_spm_paired(Y_level: np.ndarray, Y_ramp: np.ndarray, alpha: float):
    """Run SPM(1D) paired t-test.
    Returns (spm, res) where spm is the SPM{t} object and res is inference result.
    """
    try:
        import spm1d
    except Exception:
        raise SystemExit(
            "[ERR] spm1d is not installed. Install via: pip install spm1d"
        )
    if Y_level.shape != Y_ramp.shape:
        raise SystemExit(f"[ERR] Shape mismatch: {Y_level.shape} vs {Y_ramp.shape}")
    if Y_level.shape[0] < 3:
        raise SystemExit("[ERR] Number of paired observations N<3. Need at least 3 sets for SPM.")
    ttest = spm1d.stats.ttest_paired(Y_level, Y_ramp)
    res = ttest.inference(alpha=alpha, two_tailed=True, interp=True)
    return ttest, res


def compute_dz(Y_level: np.ndarray, Y_ramp: np.ndarray) -> np.ndarray:
    """Compute pointwise effect size dz = mean(diff)/std(diff)."""
    D = Y_ramp - Y_level
    m = np.nanmean(D, axis=0)
    s = np.nanstd(D, axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        dz = np.where(s > 0, m / s, 0.0)
    return dz


def _contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    if mask.size == 0:
        return runs
    i = 0
    n = mask.size
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def summarize_clusters(res, t_axis: np.ndarray, dz: np.ndarray, spm_z: np.ndarray) -> pd.DataFrame:
    """Create a summary DataFrame of significant clusters (FWER corrected).

    Attempts to map clusters from res.clusters if available; otherwise infers clusters
    by thresholding |spm_z| > zstar.
    """
    # threshold and supra mask
    zstar = getattr(res, 'zstar', None)
    if zstar is None:
        # fallback attribute names
        zstar = getattr(res, 'z_star', None)
    if zstar is None:
        raise RuntimeError("SPM inference object missing zstar threshold.")
    supra = np.abs(spm_z) > float(zstar)
    runs = _contiguous_runs(supra)

    # Try to get corrected p-values from res.clusters
    pvals: List[Optional[float]] = []
    clusters_obj = getattr(res, 'clusters', None)
    if clusters_obj:
        for cl in clusters_obj:
            p = getattr(cl, 'p', None)
            if p is None:
                p = getattr(cl, 'p_corrected', None)
            if p is None:
                p = getattr(cl, 'P', None)
            pvals.append(float(p) if p is not None else math.nan)

    rows: List[Dict[str, object]] = []
    for k, (a, b) in enumerate(runs, start=1):
        start_pct = float(t_axis[a])
        end_pct = float(t_axis[b])
        seg = spm_z[a:b+1]
        peak_t = float(seg[np.argmax(np.abs(seg))]) if seg.size else math.nan
        dz_mean = float(np.nanmean(dz[a:b+1])) if dz.size else math.nan
        p_corr = float(pvals[k-1]) if (k-1) < len(pvals) else math.nan
        rows.append({
            "subject_count": None,
            "T": t_axis.size,
            "alpha": None,
            "h0reject": bool(np.any(supra)),
            "n_clusters": len(runs),
            "cluster_id": k,
            "start_pct": start_pct,
            "end_pct": end_pct,
            "peak_t": peak_t,
            "p_corrected": p_corr,
            "dz_mean_in_cluster": dz_mean,
        })
    return pd.DataFrame(rows)


# ============ Plotting ============

def _mean_ci95(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = Y.shape[0]
    m = np.nanmean(Y, axis=0)
    s = np.nanstd(Y, axis=0, ddof=1)
    half = 1.96 * s / math.sqrt(max(n, 1)) if n > 0 else np.zeros_like(m)
    return m, half


def plot_results(t_axis: np.ndarray,
                 Y_level: np.ndarray,
                 Y_ramp: np.ndarray,
                 spm,
                 res,
                 out_png: Path,
                 subj_label: str,
                 unwrap_angle: bool = False,
                 ylabel: str = "value (unit)") -> None:
    """Plot mean±95%CI (top) and SPM{t} with threshold and significant bands (bottom)."""
    z = getattr(spm, 'z', None)
    if z is None:
        # Some spm1d returns object with .z attribute on res
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

    # Top: mean±95%CI
    mL, hL = _mean_ci95(Y_level)
    mR, hR = _mean_ci95(Y_ramp)

    # If unwrap_angle: convert radians back to degrees for plotting
    if unwrap_angle:
        mL, mR = np.rad2deg(mL), np.rad2deg(mR)
        hL, hR = np.rad2deg(hL), np.rad2deg(hR)
        ylabel_plot = "value (deg)"
    else:
        ylabel_plot = ylabel

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})

    ax1.plot(t_axis, mL, color="tab:blue", label="level")
    ax1.fill_between(t_axis, mL - hL, mL + hL, color="tab:blue", alpha=0.2)
    ax1.plot(t_axis, mR, color="tab:orange", label="ramp_ascent")
    ax1.fill_between(t_axis, mR - hR, mR + hR, color="tab:orange", alpha=0.2)
    ax1.set_ylabel(ylabel_plot)
    ax1.set_title(f"{subj_label}: level vs ramp_ascent")
    ax1.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax1.legend(loc="best")

    ax2.plot(t_axis, z, color="k", label="SPM{t}")
    ax2.axhline(float(zstar), linestyle="--", color="r", linewidth=1.2, label=f"z*={float(zstar):.3f}")
    ax2.axhline(-float(zstar), linestyle="--", color="r", linewidth=1.2)
    for (a, b) in runs:
        ax2.axvspan(t_axis[a], t_axis[b], color="gold", alpha=0.3)
    ax2.set_xlabel("% gait cycle")
    ax2.set_ylabel("SPM{t}")
    ax2.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax2.legend(loc="best")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============ Dummy data ============

def _make_dummy(subject: str, Traw: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    sets = [f"Set-{i:04d}" for i in range(1, 11)]
    rows: List[Dict[str, object]] = []
    for sid in sets:
        for trial in range(1, 6):
            # create non-uniform sampling
            x = np.linspace(0, 100, Traw) + rng.normal(0, 0.2, Traw)
            base = 10 * np.sin(2 * np.pi * (x / 100.0))
            noise = rng.normal(0, 0.7, Traw)
            y_level = base + noise
            y_ramp = base + noise + (np.where((x >= 30) & (x <= 60), 1.0, 0.0))
            for cond, Y in (("level", y_level), ("ramp_ascent", y_ramp)):
                for xi, yi in zip(x, Y):
                    rows.append({
                        "subject_id": subject,
                        "label": cond,
                        "trial_id": f"{trial}",
                        "time_pct": float(xi),
                        "value": float(yi),
                        "set_id": sid,
                    })
    return pd.DataFrame(rows)


# ============ Main ============

def main() -> None:
    ap = argparse.ArgumentParser(description="SPM(1D) paired t-test: level vs ramp_ascent (within-subject across sets)")
    ap.add_argument("--subject", required=False, default=None, help="Subject ID (exact). If omitted, auto-detect per filename and run per subject.")
    ap.add_argument("--root", type=Path, default=Path("Labelled_data"))
    ap.add_argument("--points", type=int, default=101, help="Number of resampled points (0–100%)")
    ap.add_argument("--aggregate", choices=["mean", "median"], default="mean")
    ap.add_argument("--unwrap-angle", action="store_true", help="Unwrap angle: degrees -> unwrap(rad) for analysis")
    ap.add_argument("--phase", choices=["stance", "swing"], default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    ap.add_argument("--ylabel", type=str, default="value (unit)")
    ap.add_argument("--channel", type=str, default="all", help="Channel name or index, or 'all' for all 17")
    ap.add_argument("--make-dummy", action="store_true", help="Use synthetic data for quick demo (ignores files)")
    args = ap.parse_args()

    print(f"[INFO] root={args.root}, subject={args.subject}")

    # Log-format fast path: run per-channel using Build_RF_dataset loader, then return
    if not args.make_dummy and BRD is not None and hasattr(BRD, "load_set_series"):
        files = discover_files(args.root, args.subject)
        if files:
            try:
                _ = BRD.load_set_series(files[0])  # type: ignore[attr-defined]
                logs_mode = True
            except Exception:
                logs_mode = False
        else:
            logs_mode = False

        if logs_mode:
            # Determine subjects to run
            if args.subject:
                subjects = [args.subject]
            else:
                subs: List[str] = []
                for p in files:
                    s = _infer_subject_from_filename(p)
                    if s and s not in subs:
                        subs.append(s)
                if not subs:
                    print("[ERR] Could not infer subject IDs from filenames. Please pass --subject.")
                    sys.exit(1)
                subjects = sorted(subs)

            combined_summaries: List[pd.DataFrame] = []
            ch_arg = (args.channel or "all").strip().lower()
            if ch_arg in {"all", "*"}:
                ch_list = [(i, nm) for i, nm in enumerate(CHANNEL_NAMES)]
            else:
                idx, nm = _resolve_channel(args.channel)
                ch_list = [(idx, nm)]

            for subj in subjects:
                for idx, nm in ch_list:
                    print(f"[INFO] Subject: {subj} | Channel: {nm} (#{idx})")
                    df_all = load_logs_concat_for_channel(files, subj, idx)
                    if df_all.empty:
                        print(f"[WARN] No data for subject '{subj}' channel '{nm}'. Skipping.")
                        continue
                    # Filter for subject and level/ramp_ascent
                    df_sub = filter_subject_and_conditions(df_all, subj)
                    if df_sub.empty:
                        print(f"[WARN] No rows for required conditions: subject '{subj}' channel '{nm}'. Skipping.")
                        continue
                    # Resample per trial
                    df_T = resample_to_T(df_sub, args.points, unwrap_angle=args.unwrap_angle and (nm == "encoder_angle"))
                    # Aggregate within subject (per set)
                    Y_level, Y_ramp, set_ids = aggregate_within_subject(df_T, agg=args.aggregate, phase=args.phase)
                    print(f"[INFO] collected sets (paired): {len(set_ids)} for subject {subj} channel {nm}")
                    if len(set_ids) < 3:
                        print(f"[WARN] Not enough paired sets (N<3): subject '{subj}' channel '{nm}'. Skipping SPM.")
                        continue
                    # Run SPM
                    ttest, res = run_spm_paired(Y_level, Y_ramp, alpha=args.alpha)
                    h0 = bool(getattr(res, 'h0reject', False))
                    print(f"[INFO] T={args.points}, alpha={args.alpha}, h0reject={h0}")
                    # Summaries
                    dz = compute_dz(Y_level, Y_ramp)
                    t_axis = np.linspace(0.0, 100.0, args.points)
                    spm_z = getattr(ttest, 'z', None)
                    if spm_z is None:
                        spm_z = getattr(res, 'z', None)
                    spm_z = np.asarray(spm_z, dtype=float)
                    df_sum = summarize_clusters(res, t_axis, dz, spm_z)
                    if df_sum.empty:
                        df_sum = pd.DataFrame([
                            {"subject_count": Y_level.shape[0], "T": args.points, "alpha": args.alpha, "h0reject": h0,
                             "n_clusters": 0, "cluster_id": None, "start_pct": None, "end_pct": None, "peak_t": None,
                             "p_corrected": None, "dz_mean_in_cluster": None}
                        ])
                    else:
                        df_sum["subject_count"] = Y_level.shape[0]
                        df_sum["alpha"] = args.alpha
                    df_sum["channel"] = nm
                    df_sum["subject_id"] = subj

                    # Output per subject x channel
                    outdir = args.outdir
                    outdir.mkdir(parents=True, exist_ok=True)
                    unit = UNITS_BY_CHANNEL.get(nm, "")
                    ylab = args.ylabel if args.ylabel != "value (unit)" else (f"{nm} ({unit})" if unit else nm)
                    out_png = outdir / f"spm_result_{subj}_{nm}.png"
                    plot_results(t_axis, Y_level, Y_ramp, ttest, res, out_png, f"{subj} · {nm}", unwrap_angle=args.unwrap_angle and (nm == "encoder_angle"), ylabel=ylab)
                    out_csv = outdir / f"spm_summary_{subj}_{nm}.csv"
                    df_sum.to_csv(out_csv, index=False)
                    print(f"[INFO] Saved figure -> {out_png}")
                    print(f"[INFO] Saved summary -> {out_csv}")
                    combined_summaries.append(df_sum)

            if combined_summaries:
                df_all_sum = pd.concat(combined_summaries, axis=0, ignore_index=True)
                out_all = args.outdir / "spm_summary_all.csv"
                df_all_sum.to_csv(out_all, index=False)
                print(f"[INFO] Saved combined summary -> {out_all}")
            else:
                print("[WARN] No subject/channel produced a valid SPM result.")
            return

    # Load
    if args.make_dummy:
        df_all = _make_dummy(args.subject)
    else:
        files = discover_files(args.root, args.subject)
        if not files:
            print(f"[ERR] No files found under {args.root}")
            print("       Try --make-dummy for a quick demo.")
            sys.exit(1)
        df_all = load_and_concat(files)
        if df_all.empty:
            print(f"[ERR] Failed to load any valid data from {len(files)} files.")
            sys.exit(1)

    # Filter
    df_sub = filter_subject_and_conditions(df_all, args.subject)
    if df_sub.empty:
        print("[ERR] No rows for the subject and required conditions {level, ramp_ascent}.")
        sys.exit(1)

    # Check condition coverage
    has_level = (df_sub["condition"] == "level").any()
    has_ramp = (df_sub["condition"] == "ramp_ascent").any()
    if not (has_level and has_ramp):
        print("[ERR] Missing one or both conditions for the subject.")
        print("      Found counts:")
        print(df_sub["condition"].value_counts())
        sys.exit(1)

    # Resample per trial
    df_T = resample_to_T(df_sub, args.points, unwrap_angle=args.unwrap_angle)

    # Aggregate within subject (per set)
    Y_level, Y_ramp, set_ids = aggregate_within_subject(df_T, agg=args.aggregate, phase=args.phase)
    print(f"[INFO] collected sets: pairs={len(set_ids)} (level & ramp_ascent)")
    if len(set_ids) < 3:
        print("[ERR] Not enough paired sets (N<3). Provide at least 3 sets to run SPM.")
        sys.exit(1)

    # Run SPM paired t-test
    ttest, res = run_spm_paired(Y_level, Y_ramp, alpha=args.alpha)
    h0 = bool(getattr(res, 'h0reject', False))
    print(f"[INFO] T={args.points}, alpha={args.alpha}")
    print(f"[INFO] SPM h0reject={h0}")

    # Effect size and cluster summary
    dz = compute_dz(Y_level, Y_ramp)
    t_axis = np.linspace(0.0, 100.0, args.points)
    spm_z = getattr(ttest, 'z', None)
    if spm_z is None:
        spm_z = getattr(res, 'z', None)
    spm_z = np.asarray(spm_z, dtype=float)
    df_sum = summarize_clusters(res, t_axis, dz, spm_z)
    if df_sum.empty:
        df_sum = pd.DataFrame([{
            "subject_count": Y_level.shape[0],
            "T": args.points,
            "alpha": args.alpha,
            "h0reject": h0,
            "n_clusters": 0,
            "cluster_id": None,
            "start_pct": None,
            "end_pct": None,
            "peak_t": None,
            "p_corrected": None,
            "dz_mean_in_cluster": None,
        }])
    else:
        df_sum["subject_count"] = Y_level.shape[0]
        df_sum["alpha"] = args.alpha

    # Logs per cluster
    ncl = int(df_sum["n_clusters"].max()) if not df_sum.empty else 0
    if ncl > 0:
        for _, row in df_sum.iterrows():
            if pd.isna(row["cluster_id"]):
                continue
            print(
                f"[INFO] Cluster#{int(row['cluster_id'])}: {row['start_pct']:.1f}–{row['end_pct']:.1f}%GC, "
                f"peak_t={row['peak_t']:.2f}, p(FWER)={row['p_corrected'] if not pd.isna(row['p_corrected']) else 'NA'}, "
                f"dz_mean={row['dz_mean_in_cluster']:.3f}"
            )

    # Output figure and CSV
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / f"spm_result_{args.subject}.png"
    plot_results(t_axis, Y_level, Y_ramp, ttest, res, out_png, args.subject, unwrap_angle=args.unwrap_angle, ylabel=args.ylabel)
    out_csv = outdir / f"spm_summary_{args.subject}.csv"
    df_sum.to_csv(out_csv, index=False)
    print(f"[INFO] Saved figure -> {out_png}")
    print(f"[INFO] Saved summary -> {out_csv}")


if __name__ == "__main__":
    main()
