from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm, rcParams as _rc

# Japanese font (auto-pick if available)
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


# Defaults
DEFAULT_FILE1: Path = Path("Labelled_data/ramp_t1_kaneishi_set-0001.txt")
DEFAULT_FILE2: Path = Path("Labelled_data/stair_t1_kaneishi_set-0001.txt")
DEFAULT_FEATURE: Union[str, int] = "grf_x"  # or 0..N index
DEFAULT_UNIT: Optional[str] = None
DEFAULT_TITLE: Optional[str] = None
DEFAULT_SAVE: Optional[Path] = None


# Base channels (index -> name + unit). Power is derived (encoder_angle * grt_x).
FEATURES: List[Tuple[str, str]] = [
    ("acc_x", "g"),
    ("acc_y", "g"),
    ("acc_z", "g"),
    ("gyro_x", "deg/s"),
    ("gyro_y", "deg/s"),
    ("gyro_z", "deg/s"),
    ("quat_w", ""),
    ("quat_x", ""),
    ("quat_y", ""),
    ("quat_z", ""),
    ("encoder_angle", "deg"),
    ("grf_x", "N"),
    ("grf_y", "N"),
    ("grf_z", "N"),
    ("grt_x", "N·m"),
    ("grt_y", "N·m"),
    ("grt_z", "N·m"),
    ("power", "arb"),  # derived
]

NAME_TO_INDEX: Dict[str, int] = {name: i for i, (name, _) in enumerate(FEATURES)}
POWER_INDEX = NAME_TO_INDEX["power"]


# Line style & color mapping by label keyword
STYLE_BY_LABEL: Dict[str, Tuple[str, Optional[str]]] = {
    "stair-ascent":  ("-.", "tab:red"),
    "stair-descent": ("--", "tab:brown"),
    "ramp-ascent":   ("-.", "tab:green"),
    "ramp-descent":  (":",  "tab:purple"),
    "stop":          ("-",  "tab:gray"),     # 停止はグレーの実線
    # "Level-walk":          ("-",  "tab:"),     # 停止はグレーの実線
}
DEFAULT_LINESTYLE = "-"
DEFAULT_COLOR: Optional[str] = None  # Noneなら呼び出し側の色（例: C0/C1）を使う



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


def parse_values(line: str) -> Optional[List[float]]:
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
    tokens = rest.split()
    cut = len(tokens)
    for i, tok in enumerate(tokens):
        if tok.startswith('"'):
            cut = i
            break
    tokens = tokens[:cut]
    vals: List[float] = []
    for tok in tokens:
        try:
            vals.append(float(tok))
        except ValueError:
            return None
    if len(vals) < 17:
        return None
    return vals[:17]


def _extract_penultimate_quoted(line: str) -> Optional[str]:
    parts: List[str] = []
    i = 0
    while True:
        try:
            s = line.index('"', i)
            e = line.index('"', s + 1)
        except ValueError:
            break
        parts.append(line[s + 1 : e].strip())
        i = e + 1
    if len(parts) >= 2:
        return parts[-2]
    return None


def choose_style_for_label(label: Optional[str]) -> Tuple[str, Optional[str]]:
    if not label:
        return (DEFAULT_LINESTYLE, DEFAULT_COLOR)
    key = label.strip().lower().replace(" ", "-").replace("_", "-")
    return STYLE_BY_LABEL.get(key, (DEFAULT_LINESTYLE, DEFAULT_COLOR))



@dataclass
class Series:
    times_s: List[float]
    values: List[float]
    labels: List[Optional[str]]


def load_series(path: Path, feature_index: int) -> Series:
    t0: Optional[datetime] = None
    times: List[float] = []
    vals: List[float] = []
    lbls: List[Optional[str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ts = parse_timestamp(line)
            if ts is None:
                continue
            v = parse_values(line)
            if v is None:
                continue
            if t0 is None:
                t0 = ts
            times.append((ts - t0).total_seconds())
            if feature_index == POWER_INDEX:
                if len(v) >= 15:
                    vals.append(v[10] * v[14])
                else:
                    vals.append(float("nan"))
            else:
                vals.append(v[feature_index])
            lbls.append(_extract_penultimate_quoted(line))
    return Series(times, vals, lbls)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the same channel from two labeled text files",
    )
    parser.add_argument(
        "file1",
        type=Path,
        nargs="?",
        default=DEFAULT_FILE1,
        help=f"First Labelled_data file (default: {DEFAULT_FILE1})",
    )
    parser.add_argument(
        "file2",
        type=Path,
        nargs="?",
        default=DEFAULT_FILE2,
        help=f"Second Labelled_data file (default: {DEFAULT_FILE2})",
    )

    feat_group = parser.add_mutually_exclusive_group(required=False)
    feat_group.add_argument(
        "--index",
        type=int,
        default=None,
        help=f"Channel index to plot (0-{len(FEATURES)-1})",
    )
    feat_group.add_argument(
        "--feature",
        type=str,
        choices=sorted(NAME_TO_INDEX.keys()),
        help=f"Channel name ({', '.join(sorted(NAME_TO_INDEX.keys()))})",
    )
    parser.add_argument(
        "--unit",
        type=str,
        default=DEFAULT_UNIT,
        help="Override y-axis unit",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=DEFAULT_TITLE,
        help="Graph title (optional)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_SAVE,
        help="If set, save PNG to this path",
    )

    args = parser.parse_args()

    # Feature selection
    if args.feature is not None:
        feat_idx = NAME_TO_INDEX[args.feature.lower()]
    elif args.index is not None:
        feat_idx = int(args.index)
    else:
        if isinstance(DEFAULT_FEATURE, str):
            feat_idx = NAME_TO_INDEX[DEFAULT_FEATURE.lower()]
        else:
            feat_idx = int(DEFAULT_FEATURE)
    if not (0 <= feat_idx < len(FEATURES)):
        raise SystemExit(f"Channel index must be in 0-{len(FEATURES)-1}")
    feat_name, feat_unit = FEATURES[feat_idx]
    if args.unit is not None:
        feat_unit = args.unit

    s1 = load_series(args.file1, feat_idx)
    s2 = load_series(args.file2, feat_idx)

    legend1 = args.file1.name
    legend2 = args.file2.name
    color1 = "C0"
    color2 = "C1"

    plt.figure(figsize=(10, 5))

    def _plot_segmented(series: Series, default_color: str, legend_label: str) -> None:
        if not series.times_s:
            return

        # 最初のスタイルを決定
        ls0, col0 = choose_style_for_label(series.labels[0])
        cur_ls = ls0
        cur_col = col0 if col0 is not None else default_color

        start = 0
        first_drawn = False

        def _plot_range(a: int, b: int, ls: str, col: str, with_label: bool) -> None:
            plt.plot(
                series.times_s[a:b],
                series.values[a:b],
                linestyle=ls,
                color=col,
                label=(legend_label if with_label else None),
            )

        # ラベルの変化でスタイル切替
        for i in range(1, len(series.times_s)):
            ls_i, col_i = choose_style_for_label(series.labels[i])
            ls_i = ls_i
            col_i = col_i if col_i is not None else default_color

            if (ls_i != cur_ls) or (col_i != cur_col):
                _plot_range(start, i, cur_ls, cur_col, not first_drawn)
                first_drawn = True
                start = i
                cur_ls, cur_col = ls_i, col_i

        # 最後の区間
        _plot_range(start, len(series.times_s), cur_ls, cur_col, not first_drawn)

    _plot_segmented(s1, color1, legend1)
    _plot_segmented(s2, color2, legend2)


    y_label = feat_name
    if feat_unit:
        y_label = f"{feat_name} [{feat_unit}]"
    plt.xlabel("Time [s]")
    plt.ylabel(y_label)
    if args.title:
        plt.title(args.title)
    plt.legend(title="Legend")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if args.save is not None:
        out_path = args.save
        if out_path.suffix.lower() == "":
            out_path = out_path.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

