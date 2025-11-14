from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm, rcParams as _rc

# 日本語フォント設定（環境にあるものを自動選択）
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


# ======== ユーザー編集用デフォルト (ここを書き換えるだけでOK) ========
DEFAULT_FILE1: Path = Path("Featured_data/ramp_t1_set-0001+Featured_w1000ms_h100ms.txt")
DEFAULT_FILE2: Path = Path("Featured_data/stair_t2_set-0001+Featured_w1000ms_h100ms.txt")
DEFAULT_COLUMN: Union[str, int] = "grt_x_rms"  # 例: "grf_z_rms" や 5 (列インデックス)
# acc_x_mean,acc_x_std,acc_x_min,acc_x_max,acc_x_range,acc_x_median,acc_x_iqr,acc_x_rms,acc_x_skewness,acc_x_kurtosis,acc_x_zcr,acc_x_abs_integral,
# acc_y_mean,acc_y_std,acc_y_min,acc_y_max,acc_y_range,acc_y_median,acc_y_iqr,acc_y_rms,acc_y_skewness,acc_y_kurtosis,acc_y_zcr,acc_y_abs_integral,
# acc_z_mean,acc_z_std,acc_z_min,acc_z_max,acc_z_range,acc_z_median,acc_z_iqr,acc_z_rms,acc_z_skewness,acc_z_kurtosis,acc_z_zcr,acc_z_abs_integral,
# gyro_x_mean,gyro_x_std,gyro_x_min,gyro_x_max,gyro_x_range,gyro_x_median,gyro_x_iqr,gyro_x_rms,gyro_x_skewness,gyro_x_kurtosis,gyro_x_zcr,gyro_x_abs_integral,
# gyro_y_mean,gyro_y_std,gyro_y_min,gyro_y_max,gyro_y_range,gyro_y_median,gyro_y_iqr,gyro_y_rms,gyro_y_skewness,gyro_y_kurtosis,gyro_y_zcr,gyro_y_abs_integral,
# ,gyro_z_std,gyro_z_min,gyro_z_max,gyro_z_range,gyro_z_median,gyro_z_iqr,gyro_z_rms,gyro_z_skewness,gyro_z_kurtosis,gyro_z_zcr,gyro_z_abs_integral,
# quat_w_mean,quat_w_std,quat_w_min,quat_w_max,quat_w_range,quat_w_median,quat_w_iqr,quat_w_rms,quat_w_skewness,quat_w_kurtosis,quat_w_zcr,quat_w_abs_integral,
# quat_x_mean,quat_x_std,quat_x_min,quat_x_max,quat_x_range,quat_x_median,quat_x_iqr,quat_x_rms,quat_x_skewness,quat_x_kurtosis,quat_x_zcr,quat_x_abs_integral,
# quat_y_mean,quat_y_std,quat_y_min,quat_y_max,quat_y_range,quat_y_median,quat_y_iqr,quat_y_rms,quat_y_skewness,quat_y_kurtosis,quat_y_zcr,quat_y_abs_integral,
# quat_z_mean,quat_z_std,quat_z_min,quat_z_max,quat_z_range,quat_z_median,quat_z_iqr,quat_z_rms,quat_z_skewness,quat_z_kurtosis,quat_z_zcr,quat_z_abs_integral,
# encoder_angle_mean,encoder_angle_std,encoder_angle_min,encoder_angle_max,encoder_angle_range,encoder_angle_median,encoder_angle_iqr,encoder_angle_rms,encoder_angle_skewness,encoder_angle_kurtosis,encoder_angle_zcr,encoder_angle_abs_integral,
# grf_x_mean,grf_x_std,grf_x_min,grf_x_max,grf_x_range,grf_x_median,grf_x_iqr,grf_x_rms,grf_x_skewness,grf_x_kurtosis,grf_x_zcr,grf_x_abs_integral,
# grf_y_mean,grf_y_std,grf_y_min,grf_y_max,grf_y_range,grf_y_median,grf_y_iqr,grf_y_rms,grf_y_skewness,grf_y_kurtosis,grf_y_zcr,grf_y_abs_integral,
# grf_z_mean,grf_z_std,grf_z_min,grf_z_max,grf_z_range,grf_z_median,grf_z_iqr,grf_z_rms,grf_z_skewness,grf_z_kurtosis,grf_z_zcr,grf_z_abs_integral,
# grt_x_mean,grt_x_std,grt_x_min,grt_x_max,grt_x_range,grt_x_median,grt_x_iqr,grt_x_rms,grt_x_skewness,grt_x_kurtosis,grt_x_zcr,grt_x_abs_integral,
# grt_y_mean,grt_y_std,grt_y_min,grt_y_max,grt_y_range,grt_y_median,grt_y_iqr,grt_y_rms,grt_y_skewness,grt_y_kurtosis,grt_y_zcr,grt_y_abs_integral,
# grt_z_mean,grt_z_std,grt_z_min,grt_z_max,grt_z_range,grt_z_median,grt_z_iqr,grt_z_rms,grt_z_skewness,grt_z_kurtosis,grt_z_zcr,grt_z_abs_integral,
DEFAULT_TITLE: Optional[str] = None
DEFAULT_SAVE: Optional[Path] = None
# ===================================================================


@dataclass
class Series:
    t_s: List[float]
    y: List[float]
    label: str


def read_feature_series(path: Path, column: Union[str, int]) -> Series:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise SystemExit(f"Empty file: {path}")

        # 列決定
        if isinstance(column, str):
            try:
                col_idx = header.index(column)
            except ValueError:
                raise SystemExit(
                    f"Column '{column}' not found in {path.name}. Available: {', '.join(header)}"
                )
            col_name = column
        else:
            col_idx = int(column)
            if not (0 <= col_idx < len(header)):
                raise SystemExit(f"Column index out of range 0..{len(header)-1}")
            col_name = header[col_idx]

        # 必須: 先頭2列は t_start_s, t_end_s を想定
        try:
            t_start_idx = header.index("t_start_s")
            t_end_idx = header.index("t_end_s")
        except ValueError:
            raise SystemExit("Header must contain t_start_s and t_end_s columns")

        t_vals: List[float] = []
        y_vals: List[float] = []
        for row in reader:
            if not row:
                continue
            # 数値列以外（ラベル）の混入に備えてトライ
            try:
                t_start = float(row[t_start_idx])
                t_end = float(row[t_end_idx])
                y = float(row[col_idx])
            except Exception:
                # 最終列が文字列ラベルでも動くようにスキップしない（y列が数値化できない場合のみスキップ）
                continue
            t_center = 0.5 * (t_start + t_end)
            t_vals.append(t_center)
            y_vals.append(y)

    # 先頭を0sに正規化（各ファイル独立）
    if t_vals:
        t0 = t_vals[0]
        t_vals = [t - t0 for t in t_vals]

    return Series(t_vals, y_vals, label=path.name)


def main():
    ap = argparse.ArgumentParser(description="Featured_data の2ファイルから同一特徴量を比較描画")
    ap.add_argument("file1", type=Path, nargs="?", default=DEFAULT_FILE1, help="1つ目の特徴量ファイル")
    ap.add_argument("file2", type=Path, nargs="?", default=DEFAULT_FILE2, help="2つ目の特徴量ファイル")
    grp = ap.add_mutually_exclusive_group(required=False)
    grp.add_argument("--column", type=str, help="描画する列名 (例: acc_x_mean)")
    grp.add_argument("--index", type=int, help="描画する列のインデックス(0起算)")
    ap.add_argument("--title", type=str, default=DEFAULT_TITLE, help="グラフタイトル")
    ap.add_argument("--save", type=Path, default=DEFAULT_SAVE, help="PNG保存パス")
    args = ap.parse_args()

    # 列の選択（未指定ならデフォルト）
    if args.column is not None:
        col = args.column
        y_label = args.column
    elif args.index is not None:
        col = int(args.index)
        y_label = f"col[{col}]"
    else:
        col = DEFAULT_COLUMN
        y_label = col if isinstance(col, str) else f"col[{col}]"

    s1 = read_feature_series(args.file1, col)
    s2 = read_feature_series(args.file2, col)

    plt.figure(figsize=(10, 5))
    plt.plot(s1.t_s, s1.y, label=s1.label)
    plt.plot(s2.t_s, s2.y, label=s2.label)
    plt.xlabel("時間 [s]")
    plt.ylabel(y_label)
    if args.title:
        plt.title(args.title)
    plt.legend(title="凡例")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if args.save is not None:
        out = args.save
        if out.suffix == "":
            out = out.with_suffix(".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

