
"""
ディレクトリ内からファイルを探し、1列目(タイムスタンプ)と8~11列目(クオータニオン)を読み取り、
剛体姿勢(機体座標の3軸)を Matplotlib でアニメ表示するスクリプト。

【使い方の例】
- 設定ブロック(CONFIG)だけで実行（おすすめ）
    python visualize_attitude.py
- 設定ブロックを初期値にしつつ CLI で一部上書き
    python visualize_attitude.py --dir /mnt/data --pattern "ramp_*.txt" --rate 1.2
- アップロード済みの単一ファイルを直接
    python visualize_attitude.py --file /mnt/data/ramp_t1_kaneishi_set-0001.txt
"""

import argparse
import glob
import os
import re
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime

# =========================
# 設定ブロック（ここを編集）
# =========================
CONFIG = {
    "DIR": r"C:\Users\ryout\OneDrive\ドキュメント\デスクトップ\Keio_lab\修士研究\Murata_HAR\Labelled_data",       # 探索ディレクトリ
    "PATTERN": "ramp_t1_kaneishi_*.txt",          # グロブパターン（例: "ramp_*.txt"）
    "FILE": None,                # 直接パスを指定する場合（指定があれば DIR/PATTERN は無視）
    "QORDER": "wxyz",            # ファイル中のクオータニオン順序: "wxyz" or "xyzw"
    "DOWNSAMPLE":3 ,             # 行の間引き係数（>=1）
    "RATE": 1.0,                 # 再生レート倍率
    "INTERACTIVE_PICK": True,    # FILE未指定時、候補から番号選択する
}

TS_LINE_RE = re.compile(r'^\[(?P<ts>[^]]+)\]\s+(?P<rest>.+)$')

def find_files(directory: str, pattern: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    return [p for p in paths if os.path.isfile(p)]

def choose_file_interactively(files: List[str]) -> Optional[str]:
    """候補を番号付きで表示し、選ばれたファイルパスを返す。キャンセル時は None。"""
    if not files:
        return None
    print("\n[interactive] 候補ファイル:")
    for i, p in enumerate(files):
        print(f"  [{i}] {p}")
    while True:
        s = input("番号で選択してください（Enterでキャンセル）: ").strip()
        if s == "":
            return None
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < len(files):
                return files[idx]
        print("無効な入力です。")

def parse_file(path: str, qorder: str = 'wxyz', downsample: int = 1
               ) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Returns:
      timestamps: list[str]           表示用の時刻文字列 "[YYYY-mm-dd ...]"
      t:          np.ndarray shape(N,) 先頭を0とした相対秒（float）
      quats:      np.ndarray shape(N,4) 正規化済み [w,x,y,z]
    """
    ts_list: List[str] = []
    t_abs:   List[datetime] = []
    q_list:  List[List[float]] = []

    def to_wxyz(qraw: List[float]) -> List[float]:
        if qorder.lower() == 'wxyz':
            w, x, y, z = qraw
        elif qorder.lower() == 'xyzw':
            x, y, z, w = qraw
        else:
            raise ValueError(f"Unsupported qorder: {qorder} (use 'wxyz' or 'xyzw')")
        return [w, x, y, z]

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if downsample > 1 and (i % downsample != 0):
                continue
            line = line.strip()
            if not line:
                continue
            m = TS_LINE_RE.match(line)
            if not m:
                continue

            # ① 文字列のタイムスタンプ（表示用）
            ts = m.group('ts')  # "2024-12-13 14:47:25.394"
            # ② datetimeにパース（実時間計算用）
            try:
                dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                # ミリ秒が無い行にも対応
                dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')

            toks = m.group('rest').split()
            if len(toks) < 11:
                continue
            try:
                qraw = [float(toks[7]), float(toks[8]), float(toks[9]), float(toks[10])]
            except ValueError:
                continue

            wxyz = to_wxyz(qraw)
            q = np.asarray(wxyz, dtype=float)
            n = np.linalg.norm(q)
            if n == 0 or not np.isfinite(n):
                continue
            q /= n

            ts_list.append(ts)
            t_abs.append(dt)
            q_list.append(q.tolist())

    if not q_list:
        raise RuntimeError(f"No valid quaternion rows found in: {path}")

    # 相対秒（先頭を 0.0）
    t0 = t_abs[0]
    t_rel = np.array([(ti - t0).total_seconds() for ti in t_abs], dtype=float)
    # ===== ここから追加：符号連続性の強制 =====
    quats = np.asarray(q_list, dtype=float)  # shape (N,4) [w,x,y,z]
    for i in range(1, len(quats)):
        if np.dot(quats[i-1], quats[i]) < 0.0:
            quats[i] = -quats[i]
    # ===== ここまで追加 =====
    return ts_list, t_rel, np.asarray(q_list)

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def euler_zyx_from_R(R: np.ndarray) -> tuple[float, float, float]:
    """
    ZYX (yaw, pitch, roll) を返す [rad]。右手系、R = Rz(yaw) @ Ry(pitch) @ Rx(roll) 前提。
    """
    # pitch の特異点に注意（|R[2,0]| ≈ 1）
    sy = -R[2,0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)

    # cos(pitch) が小さいときに数値不安定になるので分岐
    if abs(np.cos(pitch)) > 1e-8:
        roll  = np.arctan2(R[2,1], R[2,2])     # X回り
        yaw   = np.arctan2(R[1,0], R[0,0])     # Z回り
    else:
        # gimbal lock: R[2,1] ≈ 0, R[2,2] ≈ 0
        roll  = np.arctan2(-R[1,2], R[1,1])
        yaw   = 0.0
    return yaw, pitch, roll


def make_rod(length=0.5, width=0.05, height=0.05):
    """
    原点から +X に length だけ伸びる直方体(棒)の 8頂点 (8,3) を返す。
    局所座標系：X前(+)、Y左(+)、Z上(+)
    """
    x0, x1 = 0.0, length
    yh, zh = width/2.0, height/2.0
    V = np.array([
        [x0, -yh, -zh],
        [x1, -yh, -zh],
        [x1,  yh, -zh],
        [x0,  yh, -zh],
        [x0, -yh,  zh],
        [x1, -yh,  zh],
        [x1,  yh,  zh],
        [x0,  yh,  zh],
    ], dtype=float)
    return V

def box_faces(verts):
    idx = [
        [0,1,2,3],  # bottom
        [4,5,6,7],  # top
        [0,1,5,4],  # side
        [2,3,7,6],  # side
        [1,2,6,5],  # side
        [0,3,7,4],  # side
    ]
    return [[verts[i] for i in f] for f in idx]


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """クオータニオン [w,x,y,z] -> 回転行列(3x3)"""
    w, x, y, z = q
    xx, yy, zz = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)]
    ], dtype=float)
    return R

# +X(世界) を 画面の「下」に向けて描くための整列回転（右手系・det=+1）
# WベクトルをP(表示)へ:  v_plot = R_align @ v_world
R_align = np.array([
    [ 0,  1,  0],   # Y_world → X_plot
    [ 0,  0, -1],   # -Z_world → Y_plot
    [-1,  0,  0],   # -X_world → Z_plot（= +Xが下に見える）
], dtype=float)


def animate_attitude(timestamps: List[str], t: np.ndarray, quats: np.ndarray, rate: float = 1.0):
    # 表示スケール
    L = 0.5  # 棒の長さ [m]
    rod_base = make_rod(length=L, width=0.05, height=0.05)   # 局所ボディ座標

    # 参考用：ボディ局所軸（原点から出る短い矢印）
    axis_len = 0.25
    basis = np.eye(3) * axis_len  # 列: ex,ey,ez

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Rigid Body (0.5 m rod along +X)  —  +X is downward (right-handed)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    lim = L * 0.7
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
    ax.view_init(elev=20, azim=45)

    # 棒メッシュ
    faces0 = box_faces(rod_base)
    poly = Poly3DCollection(faces0, alpha=0.5, edgecolor='k')
    ax.add_collection3d(poly)

    # 軸ライン（参照）
    (line_x,) = ax.plot([0, basis[0,0]], [0, basis[1,0]], [0, basis[2,0]], lw=2, label='Body X')
    (line_y,) = ax.plot([0, basis[0,1]], [0, basis[1,1]], [0, basis[2,1]], lw=2, label='Body Y')
    (line_z,) = ax.plot([0, basis[0,2]], [0, basis[1,2]], [0, basis[2,2]], lw=2, label='Body Z')
    # 先端ガイド（原点→棒の先端）
    tip_line, = ax.plot([0, L], [0, 0], [0, 0], lw=3, linestyle='--', label='Rod tip')

    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    angle_text = ax.text2D(0.02, 0.90, "", transform=ax.transAxes)  # ★追加：角度表示
    ax.legend(loc="upper right")

    
    # 実時間×倍率の dt
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = np.median(dt[1:])

    state = {"anim": None}

    def apply_R(P: np.ndarray, R: np.ndarray) -> np.ndarray:
        return (R @ P.T).T

    def update(i: int):
        # クオータニオン回転（世界座標）
        Rb = quat_to_rotmat(quats[i])      # body→world
        # 棒と軸を世界→描画へ整列（+X を下へ）
        rod_w = apply_R(rod_base, Rb)      # body→world
        rod_p = apply_R(rod_w, R_align)    # world→plot

        poly.set_verts(box_faces(rod_p))

        axes_w = apply_R(basis, Rb)
        axes_p = apply_R(axes_w, R_align)

        # 軸ライン更新
        x_end, y_end, z_end = axes_p[0], axes_p[1], axes_p[2]
        line_x.set_data([0, x_end[0]], [0, x_end[1]]); line_x.set_3d_properties([0, x_end[2]])
        line_y.set_data([0, y_end[0]], [0, y_end[1]]); line_y.set_3d_properties([0, y_end[2]])
        line_z.set_data([0, z_end[0]], [0, z_end[1]]); line_z.set_3d_properties([0, z_end[2]])

        # 先端（+X端）の線（原点→先端）
        tip_local = np.array([L, 0, 0])
        tip_world = Rb @ tip_local
        tip_plot  = R_align @ tip_world
        tip_line.set_data([0, tip_plot[0]], [0, tip_plot[1]])
        tip_line.set_3d_properties([0, tip_plot[2]])

        time_text.set_text(f"time: {timestamps[i]}")
        # ★追加：オイラー角（Rbから算出。表示は度）
        yaw, pitch, roll = euler_zyx_from_R(Rb)
        yaw_d, pitch_d, roll_d = np.degrees([yaw, pitch, roll])
        angle_text.set_text(f"yaw(Z): {yaw_d:+.1f}°  pitch(Y): {pitch_d:+.1f}°  roll(X): {roll_d:+.1f}°")
        # 実時間×倍率の可変インターバル
        next_ms = max(int(1000.0 * dt[i] / max(rate, 1e-6)), 10)
        if state["anim"] is not None:
            state["anim"].event_source.interval = next_ms

        return poly, line_x, line_y, line_z, tip_line, time_text
    

    init_ms = max(int(1000.0 * np.median(dt) / max(rate, 1e-6)), 10)
    anim = FuncAnimation(fig, update, frames=len(quats), interval=init_ms, blit=False, repeat=True)
    state["anim"] = anim

    fig._anim = anim
    plt.tight_layout()
    plt.show()
    return anim


def main():
    # CLI のデフォルト値に設定ブロックを反映
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",      type=str, default=CONFIG["DIR"],      help="探索するディレクトリ")
    ap.add_argument("--pattern",  type=str, default=CONFIG["PATTERN"],  help='globパターン (例: "ramp_*.txt")')
    ap.add_argument("--file",     type=str, default=CONFIG["FILE"],     help="直接ファイルパスを指定（指定時はdir/pattern無視）")
    ap.add_argument("--qorder",   type=str, default=CONFIG["QORDER"],   choices=["wxyz", "xyzw"], help="ファイル内クオータニオンの順序")
    ap.add_argument("--downsample", type=int, default=CONFIG["DOWNSAMPLE"], help="行の間引き係数（>=1）")
    ap.add_argument("--rate",     type=float, default=CONFIG["RATE"],   help="実時間に対する再生倍率（2.0で2倍速）")
    ap.add_argument("--no-interactive", action="store_true", help="インタラクティブ選択を無効化")
    args = ap.parse_args()

    # 優先順位: CLIの --file > CONFIG["FILE"]
    target = args.file or CONFIG["FILE"]
    if not target:
        # ディレクトリ探索
        files = find_files(args.dir, args.pattern)
        if not files:
            raise FileNotFoundError(f"No files matched in {args.dir!r} with pattern {args.pattern!r}")
        # 必要ならインタラクティブ選択
        if CONFIG.get("INTERACTIVE_PICK", False) and not args.no_interactive:
            picked = choose_file_interactively(files)
            target = picked or files[0]
            print(f"[info] selected: {target}" if picked else f"[info] canceled; using first file: {target}")
        else:
            target = files[0]
            print(f"[info] Using file: {target}")
    else:
        if not os.path.isfile(target):
            raise FileNotFoundError(f"file not found: {target}")

    # 解析 → 実時間×倍率でアニメ
    timestamps, t, quats = parse_file(
        target,
        qorder=args.qorder,
        downsample=max(1, args.downsample)
    )
    print(f"[info] loaded {len(quats)} frames from {target}")

    anim = animate_attitude(timestamps, t, quats, rate=args.rate)
    globals()["_KEEP_ANIM"] = anim  # 参照保持（警告回避の保険）

if __name__ == "__main__":
    main()

