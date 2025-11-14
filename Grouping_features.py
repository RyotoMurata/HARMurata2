# Group_Features_From_Corr_inline.py
from __future__ import annotations
import argparse, glob
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

# ===== 手動指定ゾーン =====
# CSV/フォルダ/ワイルドカードの混在OK。空リストならCLI --corr を使います。
INLINE_CSVS: List[str] = [
    # 例）
    "feature_corr/kaneishi/set-0001/ramp/kaneishi_set-0001_ramp_ALL_corr_pearson.csv",
    # "feature_corr/kaneishi/**/*.csv",          # ワイルドカード
    # "feature_corr/kanoga/set-0002",            # フォルダ（配下の *_corr_*.csv を全部）
]
INLINE_METHOD: str = "both"   # "hierarchical" | "graph" | "both"
INLINE_DIST_THR: float = 0.3  # 階層法: 距離=1-|corr| のしきい値
INLINE_N_CLUSTERS: int = 0    # >0ならクラスタ数固定
INLINE_TAU: float = 0.90      # グラフ法: |corr| のしきい値
INLINE_OUT_DIR: str = "feature_groups"
# ================================================

def iter_corr_csvs_from_args_or_inline(corr_arg: Optional[Path]) -> Iterable[Path]:
    if INLINE_CSVS:  # 手動指定を優先
        seen: set[Path] = set()
        for pat in INLINE_CSVS:
            p = Path(pat)
            if p.suffix.lower() == ".csv" and p.exists():
                seen.add(p.resolve())
            elif any(ch in pat for ch in "*?[]"):
                for q in glob.glob(pat, recursive=True):
                    qpath = Path(q)
                    if qpath.suffix.lower() == ".csv" and "_corr_" in qpath.name:
                        seen.add(qpath.resolve())
            elif p.is_dir():
                for q in p.rglob("*.csv"):
                    if "_corr_" in q.name:
                        seen.add(q.resolve())
            else:
                # 明示CSVだが存在しない or フォルダでもワイルドカードでもない → 無視
                pass
        for x in sorted(seen):
            yield x
        return

    # ここから従来のCLI指定
    if corr_arg is None:
        return
    if corr_arg.is_file():
        yield corr_arg
    else:
        for p in corr_arg.rglob("*.csv"):
            if "_corr_" in p.name:
                yield p

def load_corr(corr_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(corr_csv, index_col=0)
    df = df.loc[df.index, df.index]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def corr_to_distance(corr_df: pd.DataFrame, use_abs: bool = True) -> pd.DataFrame:
    C = corr_df.values.astype(float)
    if use_abs:
        C = np.abs(C)
    C = np.clip(C, 0.0, 1.0)          # 相関の絶対値を [0,1] に
    D = 1.0 - C

    # --- ここから堅牢化 ---
    D = (D + D.T) / 2                 # 対称化
    np.fill_diagonal(D, 0.0)          # 対角は 0
    D = np.nan_to_num(D, nan=1.0)     # NaN があれば最大距離で埋める
    D[D < 0] = 0.0                    # ★ 微小負を 0 にクリップ（重要）
    # 数値誤差で 2.0 を越える可能性はほぼ無いが念のため
    D[D > 2] = 2.0
    # --- ここまで ---

    return pd.DataFrame(D, index=corr_df.index, columns=corr_df.columns)


def choose_representatives(dist_df: pd.DataFrame, clusters: Dict[int, List[str]]) -> Dict[int, str]:
    reps: Dict[int, str] = {}
    for cid, feats in clusters.items():
        if len(feats) == 1:
            reps[cid] = feats[0]
        else:
            sub = dist_df.loc[feats, feats].values
            reps[cid] = feats[int(np.argmin(sub.sum(axis=1)))]
    return reps

def hierarchical_group(dist_df: pd.DataFrame, dist_thr: float | None, n_clusters: int | None
                      ) -> Tuple[np.ndarray, Dict[int, List[str]], Dict[int, str]]:
    condensed = squareform(dist_df.values, checks=False)
    Z = linkage(condensed, method="average")
    if n_clusters and n_clusters > 0:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    else:
        if dist_thr is None:
            dist_thr = 0.2
        labels = fcluster(Z, t=dist_thr, criterion="distance")

    names = dist_df.index.tolist()
    clusters: Dict[int, List[str]] = {}
    for name, lab in zip(names, labels):
        clusters.setdefault(int(lab), []).append(name)

    reps = choose_representatives(dist_df, clusters)
    return Z, clusters, reps

def save_dendrogram(out_dir: Path, prefix: str, Z: np.ndarray, labels: List[str]) -> None:
    ensure_dir(out_dir)
    plt.figure(figsize=(max(8, 0.12 * len(labels)), 6))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title("Hierarchical clustering (distance = 1 - |corr|)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_dendrogram.png", dpi=200)
    plt.close()

def graph_threshold_group(corr_df: pd.DataFrame, tau: float = 0.95, use_abs: bool = True
                         ) -> Tuple[Dict[int, List[str]], Dict[int, str]]:
    names = corr_df.index.tolist()
    C = corr_df.values.astype(float)
    if use_abs:
        C = np.abs(C)
    n = len(names)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] >= tau:
                adj[i].add(j); adj[j].add(i)
    seen = set(); clusters: Dict[int, List[str]] = {}; cid = 0
    for i in range(n):
        if i in seen: continue
        cid += 1
        stack = [i]; comp = []
        while stack:
            v = stack.pop()
            if v in seen: continue
            seen.add(v); comp.append(names[v])
            stack.extend([u for u in adj[v] if u not in seen])
        clusters[cid] = sorted(comp)
    dist_df = pd.DataFrame(1.0 - C, index=names, columns=names)
    reps = choose_representatives(dist_df, clusters)
    return clusters, reps

def save_clusters(out_dir: Path, prefix: str, clusters: Dict[int, List[str]], reps: Dict[int, str]) -> None:
    ensure_dir(out_dir)
    rows = []
    for cid, feats in sorted(clusters.items()):
        rep = reps.get(cid, "")
        for f in feats:
            rows.append({"cluster_id": cid, "representative": rep, "feature": f})
    pd.DataFrame(rows).to_csv(out_dir / f"{prefix}_clusters.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        [{"cluster_id": cid, "representative": reps[cid], "size": len(clusters[cid])}
         for cid in sorted(clusters)]
    ).to_csv(out_dir / f"{prefix}_representatives.csv", index=False, encoding="utf-8")

def process_one(corr_csv: Path, out_root: Path, do_hier: bool, do_graph: bool,
                dist_thr: float | None, n_clusters: int | None, tau: float) -> None:
    corr = load_corr(corr_csv)
    name = corr_csv.stem

    # デバッグ用
    C = corr.values.astype(float)
    mx = np.nanmax(C); mn = np.nanmin(C)
    asym = np.nanmax(np.abs(C - C.T))
    diag_min = np.nanmin(np.diag(C)); diag_max = np.nanmax(np.diag(C))
    over = np.sum(np.abs(C) > 1.0 + 1e-12)

    print(f"[CHECK] min={mn:.12f}, max={mx:.12f}, asym={asym:.3e}, diag=[{diag_min:.12f},{diag_max:.12f}], over1={over}")

    # 入力CSVの2階層上から相対を再現（適宜調整OK）
    try:
        out_dir = Path(out_root) / corr_csv.parent.relative_to(corr_csv.parents[1])
    except Exception:
        out_dir = Path(out_root) / corr_csv.parent.name
    if do_hier:
        dist = corr_to_distance(corr, use_abs=True)
        Z, clusters, reps = hierarchical_group(dist, dist_thr, n_clusters)
        save_clusters(out_dir, f"{name}_hier", clusters, reps)
        save_dendrogram(out_dir, f"{name}_hier", Z, labels=dist.index.tolist())
    if do_graph:
        clusters, reps = graph_threshold_group(corr, tau=tau, use_abs=True)
        save_clusters(out_dir, f"{name}_graph_tau{tau:g}", clusters, reps)

def main() -> None:
    ap = argparse.ArgumentParser(description="相関CSVを読み、階層法/グラフ法でグルーピング（スクリプト内手動指定に対応）")
    ap.add_argument("--corr", type=Path, default=None, help="（任意）CSVまたはフォルダ。INLINE_CSVS が空の時だけ使用")
    ap.add_argument("--method", type=str, default=INLINE_METHOD, choices=["hierarchical", "graph", "both"])
    ap.add_argument("--dist-thr", type=float, default=INLINE_DIST_THR)
    ap.add_argument("--n-clusters", type=int, default=INLINE_N_CLUSTERS)
    ap.add_argument("--tau", type=float, default=INLINE_TAU)
    ap.add_argument("--out-dir", type=Path, default=Path(INLINE_OUT_DIR))
    args = ap.parse_args()

    do_hier = args.method in ("hierarchical", "both")
    do_graph = args.method in ("graph", "both")

    targets = list(iter_corr_csvs_from_args_or_inline(args.corr))
    if not targets:
        raise SystemExit("処理対象のCSVが見つかりませんでした。INLINE_CSVS か --corr を確認してください。")

    for csv_path in targets:
        try:
            process_one(
                corr_csv=csv_path,
                out_root=args.out_dir,
                do_hier=do_hier,
                do_graph=do_graph,
                dist_thr=(None if args.n_clusters and args.n_clusters > 0 else args.dist_thr),
                n_clusters=(args.n_clusters if args.n_clusters and args.n_clusters > 0 else None),
                tau=args.tau,
            )
            print(f"[OK] {csv_path}")
        except Exception as e:
            print(f"[ERR] {csv_path}: {e}")

if __name__ == "__main__":
    main()
