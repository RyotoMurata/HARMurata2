from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def read_all_mis(dir_path: Path, pattern: str, encoding: str = "utf-8") -> pd.DataFrame:
    """dir_path 配下を再帰探索して、pattern に一致する *_misclassified.csv を結合して返す。"""
    files = sorted(dir_path.rglob(pattern))
    if not files:
        raise SystemExit(f"[ERR] No files matched: {dir_path.resolve()}/**/{pattern}")

    dfs: List[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp, encoding=encoding)
        except Exception as e:
            print(f"[WARN] skip (read error): {fp} | {e}")
            continue

        # 必須列チェック
        required = {"t_sec", "src", "true_label", "pred_label"}
        miss = required - set(df.columns)
        if miss:
            print(f"[WARN] skip (missing columns {miss}): {fp}")
            continue

        # 追加メタ情報（subject, fold）をファイル名から抽出
        # 例: CV_kanoga_fold3_misclassified.csv
        m = re.search(r"CV_([a-zA-Z0-9\-]+)_fold(\d+)_misclassified\.csv", fp.name)
        if m:
            df["subject"] = m.group(1)
            df["fold"] = int(m.group(2))
        else:
            df["subject"] = "unknown"
            df["fold"] = -1

        # src から set-XXXX を拾う（無ければ NaN）
        m2 = df["src"].astype(str).str.extract(r"(set-\d+)", expand=False)
        df["set_id"] = m2

        dfs.append(df.assign(_file=str(fp)))

    if not dfs:
        raise SystemExit("[ERR] No valid CSV rows found after filtering.")
    return pd.concat(dfs, ignore_index=True)


def rank_pairs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """総合のラベル遷移ペア、subject別、src別、set別のランキングを返す。"""
    # 総合: true→pred ペア
    overall = (
        df.value_counts(["true_label", "pred_label"])
          .reset_index(name="count")
          .sort_values("count", ascending=False)
          .reset_index(drop=True)
    )
    overall["rank"] = overall.index + 1
    total = int(overall["count"].sum())
    overall["ratio_%"] = (overall["count"] * 100.0 / total).round(2)

    # subject 別
    by_subject = (
        df.value_counts(["subject", "true_label", "pred_label"])
          .reset_index(name="count")
          .sort_values(["subject", "count"], ascending=[True, False])
          .reset_index(drop=True)
    )

    # ファイル（src）別
    by_src = (
        df.value_counts(["src", "true_label", "pred_label"])
          .reset_index(name="count")
          .sort_values(["src", "count"], ascending=[True, False])
          .reset_index(drop=True)
    )

    # set 単位
    by_set = (
        df.value_counts(["set_id", "true_label", "pred_label"])
          .reset_index(name="count")
          .sort_values(["set_id", "count"], ascending=[True, False])
          .reset_index(drop=True)
    )

    return overall, by_subject, by_src, by_set


def main():
    ap = argparse.ArgumentParser(
        description="誤分類ログ(*_misclassified.csv)から true→pred ペアの誤認識件数と順位を集計"
    )
    ap.add_argument(
        "dir",
        nargs="?",                # 省略可にする
        type=Path,
        default=Path.cwd(),       # 既定はカレントディレクトリ
        help="誤分類CSVを再帰探索する起点フォルダ（省略時はカレントディレクトリ）"
    )
    ap.add_argument(
        "--glob", type=str, default="*misclassified.csv",
        help="探索パターン（Path.rglob に渡すパターン）"
    )
    ap.add_argument(
        "--out", type=Path, default=Path("misrank_out"),
        help="結果CSVの出力先ディレクトリ"
    )
    ap.add_argument(
        "--top", type=int, default=20,
        help="コンソール表示や *top.csv に含める上位ペア数"
    )
    ap.add_argument(
        "--encoding", type=str, default="utf-8",
        help="入力CSVの文字コード（既定: utf-8）"
    )
    args = ap.parse_args()

    # 簡易バリデーション
    if not args.dir.exists() or not args.dir.is_dir():
        raise SystemExit(f"[ERR] 指定フォルダが存在しないか、ディレクトリではありません: {args.dir.resolve()}")
    print(f"[INFO] 探索起点: {args.dir.resolve()} | パターン: {args.glob} | encoding: {args.encoding}")

    # 読み込み & 集計
    df = read_all_mis(args.dir, args.glob, encoding=args.encoding)
    overall, by_subject, by_src, by_set = rank_pairs(df)

    # 保存
    args.out.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.out / "overall_pair_rank.csv", index=False)
    by_subject.to_csv(args.out / "by_subject_pair_rank.csv", index=False)
    by_src.to_csv(args.out / "by_src_pair_rank.csv", index=False)
    by_set.to_csv(args.out / "by_set_pair_rank.csv", index=False)

    # TOP だけのCSVも保存（レポート用）
    overall.head(args.top).to_csv(args.out / "overall_pair_rank_top.csv", index=False)
    pd.concat(
        [g.head(min(args.top, 10)) for _, g in by_subject.groupby("subject", sort=False)],
        ignore_index=True
    ).to_csv(args.out / "by_subject_pair_rank_top.csv", index=False)
    pd.concat(
        [g.head(min(args.top, 10)) for _, g in by_src.groupby("src", sort=False)],
        ignore_index=True
    ).to_csv(args.out / "by_src_pair_rank_top.csv", index=False)
    pd.concat(
        [g.head(min(args.top, 10)) for _, g in by_set.groupby("set_id", sort=False)],
        ignore_index=True
    ).to_csv(args.out / "by_set_pair_rank_top.csv", index=False)

    # 画面表示（上位のみ）
    print("\n=== Overall misclassification pair ranking (true_label -> pred_label) ===")
    print(overall.head(args.top).to_string(index=False))

    print("\n=== Top pairs per subject (first few rows) ===")
    for subj in by_subject["subject"].unique():
        subdf = by_subject[by_subject["subject"] == subj].head(min(args.top, 10))
        print(f"\n[subject={subj}]")
        print(subdf.to_string(index=False))

    print("\n=== Top pairs per src (first few rows) ===")
    for s in by_src["src"].unique()[:5]:  # 表示は多くなりがちなので上位5 srcだけ抜粋
        subdf = by_src[by_src["src"] == s].head(min(args.top, 10))
        print(f"\n[src={s}]")
        print(subdf.to_string(index=False))

    print(f"\nSaved CSVs to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
