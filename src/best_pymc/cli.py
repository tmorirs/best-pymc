"""コマンドラインから BEST を実行するためのエントリポイント。

インストール後:

    best-demo                       # 同梱の Kruschke 例を実行
    best-demo --csv data.csv --group-column arm --value-column score

uv なら:

    uv run best-demo
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _load_csv(path: str, group_col: str, value_col: str):
    import pandas as pd

    df = pd.read_csv(path)
    for col in (group_col, value_col):
        if col not in df.columns:
            raise SystemExit(
                f"列 '{col}' が {path} にありません。実際の列: {list(df.columns)}"
            )
    levels = df[group_col].dropna().unique().tolist()
    if len(levels) != 2:
        raise SystemExit(
            f"群の水準がちょうど2つ必要ですが {len(levels)} 個ありました: {levels}"
        )
    g1, g2 = levels
    y1 = df.loc[df[group_col] == g1, value_col].astype(float).to_numpy()
    y2 = df.loc[df[group_col] == g2, value_col].astype(float).to_numpy()
    return y1, y2, (str(g1), str(g2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="best-demo",
        description="BEST (ベイズ推定による2群比較) を実行する",
    )
    p.add_argument("--csv", help="長形式 CSV のパス（省略時は同梱の例を使う）")
    p.add_argument("--group-column", default="group", help="群を表す列名")
    p.add_argument("--value-column", default="value", help="測定値の列名")
    p.add_argument(
        "--rope",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="ROPE（実質的同等領域）。データを見る前に決めること",
    )
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--tune", type=int, default=2000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260101)
    p.add_argument("--hdi-prob", type=float, default=0.95)
    p.add_argument("--plot", metavar="PATH", help="事後分布の図をこのパスに保存する")
    p.add_argument(
        "--overlaid",
        action="store_true",
        help="平均・標準偏差などを群ごとに別パネルへ並べず、色分けして同一パネルに重ねる",
    )
    p.add_argument(
        "--sensitivity",
        action="store_true",
        help="事前分布の感度分析もあわせて実行する",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.csv:
        y1, y2, names = _load_csv(args.csv, args.group_column, args.value_column)
    else:
        from .datasets import smart_drug

        y1, y2 = smart_drug()
        names = ("drug", "placebo")
        print("[同梱の Kruschke (2013) スマートドラッグ例を使用]\n")

    from .core import analyze_two

    rope = tuple(args.rope) if args.rope else None
    result = analyze_two(
        y1,
        y2,
        group_names=names,
        rope=rope,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        random_seed=args.seed,
    )
    print(result.report(prob=args.hdi_prob))
    print()
    print(result.summary(prob=args.hdi_prob).to_string())

    if args.sensitivity:
        from .sensitivity import sensitivity_analysis

        print("\n--- 事前分布の感度分析 ---")
        print(
            sensitivity_analysis(
                y1, y2, rope=rope, prob=args.hdi_prob,
                draws=args.draws, tune=args.tune,
                chains=args.chains, random_seed=args.seed,
            ).to_string()
        )

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        fig = result.plot_all(prob=args.hdi_prob, overlaid=args.overlaid)
        fig.savefig(args.plot, dpi=120)
        print(f"\n図を {args.plot} に保存しました。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
