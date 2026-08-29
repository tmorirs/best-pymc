"""Kruschke (2013) の「スマートドラッグ」例をそのまま再現する。

    python examples/smart_drug.py
    python examples/smart_drug.py --overlaid   # 平均・標準偏差を群ごとに重ねて描く
    # または、インストール済みなら:  uv run best-demo --rope -1 1

t 検定とベイズ推定で結論がどう食い違うかがそのまま見られる例になっている。
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # 画面のない環境でも動くように

from best_pymc import analyze_two, plot_data_with_ppc, smart_drug  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--overlaid",
        action="store_true",
        help="平均・標準偏差を群ごとに別パネルへ並べず、色分けして同一パネルに重ねる。",
    )
    args = p.parse_args(argv)

    iq_drug, iq_placebo = smart_drug()

    # ROPE は「IQ で 1 点未満の差は実質的に意味がない」という想定。
    # 実際の研究では、この値を分析前に領域知識から決めて事前登録すること。
    result = analyze_two(
        iq_drug,
        iq_placebo,
        group_names=("drug", "placebo"),
        rope=(-1.0, 1.0),
        random_seed=20260101,
    )

    print(result.report())
    print()
    print(result.summary().to_string())
    print()

    # 任意の区間の事後確率を直接問い合わせられる
    p = result.posterior_prob("diff_of_means", low=0.5)
    print(f"P(平均の差 > 0.5 | データ) = {p:.3f}")

    fig = result.plot_all(overlaid=args.overlaid)
    fig.savefig("smart_drug_posterior.png", dpi=120)
    fig2 = plot_data_with_ppc(result)
    fig2.savefig("smart_drug_ppc.png", dpi=120)
    print("\n図を smart_drug_posterior.png / smart_drug_ppc.png に保存しました。")


if __name__ == "__main__":
    main()
