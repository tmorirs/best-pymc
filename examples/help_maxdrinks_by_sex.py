"""HELP study の「差があるはずなのに p = 0.051」データをベイズ推定で見直す。

    python examples/help_maxdrinks_by_sex.py

米国ボストンの解毒病棟に入院した依存症患者 453 名 (male=346 / female=107) の、
過去 30 日間の 1 日あたり最大飲酒量 (max_drinks_per_day) を性別で比較する。

Welch の t 検定では p = 0.051 と「有意差なし」の境界に落ちるが、これは効果量が
小さいからではなく、右に強く歪んだ分布 (歪度 2.3〜2.5、最大 184 杯) が平均の
標準誤差を膨らませているのが主因。BEST は外れ値にロバストな t 分布で尤度を組むため、
「平均という要約統計量がこの分布ではロバストでない」という問題にそのまま向き合える。
詳細は同ディレクトリの help_maxdrinks_by_sex.md を参照。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 画面のない環境でも動くように

import pandas as pd  # noqa: E402

from best_pymc import analyze_two, plot_data_with_ppc  # noqa: E402

CSV_PATH = Path(__file__).with_name("help_maxdrinks_by_sex.csv")


def load_maxdrinks_by_sex() -> tuple["pd.Series", "pd.Series"]:
    """CSV から (male, female) の 1 日あたり最大飲酒量を返す。"""
    d = pd.read_csv(CSV_PATH)
    male = d.loc[d.sex == "male", "max_drinks_per_day"]
    female = d.loc[d.sex == "female", "max_drinks_per_day"]
    return male, female


def main() -> None:
    male, female = load_maxdrinks_by_sex()

    # ROPE は「1 日あたり 1 杯未満の差は実質的に意味がない」という想定。
    # 実際の研究では、この値を分析前に領域知識から決めて事前登録すること。
    result = analyze_two(
        male,
        female,
        group_names=("male", "female"),
        rope=(-1.0, 1.0),
        random_seed=20260101,
    )

    print(result.report())
    print()
    print(result.summary().to_string())
    print()

    # 「平均差が実質的に意味のある大きさ (5 杯超) である」事後確率を直接問い合わせる
    p = result.posterior_prob("diff_of_means", low=5.0)
    print(f"P(平均の差 > 5 杯 | データ) = {p:.3f}")

    fig = result.plot_all()
    fig.savefig("help_maxdrinks_posterior.png", dpi=120)
    fig2 = plot_data_with_ppc(result)
    fig2.savefig("help_maxdrinks_ppc.png", dpi=120)
    print(
        "\n図を help_maxdrinks_posterior.png / help_maxdrinks_ppc.png に保存しました。"
    )


if __name__ == "__main__":
    main()
