"""HELP study の「差があるはずなのに t 検定ではぎりぎり非有意」データをベイズ推定で見直す。

    # 既定 (性別 × 1日最大飲酒量) — 元からの主対象
    python examples/help_maxdrinks_by_sex.py

    # 用意した別の境界例を選ぶ
    python examples/help_maxdrinks_by_sex.py --case homeless-cesd

    # 一覧を見る / 列を自由に組み合わせる
    python examples/help_maxdrinks_by_sex.py --list-cases
    python examples/help_maxdrinks_by_sex.py --group substance --outcome cesd \
        --levels alcohol heroin

米国ボストンの解毒病棟に入院した依存症患者 453 名 (male=346 / female=107) の
ベースライン面接データ (R パッケージ mosaicData の HELPrct)。CSV には主対象の
max_drinks_per_day (性別比較) 以外にも avg_drinks_per_day / age / cesd などの
量的変数と、sex / substance / homeless の群分け列が入っている。

これらの組み合わせのいくつかは、主対象と同じ「t 検定ではぎりぎり非有意 (p が
0.05 の少し上) だが、順位ベースの検定では明確に有意」という構図になる。効果量が
小さいのではなく、右に強く歪んだ分布が平均の標準誤差を膨らませているのが主因で、
「平均という要約統計量がこの分布ではロバストでない」問題にそのまま向き合える。
BEST は外れ値にロバストな t 分布で尤度を組むため、この見直しに向いている。

report() には比較用の Welch の t 検定・Cohen's d・検定力も併記されるので、
出力の中で「頻度論の t 検定ではどう見えるか」も同時に確認できる。
詳細は同ディレクトリの help_maxdrinks_by_sex.md を参照。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 画面のない環境でも動くように

import pandas as pd  # noqa: E402

from best_pymc import analyze_two, plot_data_with_ppc  # noqa: E402

CSV_PATH = Path(__file__).with_name("help_maxdrinks_by_sex.csv")

# CSV に含まれる量的変数 (比較する outcome の候補) と群分け列。
NUMERIC_COLUMNS = ("max_drinks_per_day", "avg_drinks_per_day", "age", "cesd")
GROUP_COLUMNS = ("sex", "substance", "homeless")

# 事前登録した想定仮説つきの比較例。
#   levels : (仮説上「大きい」と予想する群, 比較対照) の順。差はこの向きで計算する。
#   rope   : 実質的同等領域。分析前に領域知識から決め、事前登録するのが本来の運用。
#   threshold : P(平均差 > threshold | データ) を問い合わせる「実質的に意味のある差」。
# welch_p は手元計算の目安 (Welch の t 検定)。「/」の右は Mann-Whitney U の p。
CASES: dict[str, dict] = {
    "sex-maxdrinks": {
        "group": "sex",
        "levels": ("male", "female"),
        "outcome": "max_drinks_per_day",
        "rope": (-1.0, 1.0),
        "threshold": 5.0,
        "prefix": "help_maxdrinks",
        "hypothesis": "男性のほうが1日最大飲酒量が多い",
        "note": "主対象。Welch p=0.051 / MWU p=0.004。95%CI 下限が -0.025 と 0 をかすめる。",
    },
    "sex-avgdrinks": {
        "group": "sex",
        "levels": ("male", "female"),
        "outcome": "avg_drinks_per_day",
        "rope": (-1.0, 1.0),
        "threshold": 3.0,
        "prefix": "help_avgdrinks_by_sex",
        "hypothesis": "男性のほうが1日平均飲酒量が多い",
        "note": "主対象と同じ構図。Welch p=0.067 / MWU p=0.007。t 検定だけ非有意側に落ちる。",
    },
    "substance-avgdrinks": {
        "group": "substance",
        "levels": ("cocaine", "heroin"),
        "outcome": "avg_drinks_per_day",
        "rope": (-1.0, 1.0),
        "threshold": 3.0,
        "prefix": "help_avgdrinks_by_substance",
        "hypothesis": "コカイン群のほうがヘロイン群より1日平均飲酒量が多い",
        "note": "Welch p=0.054 / MWU p=0.003。t 検定はぎりぎり非有意、順位検定は明確に有意。",
    },
    "homeless-cesd": {
        "group": "homeless",
        "levels": ("homeless", "housed"),
        "outcome": "cesd",
        "rope": (-2.0, 2.0),
        "threshold": 5.0,
        "prefix": "help_cesd_by_homeless",
        "hypothesis": "ホームレス群のほうが抑うつ得点 (CES-D) が高い",
        "note": "Welch p=0.064 / MWU p=0.026。抑うつ得点でも同じ境界例が起きる。",
    },
    "sex-cesd": {
        "group": "sex",
        "levels": ("female", "male"),
        "outcome": "cesd",
        "rope": (-2.0, 2.0),
        "threshold": 5.0,
        "prefix": "help_cesd_by_sex",
        "hypothesis": "女性のほうが抑うつ得点 (CES-D) が高い",
        "note": "対比用: Welch p=0.0003 で明確に有意。同じ集団・同じ n でも変数次第で結論は変わる。",
    },
}

DEFAULT_CASE = "sex-maxdrinks"


def _auto_levels(d: "pd.DataFrame", group: str, outcome: str) -> tuple[str, str]:
    """2 水準の群について、outcome の平均が大きい水準を先にして返す。"""
    levs = sorted(str(x) for x in d[group].dropna().unique())
    if len(levs) != 2:
        raise SystemExit(
            f"列 '{group}' は {len(levs)} 水準 ({', '.join(levs)}) あります。"
            " --levels G1 G2 で比較する2水準を指定してください。"
        )
    means = {lv: d.loc[d[group] == lv, outcome].mean() for lv in levs}
    return tuple(sorted(levs, key=lambda lv: means[lv], reverse=True))  # type: ignore[return-value]


def _validate_levels(d: "pd.DataFrame", group: str, levels: tuple[str, str]) -> None:
    present = set(str(x) for x in d[group].dropna().unique())
    missing = [lv for lv in levels if lv not in present]
    if missing:
        raise SystemExit(
            f"列 '{group}' に水準 {missing} は存在しません。"
            f" 使える水準: {', '.join(sorted(present))}"
        )
    if levels[0] == levels[1]:
        raise SystemExit("--levels には異なる2水準を指定してください。")


def load_pair(
    group: str, outcome: str, levels: tuple[str, str]
) -> tuple["pd.Series", "pd.Series"]:
    """CSV から (levels[0], levels[1]) の outcome 列を返す。"""
    d = pd.read_csv(CSV_PATH)
    _validate_levels(d, group, levels)
    y1 = d.loc[d[group] == levels[0], outcome]
    y2 = d.loc[d[group] == levels[1], outcome]
    return y1, y2


def resolve_config(args: argparse.Namespace) -> dict:
    """--case とその個別上書きから、実行する比較の設定を組み立てる。"""
    base = dict(CASES[args.case or DEFAULT_CASE])
    from_case = args.case is not None or not any(
        (args.group, args.outcome, args.levels)
    )

    group = args.group or base["group"]
    outcome = args.outcome or base["outcome"]

    d = pd.read_csv(CSV_PATH)
    if args.levels is not None:
        levels = tuple(args.levels)
    elif args.group is not None:
        # 群分け列を変えたらプリセットの水準は無効。平均の大きい方を先に自動決定。
        levels = _auto_levels(d, group, outcome)
    else:
        levels = tuple(base["levels"])

    rope = tuple(args.rope) if args.rope is not None else base.get("rope")
    threshold = args.threshold if args.threshold is not None else (
        base.get("threshold") if from_case else None
    )

    if from_case and args.levels is None and args.group is None and args.outcome is None:
        hypothesis = base.get("hypothesis")
        note = base.get("note")
    else:
        hypothesis = f"「{levels[0]} のほうが {levels[1]} より {outcome} が大きい」"
        note = None

    prefix = args.prefix or (
        base.get("prefix")
        if from_case and args.group is None and args.outcome is None
        else f"help_{outcome}_by_{group}"
    )

    return {
        "group": group,
        "outcome": outcome,
        "levels": levels,
        "rope": rope,
        "threshold": threshold,
        "hypothesis": hypothesis,
        "note": note,
        "prefix": prefix,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="help_maxdrinks_by_sex.py",
        description=(
            "HELP study データで『t 検定ではぎりぎり非有意 (p が 0.05 の少し上) だが、"
            "順位検定では有意』な2群比較を、外れ値にロバストなベイズ推定 (BEST) で見直すサンプル。"
        ),
        epilog=(
            "例:\n"
            "  %(prog)s                       # 既定: 性別 × 1日最大飲酒量\n"
            "  %(prog)s --case homeless-cesd  # 住居状況 × 抑うつ得点\n"
            "  %(prog)s --list-cases          # 用意した比較例を一覧表示\n"
            "  %(prog)s --group substance --outcome cesd --levels alcohol heroin\n"
            "  %(prog)s --case sex-avgdrinks --rope -0.5 0.5 --threshold 2\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--case",
        choices=list(CASES),
        help=f"事前定義の比較例を選ぶ (既定: {DEFAULT_CASE})。--list-cases で説明を表示。",
    )
    p.add_argument(
        "--list-cases",
        action="store_true",
        help="用意した比較例の一覧 (想定仮説と目安の p 値) を表示して終了する。",
    )
    p.add_argument(
        "--group",
        choices=GROUP_COLUMNS,
        help="群分けに使う列。--case の設定を上書きする。",
    )
    p.add_argument(
        "--outcome",
        choices=NUMERIC_COLUMNS,
        help="比較する量的変数。--case の設定を上書きする。",
    )
    p.add_argument(
        "--levels",
        nargs=2,
        metavar=("G1", "G2"),
        help="比較する2水準。差は G1 - G2 の向きで計算する (substance など3水準の列で必須)。",
    )
    p.add_argument(
        "--rope",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="ROPE (実質的同等領域) の区間。分析前に領域知識から決めるのが本来の運用。",
    )
    p.add_argument(
        "--threshold",
        type=float,
        help="P(平均差 > 閾値 | データ) を問い合わせる「実質的に意味のある差」の閾値。",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260101,
        help="サンプリングの乱数シード (既定: 20260101)。",
    )
    p.add_argument(
        "--prefix",
        default=None,
        help="出力 PNG の接頭辞 (既定はケース/列名から自動決定)。",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="図の保存を省略する。",
    )
    return p


def print_cases() -> None:
    print("利用できる比較例 (--case):\n")
    for name, c in CASES.items():
        g1, g2 = c["levels"]
        default_mark = "  (既定)" if name == DEFAULT_CASE else ""
        print(f"  {name}{default_mark}")
        print(f"      {c['group']}: {g1} vs {g2}  ×  {c['outcome']}")
        print(f"      想定仮説: {c['hypothesis']}")
        print(f"      {c['note']}\n")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.list_cases:
        print_cases()
        return

    cfg = resolve_config(args)
    y1, y2 = load_pair(cfg["group"], cfg["outcome"], cfg["levels"])
    g1, g2 = cfg["levels"]

    print(f"比較     : {cfg['group']} = {g1} (n={y1.size}) vs {g2} (n={y2.size})")
    print(f"変数     : {cfg['outcome']}")
    print(f"想定仮説 : {cfg['hypothesis']}")
    if cfg["note"]:
        print(f"メモ     : {cfg['note']}")
    print()

    result = analyze_two(
        y1,
        y2,
        group_names=(g1, g2),
        rope=cfg["rope"],
        random_seed=args.seed,
    )

    print(result.report())
    print()
    print(result.summary().to_string())
    print()

    # 「平均差が実質的に意味のある大きさである」事後確率を直接問い合わせる。
    if cfg["threshold"] is not None:
        thr = cfg["threshold"]
        p = result.posterior_prob("diff_of_means", low=thr)
        print(f"P(平均の差 > {thr:g} | データ) = {p:.3f}")

    if args.no_plot:
        return

    prefix = cfg["prefix"]
    fig = result.plot_all()
    fig.savefig(f"{prefix}_posterior.png", dpi=120)
    fig2 = plot_data_with_ppc(result)
    fig2.savefig(f"{prefix}_ppc.png", dpi=120)
    print(f"\n図を {prefix}_posterior.png / {prefix}_ppc.png に保存しました。")


if __name__ == "__main__":
    main()
