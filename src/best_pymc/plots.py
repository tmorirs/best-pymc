"""Kruschke (2013) 風の事後分布プロット。

ArviZ の描画 API はバージョン間で大きく変わるため、ここでは matplotlib だけを使う。
日本語フォントが見つからない環境では自動的に英語ラベルにフォールバックする。
"""

from __future__ import annotations

import numpy as np

from .core import hdi

__all__ = [
    "plot_posterior",
    "plot_posterior_overlaid",
    "plot_all",
    "plot_data_with_ppc",
    "setup_japanese_font",
    "japanese_font_available",
]

# 群を重ねて描くときの配色（見分けやすい 2 色を先頭に）
_OVERLAY_COLORS = ["#7fa8d1", "#d98c5f"]

# 日本語が使える代表的なフォント（見つかった順に採用）
_JP_FONT_CANDIDATES = [
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "IPAGothic",
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "TakaoGothic",
    "VL Gothic",
]

_LABELS = {
    "ja": {
        "mean_of": "平均",
        "sd_of": "標準偏差",
        "normality": "正規性 nu (大きいほど正規に近い)",
        "diff_means": "平均の差",
        "diff_means_1s": "平均と基準値の差",
        "diff_stds": "標準偏差の差",
        "effect": "効果量 (標準化平均差)",
        "in_rope": "ROPE内",
        "hdi": "HDI",
        "ppc": "データと事後予測",
    },
    "en": {
        "mean_of": "mean",
        "sd_of": "std",
        "normality": "normality nu (larger = closer to normal)",
        "diff_means": "Difference of means",
        "diff_means_1s": "Difference from reference",
        "diff_stds": "Difference of stds",
        "effect": "Effect size (standardized)",
        "in_rope": "in ROPE",
        "hdi": "HDI",
        "ppc": "data & posterior predictive",
    },
}


def japanese_font_available() -> str | None:
    """利用可能な日本語フォント名を返す（無ければ None）。"""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _JP_FONT_CANDIDATES:
        if name in installed:
            return name
    return None


def setup_japanese_font(font: str | None = None) -> str | None:
    """matplotlib に日本語フォントを設定する。設定できたフォント名を返す。

    見つからない場合は None を返し、rcParams は変更しない。
    Ubuntu 系なら `apt install fonts-noto-cjk`、
    pip なら `pip install japanize-matplotlib` で導入できる。
    """
    import matplotlib as mpl

    name = font or japanese_font_available()
    if name is None:
        return None
    mpl.rcParams["font.family"] = name
    mpl.rcParams["axes.unicode_minus"] = False
    return name


def _fmt(value: float, scale: float) -> str:
    """区間幅 `scale` に応じて有効桁を決めて数値を整形する。

    `f"{x:.3g}"` だけだと平均 101.6 が "102" に丸められてしまうため、
    表示したい区間の幅から必要な小数桁を逆算する。
    """
    if not np.isfinite(scale) or scale <= 0:
        return f"{value:.3g}"
    decimals = int(max(0, min(6, 2 - np.floor(np.log10(scale)))))
    return f"{value:.{decimals}f}"


def _resolve_lang(lang: str) -> dict[str, str]:
    if lang == "auto":
        lang = "ja" if setup_japanese_font() else "en"
    return _LABELS[lang]


def plot_posterior(
    ax,
    samples: np.ndarray,
    *,
    title: str = "",
    ref_val: float | None = None,
    rope: tuple[float, float] | None = None,
    prob: float = 0.95,
    bins: int = 60,
    point: str = "mean",
    lang: str = "auto",
):
    """1 変数の事後分布を、HDI・参照値・ROPE つきで描く。"""
    L = _resolve_lang(lang)
    x = np.asarray(samples).ravel()
    lo, hi = hdi(x, prob)
    center = float(np.mean(x)) if point == "mean" else float(np.median(x))

    ax.hist(x, bins=bins, density=True, color="#7fa8d1", edgecolor="none", alpha=0.9)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)

    ymax = ax.get_ylim()[1]

    # HDI バー
    span = hi - lo
    ax.plot([lo, hi], [ymax * 0.03] * 2, color="black", lw=3, solid_capstyle="butt")
    ax.text(lo, ymax * 0.08, _fmt(lo, span), ha="center", va="bottom", fontsize=9)
    ax.text(hi, ymax * 0.08, _fmt(hi, span), ha="center", va="bottom", fontsize=9)
    ax.text(
        (lo + hi) / 2, ymax * 0.16, f"{prob:.0%} {L['hdi']}",
        ha="center", va="bottom", fontsize=9,
    )

    # 点推定
    ax.text(
        center, ymax * 0.92, f"{point} = {_fmt(center, span)}",
        ha="center", va="center", fontsize=10,
    )

    # 参照値（通常は 0）とその両側の事後確率
    if ref_val is not None:
        p_gt = float(np.mean(x > ref_val))
        ax.axvline(ref_val, color="#c0392b", lw=1.5, ls="--")
        ax.text(
            ref_val, ymax * 0.78,
            f"{(1 - p_gt):.1%} < {ref_val:g} < {p_gt:.1%}",
            ha="center", va="center", fontsize=9, color="#c0392b",
        )

    # ROPE
    if rope is not None:
        ax.axvspan(rope[0], rope[1], color="#27ae60", alpha=0.15, zorder=0)
        p_in = float(np.mean((x > rope[0]) & (x < rope[1])))
        ax.text(
            (rope[0] + rope[1]) / 2, ymax * 0.64,
            f"{L['in_rope']} {p_in:.1%}",
            ha="center", va="center", fontsize=9, color="#1e7e46",
        )

    ax.set_title(title, fontsize=11)
    return ax


def plot_posterior_overlaid(
    ax,
    samples_list,
    labels,
    *,
    title: str = "",
    prob: float = 0.95,
    bins: int = 60,
    point: str = "mean",
    lang: str = "auto",
    colors=None,
):
    """複数群の事後分布を同一パネルに色分けで重ねて描く。

    `plot_posterior` が 1 群を単独で描くのに対し、こちらは二群（以上）の
    ヒストグラムを半透明で重ね、群ごとに HDI バーと点推定を段違いで並べる。
    平均どうし・標準偏差どうしを直接見比べたいときに使う。
    """
    L = _resolve_lang(lang)
    xs = [np.asarray(s).ravel() for s in samples_list]
    cols = colors or _OVERLAY_COLORS
    cols = [cols[i % len(cols)] for i in range(len(xs))]

    # 全群共通のビン端を決めて、ヒストグラムの高さを比較可能にする
    lo_all = min(float(x.min()) for x in xs)
    hi_all = max(float(x.max()) for x in xs)
    edges = np.linspace(lo_all, hi_all, bins + 1)

    for x, color in zip(xs, cols):
        ax.hist(x, bins=edges, density=True, color=color, edgecolor="none", alpha=0.5)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)

    ymax = ax.get_ylim()[1]
    n = len(xs)
    for i, (x, color, label) in enumerate(zip(xs, cols, labels)):
        lo, hi = hdi(x, prob)
        span = hi - lo
        center = float(np.mean(x)) if point == "mean" else float(np.median(x))
        # 群ごとに HDI バーの高さをずらして重なりを避ける
        y_bar = ymax * (0.05 + 0.09 * (n - 1 - i))
        ax.plot([lo, hi], [y_bar] * 2, color=color, lw=3, solid_capstyle="butt")
        ax.text(
            center, y_bar, f" {label}: {_fmt(center, span)}",
            ha="left" if center <= (lo_all + hi_all) / 2 else "right",
            va="bottom", fontsize=9, color=color,
        )

    handles = [
        plt_patch(color, label) for color, label in zip(cols, labels)
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper right", frameon=False)
    ax.text(
        0.5, 1.0, f"{prob:.0%} {L['hdi']}",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#555555",
    )
    ax.set_title(title, fontsize=11)
    return ax


def plt_patch(color, label):
    """凡例用の色パッチ（matplotlib を関数内で import するためのヘルパ）。"""
    from matplotlib.patches import Patch

    return Patch(facecolor=color, edgecolor="none", alpha=0.5, label=label)


def plot_all(
    result, *, prob: float = 0.95, figsize=None, rope=None, lang: str = "auto",
    overlaid: bool = False,
):
    """主要パラメータと差の事後分布を一枚にまとめて描く。"""
    import matplotlib.pyplot as plt

    L = _resolve_lang(lang)
    rope = rope if rope is not None else result.settings.get("rope")
    one_sample = bool(result.settings.get("one_sample"))
    g1, g2 = result.group_names
    post = result.idata.posterior

    # パネルは 2 形式を混在させる:
    #   ("single",  var,           title, ref, rope)
    #   ("overlay", (var1, var2),  title, (label1, label2))
    if one_sample:
        # 群が 1 つなので重ね描きは効かない（単独描画のまま）
        panels = [
            ("single", "mu1", f"{L['mean_of']} ({g1})", None, None),
            ("single", "sigma1", f"{L['sd_of']} ({g1})", None, None),
            ("single", "nu", L["normality"], None, None),
            ("single", "diff_of_means", L["diff_means_1s"], 0.0, rope),
            ("single", "effect_size", L["effect"], 0.0, None),
        ]
    elif overlaid:
        panels = [
            ("overlay", ("mu1", "mu2"), L["mean_of"], (g1, g2)),
            ("overlay", ("sigma1", "sigma2"), L["sd_of"], (g1, g2)),
        ]
        if "nu" in post:
            panels.append(("single", "nu", L["normality"], None, None))
        else:
            panels.append(("overlay", ("nu1", "nu2"), L["normality"], (g1, g2)))
        panels += [
            ("single", "diff_of_means", L["diff_means"], 0.0, rope),
            ("single", "diff_of_stds", L["diff_stds"], 0.0, None),
            ("single", "effect_size", L["effect"], 0.0, None),
        ]
    else:
        panels = [
            ("single", "mu1", f"{L['mean_of']} ({g1})", None, None),
            ("single", "mu2", f"{L['mean_of']} ({g2})", None, None),
            ("single", "sigma1", f"{L['sd_of']} ({g1})", None, None),
            ("single", "sigma2", f"{L['sd_of']} ({g2})", None, None),
        ]
        if "nu" in post:
            panels.append(("single", "nu", L["normality"], None, None))
        else:
            panels.append(("single", "nu1", f"{L['normality']} ({g1})", None, None))
            panels.append(("single", "nu2", f"{L['normality']} ({g2})", None, None))
        panels += [
            ("single", "diff_of_means", L["diff_means"], 0.0, rope),
            ("single", "diff_of_stds", L["diff_stds"], 0.0, None),
            ("single", "effect_size", L["effect"], 0.0, None),
        ]

    # 対応する変数が事後分布に無いパネルは落とす
    def _panel_ok(p):
        if p[0] == "single":
            return p[1] in post
        return all(v in post for v in p[1])

    panels = [p for p in panels if _panel_ok(p)]
    ncol = 2
    nrow = int(np.ceil(len(panels) / ncol))
    figsize = figsize or (10, 2.5 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = np.asarray(axes).ravel()

    for ax, panel in zip(axes, panels):
        if panel[0] == "single":
            _, var, title, ref, rp = panel
            plot_posterior(
                ax, result.samples(var), title=title, ref_val=ref, rope=rp,
                prob=prob, lang=lang,
            )
        else:
            _, vars_, title, labels = panel
            plot_posterior_overlaid(
                ax,
                [result.samples(v) for v in vars_],
                labels,
                title=title,
                prob=prob,
                lang=lang,
            )
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(f"BEST: {g1} vs {g2}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_data_with_ppc(
    result, *, n_curves: int = 40, figsize=(10, 4), seed: int = 0, lang: str = "auto"
):
    """観測データのヒストグラムに事後予測の密度曲線を重ねる（当てはまりの目視確認）。"""
    import matplotlib.pyplot as plt
    from scipy import stats

    L = _resolve_lang(lang)
    rng = np.random.default_rng(seed)
    one_sample = bool(result.settings.get("one_sample"))
    groups = [("y1", "mu1", "sigma1", result.group_names[0])]
    if not one_sample:
        groups.append(("y2", "mu2", "sigma2", result.group_names[1]))

    fig, axes = plt.subplots(1, len(groups), figsize=figsize, squeeze=False)
    for ax, (ykey, mukey, sigkey, name) in zip(axes[0], groups):
        y = result.data[ykey]
        ax.hist(y, bins=20, density=True, color="#cccccc", edgecolor="white")
        mu = result.samples(mukey)
        sig = result.samples(sigkey)
        nu_name = (
            "nu" if "nu" in result.idata.posterior
            else ("nu1" if ykey == "y1" else "nu2")
        )
        nu = result.samples(nu_name)
        idx = rng.choice(mu.size, size=min(n_curves, mu.size), replace=False)
        grid = np.linspace(y.min() - 2 * y.std(), y.max() + 2 * y.std(), 300)
        for i in idx:
            ax.plot(
                grid,
                stats.t.pdf(grid, df=nu[i], loc=mu[i], scale=sig[i]),
                color="#7fa8d1", alpha=0.25, lw=1,
            )
        ax.set_title(f"{name}: {L['ppc']}", fontsize=11)
        ax.set_yticks([])
    fig.tight_layout()
    return fig
