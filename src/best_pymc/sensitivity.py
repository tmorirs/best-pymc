"""事前分布の感度分析。

ベイズ推定を論文に載せるとき、査読でほぼ必ず問われるのが
「その結論は事前分布の選び方に依存しないのか」という点である。
複数の事前分布設定で回して、結論（HDI と ROPE 判定）が変わらないことを
示すためのヘルパー。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .core import analyze_two

__all__ = ["sensitivity_analysis", "DEFAULT_SETTINGS"]

#: 既定で試す事前分布の組み合わせ
DEFAULT_SETTINGS: list[dict] = [
    {"label": "weak (既定)", "sigma_prior": "weak", "mu_prior_sd_factor": 2.0},
    {"label": "Kruschke 原論文", "sigma_prior": "kruschke", "mu_prior_sd_factor": 2.0},
    {"label": "mu をより広く", "sigma_prior": "weak", "mu_prior_sd_factor": 10.0},
    {"label": "nu を軽い裾寄り", "sigma_prior": "weak", "nu_mean": 10.0},
    {"label": "nu を群ごとに推定", "sigma_prior": "weak", "nu_shared": False},
]


def sensitivity_analysis(
    y1: Iterable[float],
    y2: Iterable[float],
    *,
    settings: Sequence[dict] | None = None,
    var: str = "diff_of_means",
    prob: float = 0.95,
    rope: tuple[float, float] | None = None,
    **common,
):
    """複数の事前分布設定で BEST を回し、結論の頑健性を表にして返す。

    Returns
    -------
    pandas.DataFrame
        設定ごとの事後平均・HDI・P(>0)・ROPE 判定。
    """
    import pandas as pd

    settings = list(settings or DEFAULT_SETTINGS)
    rows = []
    for cfg in settings:
        cfg = dict(cfg)
        label = cfg.pop("label", str(cfg))
        res = analyze_two(y1, y2, rope=rope, **{**common, **cfg})
        lo, hi = res.hdi(var, prob)
        row = {
            "設定": label,
            "事後平均": res.samples(var).mean(),
            f"HDI下限({prob:.0%})": lo,
            f"HDI上限({prob:.0%})": hi,
            "P(>0)": res.posterior_prob(var, low=0.0),
            "R-hat max": res.diagnostics()["r_hat_max"],
        }
        if rope is not None:
            row["ROPE判定"] = res.rope_decision(var=var, rope=rope, prob=prob)["decision"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("設定")
