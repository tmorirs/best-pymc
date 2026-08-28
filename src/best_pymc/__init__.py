"""best_pymc — t検定の代わりにベイズ推定で2群を比較する (Kruschke の BEST)。

現行 PyMC (v5 / v6) と ArviZ 1.x で動作するように書き直した実装。

    >>> from best_pymc import analyze_two
    >>> res = analyze_two(treatment, control, rope=(-1, 1))
    >>> print(res.report())
    >>> res.plot_all()
"""

from .datasets import smart_drug
from .core import (
    BestResult,
    analyze_one,
    analyze_two,
    build_two_group_model,
    hdi,
)
from .plots import (
    japanese_font_available,
    plot_all,
    plot_data_with_ppc,
    plot_posterior,
    setup_japanese_font,
)
from .sensitivity import sensitivity_analysis

__version__ = "0.1.0"

__all__ = [
    "analyze_two",
    "analyze_one",
    "build_two_group_model",
    "BestResult",
    "hdi",
    "plot_all",
    "plot_posterior",
    "plot_data_with_ppc",
    "setup_japanese_font",
    "japanese_font_available",
    "sensitivity_analysis",
    "smart_drug",
    "__version__",
]
