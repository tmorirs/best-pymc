"""同梱データセット。

Kruschke (2013) の「スマートドラッグ」例で用いられている架空の IQ データ。
知能を高めるとされる薬を投与した群 (n=47) と、プラセボ群 (n=42) の IQ 得点。
"""

from __future__ import annotations

import numpy as np

__all__ = ["smart_drug"]

# fmt: off
_IQ_DRUG = np.array([
    101, 100, 102, 104, 102, 97, 105, 105, 98, 101, 100, 123, 105, 103,
    100, 95, 102, 106, 109, 102, 82, 102, 100, 102, 102, 101, 102, 102,
    103, 103, 97, 97, 103, 101, 97, 104, 96, 103, 124, 101, 101, 100,
    101, 101, 104, 100, 101,
], dtype=float)

_IQ_PLACEBO = np.array([
    99, 101, 100, 101, 102, 100, 97, 101, 104, 101, 102, 102, 100, 105,
    88, 101, 100, 104, 100, 100, 100, 101, 102, 103, 97, 101, 101, 100,
    101, 99, 101, 100, 100, 101, 100, 99, 101, 100, 102, 99, 100, 99,
], dtype=float)
# fmt: on


def smart_drug() -> tuple[np.ndarray, np.ndarray]:
    """(drug, placebo) の IQ 得点を返す。

    >>> drug, placebo = smart_drug()
    >>> drug.size, placebo.size
    (47, 42)
    """
    return _IQ_DRUG.copy(), _IQ_PLACEBO.copy()
