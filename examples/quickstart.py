"""自分のデータで試すための最小テンプレート。

    python examples/quickstart.py

1. 独立2群
2. 対応のあるデータ（前後比較）
3. 事前分布の感度分析
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")

from best_pymc import analyze_one, analyze_two, sensitivity_analysis  # noqa: E402

rng = np.random.default_rng(0)

# --------------------------------------------------------------------------
# 1. 独立2群
#    ここを自分のデータに置き換える（list でも numpy 配列でも可）
# --------------------------------------------------------------------------
group_a = rng.normal(52.0, 9.0, size=24)
group_b = rng.normal(48.0, 12.0, size=27)

res = analyze_two(
    group_a,
    group_b,
    group_names=("介入群", "対照群"),
    rope=(-2.0, 2.0),   # 「±2点以内なら実質差なし」という想定。必ず事前に決める
    random_seed=20260101,
)
print(res.report())

# --------------------------------------------------------------------------
# 2. 対応のあるデータ
#    前後比較は「差得点を作って1標本として扱う」。analyze_two に前後を
#    そのまま渡すのは誤り（対応関係の情報が捨てられ、不確実性を過大評価する）。
# --------------------------------------------------------------------------
pre = rng.normal(50.0, 8.0, size=30)
post = pre + rng.normal(2.5, 4.0, size=30)

res_paired = analyze_one(
    post - pre,
    ref=0.0,
    group_name="前後差",
    rope=(-1.0, 1.0),
    random_seed=20260101,
)
print()
print(res_paired.report())

# --------------------------------------------------------------------------
# 3. 事前分布の感度分析
#    「結論が事前分布の選び方に依存していない」ことを示すための表。
#    査読でほぼ必ず問われるので、論文にはこの表を載せる。
# --------------------------------------------------------------------------
print()
print("--- 事前分布の感度分析 ---")
table = sensitivity_analysis(
    group_a, group_b, rope=(-2.0, 2.0), random_seed=20260101
)
print(table.to_string())
