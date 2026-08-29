import numpy as np
import pytest

from best_pymc import analyze_one, analyze_two, hdi
from best_pymc.core import (
    BestResult,
    _cohen_d,
    _format_ttest_power_line,
    _ttest_power_analysis,
)

FAST = dict(draws=500, tune=500, chains=2, random_seed=42)


def test_hdi_normal():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200_000)
    lo, hi = hdi(x, 0.95)
    assert lo == pytest.approx(-1.96, abs=0.05)
    assert hi == pytest.approx(1.96, abs=0.05)


def test_hdi_rejects_bad_prob():
    with pytest.raises(ValueError):
        hdi(np.arange(10), prob=1.5)


def test_recovers_known_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(10.0, 1.0, 200)
    b = rng.normal(7.0, 1.0, 200)
    res = analyze_two(a, b, **FAST)
    lo, hi = res.hdi("diff_of_means")
    assert lo < 3.0 < hi
    assert res.posterior_prob("diff_of_means", low=0.0) > 0.99


def test_robust_to_outliers():
    """外れ値を足しても平均差の推定がほとんど動かないこと（t 尤度の効果）。"""
    rng = np.random.default_rng(2)
    a = rng.normal(10.0, 1.0, 60)
    b = rng.normal(10.0, 1.0, 60)
    clean = analyze_two(a, b, **FAST).samples("diff_of_means").mean()
    a_out = np.append(a, [200.0, -200.0])
    dirty = analyze_two(a_out, b, **FAST).samples("diff_of_means").mean()
    assert abs(clean - dirty) < 0.5


def test_rope_decision_equivalent():
    rng = np.random.default_rng(3)
    a = rng.normal(10.0, 1.0, 400)
    b = rng.normal(10.0, 1.0, 400)
    res = analyze_two(a, b, rope=(-0.5, 0.5), **FAST)
    d = res.rope_decision()
    assert d["decision"] in {"practically_equivalent", "undecided"}
    assert 0.0 <= d["prob_in_rope"] <= 1.0


def test_rope_decision_different():
    rng = np.random.default_rng(4)
    a = rng.normal(15.0, 1.0, 200)
    b = rng.normal(10.0, 1.0, 200)
    res = analyze_two(a, b, rope=(-0.5, 0.5), **FAST)
    assert res.rope_decision()["decision"] == "different"


def test_one_sample():
    rng = np.random.default_rng(5)
    d = rng.normal(2.0, 1.0, 150)
    res = analyze_one(d, ref=0.0, rope=(-0.2, 0.2), **FAST)
    lo, hi = res.hdi("diff_of_means")
    assert lo < 2.0 < hi
    assert "effect_size" in res.idata.posterior


def test_separate_nu():
    rng = np.random.default_rng(6)
    a = rng.normal(0.0, 1.0, 80)
    b = rng.standard_t(3, 80)
    res = analyze_two(a, b, nu_shared=False, **FAST)
    assert "nu1" in res.idata.posterior
    assert "nu2" in res.idata.posterior


def test_sigma_prior_variants():
    rng = np.random.default_rng(7)
    a = rng.normal(5.0, 2.0, 80)
    b = rng.normal(4.0, 2.0, 80)
    for sp in ["weak", "kruschke", (0.1, 20.0)]:
        res = analyze_two(a, b, sigma_prior=sp, **FAST)
        assert np.isfinite(res.samples("diff_of_means")).all()


def test_cohen_d_known_values():
    y1 = np.array([1.0, 2.0, 3.0])
    y2 = np.array([0.0, 1.0, 2.0])
    assert _cohen_d(y1, y2) == pytest.approx(1.0)
    assert _cohen_d(y1) == pytest.approx(2.0)
    assert np.isnan(_cohen_d(np.array([5.0, 5.0, 5.0])))


def test_welch_ttest_includes_effect_size():
    two = BestResult(
        idata=object(),
        model=object(),  # type: ignore[arg-type]
        data={
            "y1": np.array([1.0, 2.0, 3.0]),
            "y2": np.array([0.0, 1.0, 2.0]),
        },
        group_names=("a", "b"),
    )
    w = two.welch_ttest()
    assert w["test"] == "Welch t-test"
    assert set(w) >= {"t", "p_value", "cohen_d"}
    assert w["cohen_d"] == pytest.approx(1.0)

    one = BestResult(
        idata=object(),
        model=object(),  # type: ignore[arg-type]
        data={"y1": np.array([1.0, 2.0, 3.0]), "y2": np.array([0.0])},
        group_names=("g", "ref=0"),
        settings={"one_sample": True},
    )
    w1 = one.welch_ttest()
    assert w1["test"] == "one-sample t-test"
    assert w1["cohen_d"] == pytest.approx(2.0)


def test_ttest_power_gpower_conventions():
    """G*Power の定番: 両側 α=0.05, d=0.5 で 80% に必要な n。"""
    two = _ttest_power_analysis(0.5, 64, 64)
    assert two["observed_power"] == pytest.approx(0.80, abs=0.01)
    assert two["mdes"] == pytest.approx(0.50, abs=0.01)
    assert two["n_required"] == 64

    one = _ttest_power_analysis(0.5, 34, None)
    assert one["observed_power"] == pytest.approx(0.80, abs=0.02)
    assert one["n_required"] == 34
    assert one["n_required_is_per_group"] is False

    zero = _ttest_power_analysis(0.0, 50, 50)
    assert zero["observed_power"] == pytest.approx(0.05, abs=0.005)
    assert np.isinf(zero["n_required"])


def test_ttest_power_on_result_two_and_one_sample():
    two = BestResult(
        idata=object(),
        model=object(),  # type: ignore[arg-type]
        data={
            "y1": np.array([1.0, 2.0, 3.0]),
            "y2": np.array([0.0, 1.0, 2.0]),
        },
        group_names=("a", "b"),
    )
    p2 = two.ttest_power()
    assert 0.0 < p2["observed_power"] <= 1.0
    assert p2["n_required_is_per_group"] is True
    line = _format_ttest_power_line(p2)
    assert "事後" in line and "感度" in line and "事前" in line
    assert "/群" in line

    one = BestResult(
        idata=object(),
        model=object(),  # type: ignore[arg-type]
        data={"y1": np.array([1.0, 2.0, 3.0]), "y2": np.array([0.0])},
        group_names=("g", "ref=0"),
        settings={"one_sample": True},
    )
    p1 = one.ttest_power()
    assert p1["one_sample"] is True
    assert p1["n_required_is_per_group"] is False
    assert "/群" not in _format_ttest_power_line(p1)


def test_report_and_summary_run():
    rng = np.random.default_rng(8)
    a = rng.normal(5.0, 2.0, 60)
    b = rng.normal(4.0, 2.0, 60)
    res = analyze_two(a, b, rope=(-0.5, 0.5), **FAST)
    text = res.report()
    assert "HDI" in text
    assert "d =" in text
    assert "検定力" in text
    assert "事後 1-β=" in text
    df = res.summary()
    assert set(df.index) == {"diff_of_means", "diff_of_stds", "effect_size"}


def test_rejects_tiny_input():
    with pytest.raises(ValueError):
        analyze_two([1.0], [1.0, 2.0], **FAST)


def test_plot_all_overlaid_runs():
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.default_rng(3)
    a = rng.normal(5.0, 2.0, 50)
    b = rng.normal(4.0, 2.0, 50)
    res = analyze_two(a, b, group_names=("A", "B"), **FAST)

    fig = res.plot_all(overlaid=True)
    # 平均・標準偏差が群ごとに 1 枚へまとまるので、並列描画よりパネルが減る
    n_overlaid = len(fig.axes)
    n_side = len(res.plot_all(overlaid=False).axes)
    assert n_overlaid < n_side
    # 重ねたパネルには両群の凡例が付く
    legend_texts = {
        t.get_text()
        for ax in fig.axes
        if ax.get_legend()
        for t in ax.get_legend().get_texts()
    }
    assert {"A", "B"} <= legend_texts
