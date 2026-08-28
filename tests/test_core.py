import numpy as np
import pytest

from best_pymc import analyze_one, analyze_two, hdi

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


def test_report_and_summary_run():
    rng = np.random.default_rng(8)
    a = rng.normal(5.0, 2.0, 60)
    b = rng.normal(4.0, 2.0, 60)
    res = analyze_two(a, b, rope=(-0.5, 0.5), **FAST)
    text = res.report()
    assert "HDI" in text
    df = res.summary()
    assert set(df.index) == {"diff_of_means", "diff_of_stds", "effect_size"}


def test_rejects_tiny_input():
    with pytest.raises(ValueError):
        analyze_two([1.0], [1.0, 2.0], **FAST)
