"""BEST (Bayesian ESTimation) の中核実装。

Kruschke, J. K. (2013). Bayesian estimation supersedes the t test.
Journal of Experimental Psychology: General, 142(2), 573-603.
https://doi.org/10.1037/a0029146

モデル定義は PyMC 公式 example gallery の BEST ノートブック (MIT License)
https://www.pymc.io/projects/examples/en/latest/case_studies/BEST.html
を基に、現行 PyMC 向けに再構成したもの。
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np
import pymc as pm

__all__ = [
    "BestResult",
    "analyze_two",
    "analyze_one",
    "build_two_group_model",
    "hdi",
]

SigmaPrior = Literal["weak", "kruschke"] | tuple[float, float]


# --------------------------------------------------------------------------
# ユーティリティ
# --------------------------------------------------------------------------
def hdi(samples: np.ndarray, prob: float = 0.95) -> tuple[float, float]:
    """最高密度区間 (HDI) を返す。

    ArviZ のバージョン差に依存しないよう、単峰性を仮定した
    「与えられた確率質量を含む最短区間」として自前で計算する。
    """
    if not 0.0 < prob < 1.0:
        raise ValueError("prob は 0 と 1 の間である必要があります")
    x = np.sort(np.asarray(samples).ravel())
    n = x.size
    n_in = int(np.floor(prob * n))
    if n_in < 1 or n_in >= n:
        return float(x[0]), float(x[-1])
    widths = x[n_in:] - x[: n - n_in]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + n_in])


def _flat(idata, name: str) -> np.ndarray:
    """posterior から (chain, draw) を潰した 1 次元配列を取り出す。"""
    return np.asarray(idata.posterior[name].values).ravel()


def _resolve_cores(cores: int | None, chains: int) -> int:
    """利用可能なコア数を安全に決める。

    PyMC は環境によっては cores を 0 と算出して ZeroDivisionError を出すこと
    があるため（CPU 1 コアのコンテナなど）、ここで必ず 1 以上に丸める。
    """
    if cores is not None:
        return max(1, int(cores))
    try:
        avail = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        avail = os.cpu_count() or 1
    return max(1, min(int(chains), avail))


def _as_array(y: Iterable[float], label: str) -> np.ndarray:
    arr = np.asarray(list(y), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise ValueError(f"{label} には有限な観測値が 2 個以上必要です")
    return arr


# --------------------------------------------------------------------------
# モデル構築
# --------------------------------------------------------------------------
def _add_sigma_prior(name: str, s_pooled: float, sigma_prior: SigmaPrior):
    """σ の事前分布を追加する。

    "weak"     : HalfStudentT(nu=3, sigma=2*s_pooled)  — 弱情報事前分布（既定）
    "kruschke" : Uniform(s/1000, s*1000)               — 原論文どおり（極めて広い）
    (low, high): Uniform(low, high)                    — 分野知識で明示指定
    """
    if sigma_prior == "weak":
        return pm.HalfStudentT(name, nu=3, sigma=2.0 * s_pooled)
    if sigma_prior == "kruschke":
        return pm.Uniform(name, lower=s_pooled / 1000.0, upper=s_pooled * 1000.0)
    low, high = sigma_prior  # type: ignore[misc]
    if not 0 < low < high:
        raise ValueError("sigma_prior=(low, high) は 0 < low < high である必要があります")
    return pm.Uniform(name, lower=float(low), upper=float(high))


def build_two_group_model(
    y1: np.ndarray,
    y2: np.ndarray,
    *,
    sigma_prior: SigmaPrior = "weak",
    nu_mean: float = 30.0,
    nu_shared: bool = True,
    mu_prior_sd_factor: float = 2.0,
) -> pm.Model:
    """BEST の 2 群モデルを構築して返す（サンプリングはしない）。

    y_k[i] ~ StudentT(nu, mu_k, sigma_k),  k = 1, 2

    Parameters
    ----------
    sigma_prior
        σ の事前分布。上記 `_add_sigma_prior` 参照。
    nu_mean
        正規性パラメータ ν の事前平均。ν - 1 ~ Exponential(1/(nu_mean-1))。
    nu_shared
        True なら両群で ν を共有（原論文の仮定）。False なら群ごとに推定する。
        群によって裾の重さが異なりうる場合は False を検討する。
    mu_prior_sd_factor
        μ の事前分布 Normal(pooled_mean, factor * pooled_sd) の係数。
    """
    pooled = np.concatenate([y1, y2])
    mu_m = float(pooled.mean())
    s_pooled = float(pooled.std(ddof=1))
    if s_pooled <= 0:
        raise ValueError("全観測値が同一のため標準偏差が 0 です")
    mu_s = mu_prior_sd_factor * s_pooled

    with pm.Model() as model:
        mu1 = pm.Normal("mu1", mu=mu_m, sigma=mu_s)
        mu2 = pm.Normal("mu2", mu=mu_m, sigma=mu_s)

        sigma1 = _add_sigma_prior("sigma1", s_pooled, sigma_prior)
        sigma2 = _add_sigma_prior("sigma2", s_pooled, sigma_prior)

        if nu_shared:
            nu_minus_one = pm.Exponential("nu_minus_one", 1.0 / (nu_mean - 1.0))
            nu1 = nu2 = pm.Deterministic("nu", nu_minus_one + 1.0)
            pm.Deterministic("log10_nu", pm.math.log(nu1) / np.log(10.0))
        else:
            nu1_m1 = pm.Exponential("nu1_minus_one", 1.0 / (nu_mean - 1.0))
            nu2_m1 = pm.Exponential("nu2_minus_one", 1.0 / (nu_mean - 1.0))
            nu1 = pm.Deterministic("nu1", nu1_m1 + 1.0)
            nu2 = pm.Deterministic("nu2", nu2_m1 + 1.0)

        pm.StudentT("y1", nu=nu1, mu=mu1, sigma=sigma1, observed=y1)
        pm.StudentT("y2", nu=nu2, mu=mu2, sigma=sigma2, observed=y2)

        pm.Deterministic("diff_of_means", mu1 - mu2)
        pm.Deterministic("diff_of_stds", sigma1 - sigma2)
        pm.Deterministic(
            "effect_size",
            (mu1 - mu2) / pm.math.sqrt((sigma1**2 + sigma2**2) / 2.0),
        )
    return model


# --------------------------------------------------------------------------
# 結果オブジェクト
# --------------------------------------------------------------------------
@dataclass
class BestResult:
    """BEST の推定結果。事後分布の要約・ROPE 判定・診断をまとめて持つ。"""

    idata: object
    model: pm.Model
    data: dict[str, np.ndarray]
    group_names: tuple[str, str]
    settings: dict = field(default_factory=dict)

    @property
    def key_vars(self) -> dict[str, str]:
        """既定で要約する変数名 -> 表示ラベル。"""
        if self.settings.get("one_sample"):
            return {
                "diff_of_means": f"平均と基準値の差 ({self.group_names[1]})",
                "effect_size": "効果量 (標準化平均差)",
            }
        return {
            "diff_of_means": "平均の差 (mu1 - mu2)",
            "diff_of_stds": "標準偏差の差",
            "effect_size": "効果量 (標準化平均差)",
        }

    # ---- 事後分布へのアクセス -------------------------------------------
    def samples(self, var: str) -> np.ndarray:
        """指定変数の事後サンプル（1 次元）。"""
        return _flat(self.idata, var)

    def hdi(self, var: str, prob: float = 0.95) -> tuple[float, float]:
        """指定変数の HDI。"""
        return hdi(self.samples(var), prob)

    def posterior_prob(
        self, var: str, low: float = -np.inf, high: float = np.inf
    ) -> float:
        """P(low < var < high | data) を返す。"""
        x = self.samples(var)
        return float(np.mean((x > low) & (x < high)))

    # ---- ROPE 判定 -------------------------------------------------------
    def rope_decision(
        self, var: str = "diff_of_means", rope: tuple[float, float] | None = None,
        prob: float = 0.95,
    ) -> dict:
        """HDI + ROPE による判定。

        ROPE (Region of Practical Equivalence, 実質的同等領域) は
        「実質的に差がないとみなせる範囲」であり、**データから決めるものではなく
        分析前に領域知識から決めるもの**である。設定根拠は必ず論文に明記すること。

        判定規則 (Kruschke):
          - HDI が完全に ROPE の中   -> "practically_equivalent" (実質的同等と判断)
          - HDI が ROPE と全く重ならない -> "different" (差があると判断)
          - それ以外                  -> "undecided" (判断保留 = データ不足)
        """
        rope = rope or self.settings.get("rope")
        if rope is None:
            raise ValueError("rope=(low, high) を指定してください")
        lo, hi = self.hdi(var, prob)
        r_lo, r_hi = float(rope[0]), float(rope[1])
        if lo >= r_lo and hi <= r_hi:
            decision = "practically_equivalent"
        elif hi < r_lo or lo > r_hi:
            decision = "different"
        else:
            decision = "undecided"
        return {
            "variable": var,
            "hdi_prob": prob,
            "hdi": (lo, hi),
            "rope": (r_lo, r_hi),
            "prob_in_rope": self.posterior_prob(var, r_lo, r_hi),
            "decision": decision,
        }

    # ---- 診断 -------------------------------------------------------------
    def diagnostics(self) -> dict:
        """R-hat / ESS / divergence をまとめる。"""
        import arviz as az

        free = [rv.name for rv in self.model.free_RVs]
        rhat_ds = az.rhat(self.idata, var_names=free)
        ess_ds = az.ess(self.idata, var_names=free)
        rhat_max = float(max(float(np.asarray(rhat_ds[v]).max()) for v in rhat_ds))
        ess_min = float(min(float(np.asarray(ess_ds[v]).min()) for v in ess_ds))
        n_div = 0
        sstats = getattr(self.idata, "sample_stats", None)
        if sstats is not None and "diverging" in sstats:
            n_div = int(np.asarray(sstats["diverging"].values).sum())
        return {"r_hat_max": rhat_max, "ess_min": ess_min, "divergences": n_div}

    def check(self, rhat_tol: float = 1.01, ess_min: float = 400.0) -> list[str]:
        """収束上の問題を文字列リストで返す（問題なければ空）。"""
        d = self.diagnostics()
        issues = []
        if d["r_hat_max"] > rhat_tol:
            issues.append(f"R-hat 最大値 {d['r_hat_max']:.3f} > {rhat_tol}")
        if d["ess_min"] < ess_min:
            issues.append(f"有効サンプルサイズ最小値 {d['ess_min']:.0f} < {ess_min:.0f}")
        if d["divergences"] > 0:
            issues.append(f"発散遷移 {d['divergences']} 件")
        return issues

    # ---- 要約 -------------------------------------------------------------
    def summary(self, prob: float = 0.95, var_names: Sequence[str] | None = None):
        """主要量の要約表（pandas.DataFrame）を返す。"""
        import pandas as pd

        vars_ = list(var_names) if var_names else list(self.key_vars)
        rows = []
        for v in vars_:
            x = self.samples(v)
            lo, hi = hdi(x, prob)
            rows.append(
                {
                    "variable": v,
                    "mean": x.mean(),
                    "sd": x.std(ddof=1),
                    "median": np.median(x),
                    f"hdi_{prob:.0%}_low": lo,
                    f"hdi_{prob:.0%}_high": hi,
                    "P(>0)": float(np.mean(x > 0)),
                }
            )
        return pd.DataFrame(rows).set_index("variable")

    def welch_ttest(self) -> dict:
        """比較用に Welch の t 検定も計算する（併記して報告するため）。"""
        from scipy import stats

        if self.settings.get("one_sample"):
            t, p = stats.ttest_1samp(self.data["y1"], 0.0)
            return {"test": "one-sample t-test", "t": float(t), "p_value": float(p)}
        t, p = stats.ttest_ind(self.data["y1"], self.data["y2"], equal_var=False)
        return {"test": "Welch t-test", "t": float(t), "p_value": float(p)}

    # ---- 報告文 -----------------------------------------------------------
    def report(self, prob: float = 0.95) -> str:
        """BARG (Kruschke 2021) を意識した報告用テキストを生成する。"""
        g1, g2 = self.group_names
        n1, n2 = self.data["y1"].size, self.data["y2"].size
        s = self.settings
        d = self.diagnostics()
        n_line = (
            f"データ         : n={n1} (1標本, 基準値との差)"
            if s.get("one_sample")
            else f"データ         : n({g1})={n1}, n({g2})={n2}"
        )
        lines = [
            "=" * 70,
            (
                f"BEST: ベイズ推定による1標本比較  [{g1}] vs [{g2}]"
                if s.get("one_sample")
                else f"BEST: ベイズ推定による2群比較  [{g1}] vs [{g2}]"
            ),
            "=" * 70,
            n_line,
            (
                "尤度           : StudentT(nu, mu, sigma)"
                if s.get("one_sample")
                else "尤度           : StudentT(nu, mu_k, sigma_k)"
                + ("  (nu は両群共有)" if s.get("nu_shared", True) else "  (nu は群ごと)")
            ),
            f"事前分布 mu    : Normal("
            f"{'0' if s.get('one_sample') else 'pooled_mean'}, "
            f"{s.get('mu_prior_sd_factor')} * sd)",
            f"事前分布 sigma : {s.get('sigma_prior')}",
            f"事前分布 nu    : Exponential(1/{s.get('nu_mean', 30) - 1:.0f}) + 1",
            f"サンプリング   : {s.get('chains')} chains x {s.get('draws')} draws "
            f"(tune={s.get('tune')}, target_accept={s.get('target_accept')}, "
            f"seed={s.get('random_seed')})",
            f"診断           : R-hat_max={d['r_hat_max']:.4f}, "
            f"ESS_min={d['ess_min']:.0f}, divergences={d['divergences']}",
            "-" * 70,
        ]
        issues = self.check()
        if issues:
            lines.append("【警告】収束に問題の可能性: " + " / ".join(issues))
            lines.append("-" * 70)

        df = self.summary(prob)
        for v, label in self.key_vars.items():
            r = df.loc[v]
            lines.append(
                f"{label}\n"
                f"    事後平均 = {r['mean']:.3f}  "
                f"{prob:.0%} HDI = [{r[f'hdi_{prob:.0%}_low']:.3f}, "
                f"{r[f'hdi_{prob:.0%}_high']:.3f}]  "
                f"P(>0) = {r['P(>0)']:.3f}"
            )
        if self.settings.get("rope") is not None:
            lines.append("-" * 70)
            rd = self.rope_decision(prob=prob)
            label = {
                "practically_equivalent": "実質的に同等（HDI が ROPE 内に収まる）",
                "different": "差あり（HDI が ROPE と重ならない）",
                "undecided": "判断保留（HDI が ROPE 境界をまたぐ = データ不足）",
            }[rd["decision"]]
            lines.append(
                f"ROPE 判定 (ROPE = [{rd['rope'][0]:.3f}, {rd['rope'][1]:.3f}])\n"
                f"    {label}\n"
                f"    事後確率が ROPE 内にある割合 = {rd['prob_in_rope']:.3f}"
            )
        lines.append("-" * 70)
        w = self.welch_ttest()
        lines.append(f"[参考] {w['test']}: t = {w['t']:.3f}, p = {w['p_value']:.4g}")
        lines.append("=" * 70)
        return "\n".join(lines)

    # ---- 作図 -------------------------------------------------------------
    def plot_all(self, **kwargs):
        from .plots import plot_all as _plot_all

        return _plot_all(self, **kwargs)


# --------------------------------------------------------------------------
# 高水準 API
# --------------------------------------------------------------------------
def analyze_two(
    y1: Iterable[float],
    y2: Iterable[float],
    *,
    group_names: tuple[str, str] = ("group1", "group2"),
    rope: tuple[float, float] | None = None,
    sigma_prior: SigmaPrior = "weak",
    nu_mean: float = 30.0,
    nu_shared: bool = True,
    mu_prior_sd_factor: float = 2.0,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    cores: int | None = None,
    target_accept: float = 0.9,
    random_seed: int = 20260101,
    progressbar: bool = False,
    **sample_kwargs,
) -> BestResult:
    """2 群を BEST で比較する。

    Examples
    --------
    >>> res = analyze_two(treatment, control, rope=(-1, 1))
    >>> print(res.report())
    >>> res.hdi("effect_size")
    >>> res.posterior_prob("diff_of_means", low=0.5)

    Notes
    -----
    `random_seed` は既定値を持たせてあるので、同じ入力・同じライブラリ版であれば
    結果は再現する。論文には seed とライブラリのバージョンを必ず記載すること。
    """
    a1 = _as_array(y1, group_names[0])
    a2 = _as_array(y2, group_names[1])

    model = build_two_group_model(
        a1,
        a2,
        sigma_prior=sigma_prior,
        nu_mean=nu_mean,
        nu_shared=nu_shared,
        mu_prior_sd_factor=mu_prior_sd_factor,
    )
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=_resolve_cores(cores, chains),
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            **sample_kwargs,
        )

    result = BestResult(
        idata=idata,
        model=model,
        data={"y1": a1, "y2": a2},
        group_names=group_names,
        settings={
            "sigma_prior": sigma_prior,
            "nu_mean": nu_mean,
            "nu_shared": nu_shared,
            "mu_prior_sd_factor": mu_prior_sd_factor,
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "target_accept": target_accept,
            "random_seed": random_seed,
            "rope": rope,
            "pymc_version": pm.__version__,
        },
    )
    issues = result.check()
    if issues:
        warnings.warn(
            "MCMC の収束に問題がある可能性があります: " + " / ".join(issues),
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def analyze_one(
    y: Iterable[float],
    *,
    ref: float = 0.0,
    group_name: str = "group",
    **kwargs,
) -> BestResult:
    """1 標本（対応のある差得点など）を基準値 `ref` と比較する。

    対応のある2群比較は「差得点を作って 1 標本として扱う」のが正しい扱いで、
    `analyze_two` に前後データをそのまま渡してはいけない。

    実装上は、第2群を「点 `ref` に集中した仮想群」ではなく、
    第2群の平均を `ref` に固定した 1 群モデルとして解く。
    """
    a = _as_array(y, group_name)
    d = a - ref
    mu_m = 0.0
    s = float(d.std(ddof=1))
    sigma_prior = kwargs.pop("sigma_prior", "weak")
    nu_mean = kwargs.pop("nu_mean", 30.0)
    mu_prior_sd_factor = kwargs.pop("mu_prior_sd_factor", 2.0)
    rope = kwargs.pop("rope", None)
    draws = kwargs.pop("draws", 2000)
    tune = kwargs.pop("tune", 2000)
    chains = kwargs.pop("chains", 4)
    cores = kwargs.pop("cores", None)
    target_accept = kwargs.pop("target_accept", 0.9)
    random_seed = kwargs.pop("random_seed", 20260101)
    progressbar = kwargs.pop("progressbar", False)

    with pm.Model() as model:
        mu = pm.Normal("mu1", mu=mu_m, sigma=mu_prior_sd_factor * s)
        sigma = _add_sigma_prior("sigma1", s, sigma_prior)
        nu_minus_one = pm.Exponential("nu_minus_one", 1.0 / (nu_mean - 1.0))
        nu = pm.Deterministic("nu", nu_minus_one + 1.0)
        pm.StudentT("y1", nu=nu, mu=mu, sigma=sigma, observed=d)
        pm.Deterministic("diff_of_means", mu * 1.0)
        pm.Deterministic("effect_size", mu / sigma)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=_resolve_cores(cores, chains),
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            **kwargs,
        )

    result = BestResult(
        idata=idata,
        model=model,
        data={"y1": d, "y2": np.array([ref])},
        group_names=(group_name, f"ref={ref}"),
        settings={
            "sigma_prior": sigma_prior,
            "nu_mean": nu_mean,
            "nu_shared": True,
            "mu_prior_sd_factor": mu_prior_sd_factor,
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "target_accept": target_accept,
            "random_seed": random_seed,
            "rope": rope,
            "pymc_version": pm.__version__,
            "one_sample": True,
        },
    )
    return result
