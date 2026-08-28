"""BEST の最小実装（1ファイル・依存は pymc / numpy / matplotlib のみ）。

パッケージを入れずにコピペで試したい人向け。
そのまま実行すると Kruschke (2013) のスマートドラッグ例が走る。

    pip install "pymc>=5" matplotlib
    python best_minimal.py
"""

import os

import numpy as np
import pymc as pm


def best_two_group(y1, y2, draws=2000, tune=2000, chains=4, seed=20260101):
    """2群を Student-t 尤度でモデル化し、事後サンプルを返す。"""
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    pooled = np.concatenate([y1, y2])
    mu_m = pooled.mean()
    s = pooled.std(ddof=1)

    with pm.Model() as model:
        # 平均: データの中心まわりに広い正規事前分布
        mu1 = pm.Normal("mu1", mu=mu_m, sigma=2 * s)
        mu2 = pm.Normal("mu2", mu=mu_m, sigma=2 * s)

        # 標準偏差: 弱情報事前分布（原論文の Uniform(s/1000, s*1000) は広すぎる）
        sigma1 = pm.HalfStudentT("sigma1", nu=3, sigma=2 * s)
        sigma2 = pm.HalfStudentT("sigma2", nu=3, sigma=2 * s)

        # 正規性パラメータ nu: 両群で共有（原論文の仮定）
        nu = pm.Deterministic("nu", pm.Exponential("nu_m1", 1 / 29.0) + 1)

        pm.StudentT("y1", nu=nu, mu=mu1, sigma=sigma1, observed=y1)
        pm.StudentT("y2", nu=nu, mu=mu2, sigma=sigma2, observed=y2)

        # 関心量
        pm.Deterministic("diff_of_means", mu1 - mu2)
        pm.Deterministic("diff_of_stds", sigma1 - sigma2)
        pm.Deterministic(
            "effect_size",
            (mu1 - mu2) / pm.math.sqrt((sigma1**2 + sigma2**2) / 2),
        )

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=max(1, min(chains, os.cpu_count() or 1)),
            target_accept=0.9,
            random_seed=seed,
            progressbar=False,
        )
    return idata


def hdi(x, prob=0.95):
    """最高密度区間（単峰性を仮定した最短区間）。"""
    x = np.sort(np.asarray(x).ravel())
    n_in = int(np.floor(prob * x.size))
    widths = x[n_in:] - x[: x.size - n_in]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + n_in])


def report(idata, rope=None):
    for var in ["diff_of_means", "diff_of_stds", "effect_size"]:
        x = idata.posterior[var].values.ravel()
        lo, hi = hdi(x)
        line = (
            f"{var:15s} mean={x.mean():7.3f}  "
            f"95% HDI=[{lo:7.3f}, {hi:7.3f}]  P(>0)={np.mean(x > 0):.3f}"
        )
        if rope is not None and var == "diff_of_means":
            in_rope = np.mean((x > rope[0]) & (x < rope[1]))
            if lo >= rope[0] and hi <= rope[1]:
                d = "practically equivalent"
            elif hi < rope[0] or lo > rope[1]:
                d = "different"
            else:
                d = "undecided"
            line += f"\n{'':15s} ROPE{rope} 内={in_rope:.3f} -> {d}"
        print(line)


if __name__ == "__main__":
    # fmt: off
    iq_drug = [101, 100, 102, 104, 102, 97, 105, 105, 98, 101, 100, 123, 105,
               103, 100, 95, 102, 106, 109, 102, 82, 102, 100, 102, 102, 101,
               102, 102, 103, 103, 97, 97, 103, 101, 97, 104, 96, 103, 124,
               101, 101, 100, 101, 101, 104, 100, 101]
    iq_placebo = [99, 101, 100, 101, 102, 100, 97, 101, 104, 101, 102, 102,
                  100, 105, 88, 101, 100, 104, 100, 100, 100, 101, 102, 103,
                  97, 101, 101, 100, 101, 99, 101, 100, 100, 101, 100, 99,
                  101, 100, 102, 99, 100, 99]
    # fmt: on
    idata = best_two_group(iq_drug, iq_placebo)
    report(idata, rope=(-1.0, 1.0))
