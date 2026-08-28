# best-pymc

t 検定の代わりにベイズ推定で2群を比較する **BEST** (Bayesian ESTimation supersedes the t test) を、現行の PyMC 向けに書き直した実装。

- 原論文: Kruschke, J. K. (2013). Bayesian estimation supersedes the t test. *Journal of Experimental Psychology: General*, 142(2), 573–603. [doi:10.1037/a0029146](https://doi.org/10.1037/a0029146)
- モデル定義は [PyMC 公式 example gallery の BEST ノートブック](https://www.pymc.io/projects/examples/en/latest/case_studies/BEST.html)（MIT License）を基にしている
- 旧 Python 実装 [`strawlab/best`](https://github.com/strawlab/best)（Python 2.7 / PyMC2、2022年アーカイブ）と [`treszkai/best`](https://github.com/treszkai/best)（Python 3 / PyMC3）の後継として使えるように、API を近い形にしてある

## なぜ書き直したか

既存の `best` パッケージ（`pip install best`）は依存が **PyMC3** で、PyMC3 は開発終了しているため現行環境に入れにくい。本実装は PyMC 5 以降を前提とし、加えて以下を追加している。

| | `treszkai/best` | 本実装 |
|---|---|---|
| PyMC | 3 系 | 5 / 6 系 |
| σ の事前分布 | 固定 | `weak` / `kruschke` / 明示指定 から選択 |
| ν（正規性） | 両群共有のみ | 群ごとの推定も可 (`nu_shared=False`) |
| ROPE 判定 | なし | `rope_decision()` で3値判定 |
| 事前分布の感度分析 | なし | `sensitivity_analysis()` |
| 収束診断 | 手動 | 自動チェック＋警告 |
| 報告文生成 | なし | `report()`（BARG 準拠を意識） |
| 頻度論との併記 | なし | Welch の t 検定を自動併記 |

## 動作確認済み環境

Python 3.12 / PyMC 6.3.1 / ArviZ 1.3.0 / NumPy 2.4.4 / SciPy 1.17.1 でテスト（11 件）が通ることを確認済み。`uv sync` → `uv run pytest` → `uv build` も同環境で確認済みで、`pip` 経由と `uv` 経由で同じ seed から同じ事後平均・HDI が出ることも確認しています。ArviZ は 0.x と 1.x で API が大きく変わるため、HDI などの数値計算は ArviZ に依存せず自前で実装してある（`az.rhat` / `az.ess` のみ使用。両系統に存在する関数）。

## インストール

### uv（推奨）

```bash
uv sync                 # 仮想環境の作成・依存解決・本体の編集可能インストールまで一括
uv run best-demo        # 同梱の Kruschke 例をそのまま実行
uv run pytest           # テスト
```

`uv sync` は `.python-version`（3.12）に従って Python を用意し、`uv.lock` に
解決済みバージョンを固定します。**このロックファイルをリポジトリに含めておけば、
別のマシンでも同じライブラリ版で再現できます**。MCMC の再現性は seed だけでなく
ライブラリ版にも依存するので、論文用の解析ではロックファイルごと保存しておくのが安全です。

本体だけ入れて開発用ツールを省く場合:

```bash
uv sync --no-dev
```

自分のスクリプトから使うだけなら、プロジェクトに追加する形でも入ります:

```bash
uv add "best-pymc @ /path/to/best-pymc"   # ローカルパス
uv pip install -e /path/to/best-pymc      # pip 互換インターフェース
```

インストールせずに1回だけ動かすこともできます:

```bash
uvx --from /path/to/best-pymc best-demo --rope -1 1
```

### pip

```bash
pip install -e .          # リポジトリ直下で
pip install -e ".[test]"  # テストも動かす場合
```

## コマンドラインから使う

インストールすると `best-demo` コマンドが入ります。

```bash
# 同梱の Kruschke 例
uv run best-demo --rope -1 1

# 自分の CSV（長形式：群を表す列と測定値の列）
uv run best-demo --csv data.csv --group-column arm --value-column score \
    --rope -2 2 --plot posterior.png --sensitivity
```

主なオプション: `--rope LOW HIGH` / `--draws` / `--tune` / `--chains` /
`--seed` / `--hdi-prob` / `--plot PATH` / `--sensitivity`。

## 使い方

```python
from best_pymc import analyze_two

res = analyze_two(
    treatment, control,                 # list でも numpy 配列でも可
    group_names=("介入群", "対照群"),
    rope=(-2.0, 2.0),                   # 実質的同等領域。分析前に決めること
    random_seed=20260101,
)

print(res.report())                     # 事前分布・診断・HDI・ROPE 判定をまとめて出力
res.hdi("effect_size")                  # (0.12, 1.27)
res.posterior_prob("diff_of_means", low=0.5)   # P(差 > 0.5 | データ)
res.rope_decision()                     # {'decision': 'different', ...}
res.summary()                           # pandas.DataFrame
res.plot_all()                          # Kruschke 風の事後分布プロット
```

対応のあるデータ（前後比較）は差得点を作って1標本として扱う:

```python
from best_pymc import analyze_one
res = analyze_one(post - pre, ref=0.0, rope=(-1.0, 1.0))
```

事前分布の感度分析:

```python
from best_pymc import sensitivity_analysis
print(sensitivity_analysis(a, b, rope=(-2.0, 2.0)))
```

すぐ試すなら:

```bash
uv run python examples/smart_drug.py      # Kruschke の原典の例を再現
uv run python examples/quickstart.py      # 自分のデータに差し替えるテンプレート
uv run python standalone/best_minimal.py  # 依存最小・1ファイル版
```

同梱データは `from best_pymc import smart_drug` で `(drug, placebo)` として取り出せます。

## モデル

```
y_k[i] ~ StudentT(ν, μ_k, σ_k)          k = 1, 2

μ_k    ~ Normal(pooled_mean, 2 × pooled_sd)
σ_k    ~ HalfStudentT(3, 2 × pooled_sd)   ← 既定 ("weak")
ν − 1  ~ Exponential(1 / 29)              ← 事前平均 30
```

正規分布ではなく t 分布を使うことで外れ値に頑健になり、ν が「データの正規性」を表す。σ を群ごとに推定するので、Welch の t 検定に相当する分散不均一の扱いが既定になっている。

### σ の事前分布について

原論文は σ に `Uniform(s/1000, s×1000)` を置くが、これは実質ありえない値域に大きな事前確率を与える悪い選択で、PyMC 公式ノートブックでも同様に指摘されている。本実装の既定 `"weak"` は `HalfStudentT(ν=3, σ=2s)` の弱情報事前分布。原論文どおりに再現したい場合は `sigma_prior="kruschke"`、分野知識で範囲を切りたい場合は `sigma_prior=(low, high)` を渡す。

## 使ううえでの注意

**ROPE は分析前に決める。** ROPE（実質的同等領域）は「実質的に差がないとみなせる範囲」であり、データを見てから決めてはいけない。設定根拠は必ず論文に書く。ROPE を決めないと「差がない」という結論は原理的に出せず、`decision` は `different` か `undecided` にしかならない。

**`undecided` は失敗ではない。** HDI が ROPE の境界をまたぐ場合、それは「データが足りない」という正しい結論であって、無理に二値の判断に落とすべきではない。

**階層構造があるならこのモデルは使えない。** 施設内クラスタリング、反復測定、評価者ネストなどがある場合、独立2群のモデルは不確実性を過小評価する。その場合は PyMC / Stan / brms で階層モデルを直接書く必要がある。

**多群のペアワイズ比較には使わない。** ベイズだから多重比較の問題がなくなるわけではない。Kruschke 自身は補正ではなく階層モデルによる縮小（shrinkage）を推奨している。

**逐次的に見るなら停止則を事前登録する。** 「ベイズだから途中でデータを見てよい」は誤解で、HDI ベースの停止則には偏りが生じうる（Sanborn & Hills, 2014）。

**報告事項。** Kruschke (2021) の BARG (Bayesian Analysis Reporting Guidelines) に沿って、事前分布・MCMC 設定・収束診断・ROPE・感度分析を記載する。`report()` の出力はそのための素材として使える。

**再現性。** `random_seed` は既定値を持たせてあるので、同じ入力・同じライブラリ版なら結果は再現する。論文には seed とライブラリのバージョンを併記すること（`res.settings["pymc_version"]` で取得できる）。

## Kruschke の原典の例で見える食い違い

`examples/smart_drug.py` を実行すると、同じデータに対して次の結果が出る。

```
平均の差 (mu1 - mu2)
    事後平均 = 1.018  95% HDI = [0.157, 1.867]  P(>0) = 0.989
効果量 (標準化平均差)
    事後平均 = 0.644  95% HDI = [0.085, 1.251]  P(>0) = 0.989

[参考] Welch t-test: t = 1.622, p = 0.1098
```

Welch の t 検定では p = 0.11 で「有意差なし」だが、ベイズ推定では平均の差が正である事後確率は 0.99。外れ値（IQ 82, 123, 124）に引きずられて t 検定の分散推定が膨らむ一方、t 分布尤度はそれを裾として吸収するためこの差が生じる。ただし ROPE を ±1 に設定すると判定は `undecided` になり、「ゼロではなさそうだが実質的に意味のある差かは断定できない」というのがこのデータから言える正しい結論になる。

## ライセンス

MIT。モデル定義の由来である PyMC example gallery も MIT License。
