# 「差があるはずなのに p = 0.051」データセット

## 1. 何のデータか

米国ボストンの解毒（detoxification）病棟に入院したアルコール・ヘロイン・コカイン依存の成人を対象とした
ランダム化比較試験 **HELP study (Health Evaluation and Linkage to Primary Care)** の、
**ベースライン時点**の面接データです。追跡は2年間・6か月ごとですが、本 CSV はベースラインのみを使っています。

- 元データ: R パッケージ `mosaicData` の `HELPrct`（453名、欠測なしのサブセット）
- 取得元: Rdatasets
  `https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/mosaicData/HELPrct.csv`
- 原典: Samet JH, Larson MJ, Horton NJ, Doyle K, Winter M, Saitz R.
  *Linking alcohol and drug-dependent adults to primary medical care: a randomized controlled trial
  of a multi-disciplinary health intervention in a detoxification unit.*
  Addiction. 2003; 98(4): 509–516.
- パッケージのライセンス: GPL（`mosaicData`）

## 2. ファイル

`help_maxdrinks_by_sex.csv` — 453行 × 8列。1行1名。

| 列 | 内容 |
|---|---|
| `id` | 被験者ID |
| `sex` | `male` (n=346) / `female` (n=107) |
| `max_drinks_per_day` | **過去30日間の1日あたり最大飲酒量（standard drink 単位）**。元の `i2` |
| `avg_drinks_per_day` | 過去30日間の1日あたり平均飲酒量。元の `i1` |
| `age` | 年齢（歳） |
| `substance` | 主たる乱用物質: `alcohol` / `cocaine` / `heroin` |
| `homeless` | `homeless` / `housed`（過去6か月で1泊以上の路上・シェルター滞在の有無） |
| `cesd` | CES-D 抑うつ尺度得点（0–60、高いほど症状が重い） |

主対象は **`sex` × `max_drinks_per_day`** です。他の列は対比・層別用のおまけです。

## 3. 事前の予想

「アルコール依存の臨床集団において、男性のほうが1日最大飲酒量は多い」——
体格差・代謝差・疫学的知見からみて、まず差が出ると予想されるところです。
実際、記述統計を見るかぎり予想どおりに見えます。

## 4. 結果（手元で計算した値）

| | n | 平均 | SD | 中央値 | 歪度 | 0杯の割合 |
|---|---|---|---|---|---|---|
| male | 346 | 25.95 | 28.22 | 19.0 | 2.28 | 12.1% |
| female | 107 | 20.02 | 26.99 | 13.0 | 2.52 | 24.3% |

**Welch の t 検定**

```
t = 1.9646,  df = 183.38,  p = 0.0510
平均差 = 5.93 杯,  95% CI [-0.025, 11.884]
```

- Student の t 検定: t = 1.9187, p = 0.0557（Levene 検定 p = 0.555 で等分散は棄却されず、両者ほぼ同じ）
- Cohen's d = 0.212
- 並べ替え検定（平均差、50,000回）: p = 0.055
- **Mann–Whitney U: p = 0.0038**
- **log(1+x) 変換後の Welch: p = 0.0021**

## 5. どこが「おや」か

1. **95%信頼区間の下限が −0.025**。ほぼぴったり 0 をかすめて非有意側に落ちています。
   1〜2名データが違えば結論が反転する水準です。

2. **平均差 5.9杯は実質的には小さくない**（女性群の平均の約30%）。にもかかわらず
   「有意差なし」という一行に要約されてしまう。効果量が小さいから検出できないのではなく、
   右に強く歪んだ分布（歪度 2.3〜2.5、最大184杯）が平均の標準誤差を膨らませているのが主因です。

3. **検定手法を変えると結論が変わる**。順位ベース（Mann–Whitney, p = 0.0038）や
   対数変換後（p = 0.0021）では明確に有意です。中央値も 19 対 13 と離れています。
   一方、並べ替え検定でも p = 0.055 なので、「正規性の仮定が悪い」というより
   **「平均という要約統計量が、この分布ではロバストでない」** という話です。

4. **等分散性は問題になっていない**。SD は 28.2 対 27.0 でほぼ同じ、Levene p = 0.555。
   つまりここでの Welch vs Student の違いは本質ではなく、
   「Welch にしておけば安心」では解決しない種類の問題です。

## 6. 再現コード

### Python

```python
import pandas as pd
from scipy import stats

d = pd.read_csv("help_maxdrinks_by_sex.csv")
m = d.loc[d.sex == "male",   "max_drinks_per_day"]
f = d.loc[d.sex == "female", "max_drinks_per_day"]

print(stats.ttest_ind(m, f, equal_var=False))   # Welch
print(stats.mannwhitneyu(m, f))
```

### R

```r
d <- read.csv("help_maxdrinks_by_sex.csv")
t.test(max_drinks_per_day ~ sex, data = d)              # Welch（既定）
t.test(max_drinks_per_day ~ sex, data = d, var.equal = TRUE)
wilcox.test(max_drinks_per_day ~ sex, data = d)

# 元パッケージから直接
# install.packages("mosaicData"); library(mosaicData); t.test(i2 ~ sex, data = HELPrct)
```

## 7. 注意点

- ランダム化試験のデータですが、**性別は割付変数ではありません**。これはベースラインの
  観察的な群間比較であり、因果的な解釈はできません（年齢・主たる乱用物質・住居状況などが交絡します）。
- 群サイズが 346 対 107 と不均衡です。この不均衡自体も検出力を下げています
  （合計 453 名でも、有効サンプルサイズは均等配分時の 4·346·107/453² ≒ 0.72 倍相当）。
- 飲酒量は自己報告であり、報告バイアスが性別で非対称である可能性があります。
  女性群で「0杯」が 24.3% と男性の倍あることも、実態か報告の差か、この表からは区別できません。
- 同じファイルの `cesd` を `sex` で比べると Welch p = 0.0003 と明確に有意（女性のほうが高得点）です。
  「同じ集団・同じ n でも、変数によって検出できたりできなかったりする」対比として使えます。
  `homeless` × `cesd` なら Welch p = 0.064 で、これも境界例です。
