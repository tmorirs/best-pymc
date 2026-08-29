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
- ライセンス・再配布の扱い: **下記「9. データのライセンスと再配布について」を必ず参照**（一言で「GPL」とは言えません）

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

## 8. サンプルスクリプト（BEST で見直す）

同ディレクトリの `help_maxdrinks_by_sex.py` は、このデータを外れ値にロバストな
ベイズ推定（Kruschke の BEST）で見直すサンプルです。CSV の他の列も含め、
「t 検定ではぎりぎり非有意（p が 0.05 の少し上）だが、順位検定では明確に有意」という
主対象と同じ構図になる比較を、**コマンドラインオプションで切り替えて**実行できます。

```bash
# 既定: 性別 × 1日最大飲酒量（元からの主対象）
python examples/help_maxdrinks_by_sex.py

# 用意した別の境界例を選ぶ
python examples/help_maxdrinks_by_sex.py --case homeless-cesd

# 比較例の一覧（想定仮説と目安の p 値つき）を表示
python examples/help_maxdrinks_by_sex.py --list-cases

# 列を自由に組み合わせる（3水準の substance は --levels 必須）
python examples/help_maxdrinks_by_sex.py --group substance --outcome cesd --levels alcohol heroin

# ROPE や閾値を個別に上書き
python examples/help_maxdrinks_by_sex.py --case sex-avgdrinks --rope -0.5 0.5 --threshold 2

# ヘルプ
python examples/help_maxdrinks_by_sex.py --help
```

### 用意した比較例（`--case`）

いずれも Welch の t 検定と Mann–Whitney U を手元計算した目安の値です。

| `--case` | 比較（群 × 変数） | 想定仮説 | Welch p | MWU p |
|---|---|---|---|---|
| `sex-maxdrinks`（既定） | sex: male vs female × `max_drinks_per_day` | 男性のほうが1日最大飲酒量が多い | 0.051 | 0.004 |
| `sex-avgdrinks` | sex: male vs female × `avg_drinks_per_day` | 男性のほうが1日平均飲酒量が多い | 0.067 | 0.007 |
| `substance-avgdrinks` | substance: cocaine vs heroin × `avg_drinks_per_day` | コカイン群のほうが平均飲酒量が多い | 0.054 | 0.003 |
| `homeless-cesd` | homeless: homeless vs housed × `cesd` | ホームレス群のほうが抑うつ得点が高い | 0.064 | 0.026 |
| `sex-cesd`（対比用） | sex: female vs male × `cesd` | 女性のほうが抑うつ得点が高い | 0.0003 | — |

最初の4つは「t 検定だけが非有意側に落ちる」境界例です。最後の `sex-cesd` は逆に
明確に有意で、「同じ集団・同じ n でも、変数によって検出できたりできなかったりする」
という対比として並べてあります。

### 主なオプション

| オプション | 内容 |
|---|---|
| `--case NAME` | 事前定義の比較例を選ぶ（既定 `sex-maxdrinks`）。`--list-cases` で説明を表示 |
| `--group {sex,substance,homeless}` | 群分けに使う列（`--case` を上書き） |
| `--outcome {max_drinks_per_day,avg_drinks_per_day,age,cesd}` | 比較する量的変数（`--case` を上書き） |
| `--levels G1 G2` | 比較する2水準。差は G1 − G2 の向き（3水準の `substance` では必須） |
| `--rope LOW HIGH` | ROPE（実質的同等領域）の区間 |
| `--threshold X` | `P(平均差 > X | データ)` を問い合わせる「実質的に意味のある差」 |
| `--seed N` / `--prefix STR` / `--no-plot` | 乱数シード / 出力 PNG の接頭辞 / 図の保存を省略 |

出力される `report()` には比較用に **Welch の t 検定・Cohen's d・検定力** も併記されるため、
「頻度論の t 検定ではどう見えるか」を BEST の事後確率と同じ画面で確認できます。
たとえば `--case homeless-cesd` では Welch p = 0.064（非有意）に対し、BEST では
平均差が正である事後確率 P(> 0) ≈ 0.97 が得られます。

## 9. データのライセンスと再配布について

この節は、上の「1. 何のデータか」で **「GPL」とだけ書いていたのが不正確だった** ため、
経緯を含めて改めて整理するものです。結論を先に書くと、**このCSVの元データに
「LICENSE ファイル」が単体で付いていたわけではなく、GPL は R パッケージという配布物
全体に付いていたライセンス**です。データの行列そのものの再配布可否は、下記の
複数の情報を突き合わせて判断しています。

### 由来のたどり方（3段階）

1. **原典（データの生成元）**
   HELP study そのもの。原著は Samet JH ら (Addiction. 2003; 98(4): 509–516)。
   データの著作権・帰属はまず研究チーム側にあります。

2. **R パッケージ `mosaicData`（配布物）**
   HELP study のベースライン抽出が `HELPrct` として収録されているのがこのパッケージです。
   CRAN 上の `mosaicData` の **License 欄は `GPL-2 | GPL-3`（= `GPL (≥ 2)`）**。
   作者は Randall Pruim, Daniel Kaplan, Nicholas Horton（保守は Randall Pruim）。
   ここで重要なのは、**この GPL は「R パッケージ（コード＋ドキュメント＋同梱データを
   まとめた配布物）」に対して宣言されたライセンスであって、`HELPrct` の数値表単体に
   個別の LICENSE ファイルが添付されていたわけではない** という点です。GPL は本来
   ソフトウェアのためのライセンスであり、「観測値の行列」がそれにどこまで従属するかは
   自明ではありません。

3. **Rdatasets（実際の取得元）**
   本CSVを取ってきたのは Vincent Arel-Bundock 氏の
   [Rdatasets](https://github.com/vincentarelbundock/Rdatasets)
   （`csv/mosaicData/HELPrct.csv`）です。同リポジトリの README には、
   ライセンスについて次のように書かれています（原文ママ）。

   > The code in this repository is licensed under GPL-3.
   >
   > the R documentation which I copied to the Rdatasets html folder is licensed under GPL
   >
   > **My understanding is that these datasets are free to re-distribute.**
   >
   > I made a good faith effort to determine the license under which the actual data
   > (i.e. rows/columns of numbers) were distributed, but I was unable to find a
   > definitive answer.

   つまり Rdatasets 側でも、**GPL が掛かっているのはリポジトリの「コード」と
   「Rドキュメント」であって、データ行列そのものの厳密なライセンスは特定できなかった**、
   そのうえで作者は **「これらのデータセットは自由に再配布できると理解している」** と
   明言しています。加えて README には、権利者から連絡があればデータを速やかに削除し
   履歴からも消す、という趣旨の対応方針も書かれています。

### この配布物での扱い

- 上記を踏まえ、本リポジトリでは `help_maxdrinks_by_sex.csv` を
  **「Rdatasets 経由で自由に再配布できると理解されているデータ」** として同梱しています。
  「元データが GPL である」という単純化した表現は、GPL が本来はパッケージ配布物に対する
  ものである以上、誤解を招くため使いません。
- 本リポジトリ直下の `LICENSE`（MIT）は **`best-pymc` というソフトウェア本体** に対する
  ものであり、この HELP study 由来のデータに遡及して適用されるものではありません。
  データの帰属・原典表示は本節および「1. 何のデータか」に記載したとおりです。
- 学術利用の際は、データの出典として **原典（Samet et al., 2003）と、収録元
  `mosaicData` パッケージ** を併記してください。取得の便宜として Rdatasets を
  経由した旨も書いておくと再現しやすくなります。
- もし権利関係について懸念がある場合は、Rdatasets の方針（権利者の申し出により削除）に
  ならい、本リポジトリからも当該データを削除します。

> 注意: 本節はライセンスの状況を可能な範囲で正確に記述したものであり、法的助言では
> ありません。厳密な可否判断が必要な用途では、原典および `mosaicData` の権利者に
> 直接確認してください。
