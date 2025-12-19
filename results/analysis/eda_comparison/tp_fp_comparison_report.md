# TP vs FP 特徴量分布比較レポート

Stage 2 モデルの精度向上に向け、True Positive（正解の死亡事故）と False Positive（誤検知）の分布の違いを分析しました。

## 分析概要
- **TPデータ数**: 12290
- **FPデータ数**: 633470
- **乖離度指標**: 
  - 数値変数: KS統計量 (0.0 - 1.0)
  - カテゴリ変数: Total Variation Distance (0.0 - 1.0)

## 乖離度が大きい特徴量 Top 20
スコアが高いほど、TPとFPで分布が異なっています（＝モデルが誤検知する要因、あるいは区別に重要な特徴量）。

| Rank | Feature | Score | Type |
| :--- | :--- | :--- | :--- |
| 19 | 死者数 | 1.0000 | Categorical |
| 27 | 速度規制（指定のみ）（当事者B） | 0.2169 | Categorical |
| 3 | 一時停止規制　標識（当事者B） | 0.1968 | Categorical |
| 5 | 一時停止規制　表示（当事者B） | 0.1966 | Categorical |
| 15 | 当事者種別（当事者A） | 0.1652 | Categorical |
| 28 | 道路形状 | 0.1629 | Categorical |
| 16 | 昼夜 | 0.1571 | Categorical |
| 8 | 地形 | 0.1542 | Categorical |
| 22 | 衝突地点 | 0.1529 | Categorical |
| 25 | 車道幅員 | 0.1507 | Numerical |
| 2 | 一時停止規制　標識（当事者A） | 0.1501 | Categorical |
| 4 | 一時停止規制　表示（当事者A） | 0.1501 | Categorical |
| 30 | 都道府県コード | 0.1378 | Categorical |
| 29 | 道路線形 | 0.1297 | Categorical |
| 32 | area_id | 0.1246 | Categorical |
| 36 | hour | 0.1205 | Numerical |
| 11 | 地点コード | 0.1118 | Categorical |
| 13 | 市区町村コード | 0.1096 | Categorical |
| 23 | 警察署等コード | 0.1075 | Categorical |
| 26 | 速度規制（指定のみ）（当事者A） | 0.0989 | Categorical |

## 分布プロット
上位の特徴量の分布比較グラフ。

````carousel
![死者数](plots\死者数.png)
> **死者数** (Score: 1.0000)
<!-- slide -->
![速度規制（指定のみ）（当事者B）](plots\速度規制（指定のみ）（当事者B）.png)
> **速度規制（指定のみ）（当事者B）** (Score: 0.2169)
![一時停止規制　標識（当事者B）](plots\一時停止規制　標識（当事者B）.png)
> **一時停止規制　標識（当事者B）** (Score: 0.1968)
<!-- slide -->
![一時停止規制　表示（当事者B）](plots\一時停止規制　表示（当事者B）.png)
> **一時停止規制　表示（当事者B）** (Score: 0.1966)
<!-- slide -->
![当事者種別（当事者A）](plots\当事者種別（当事者A）.png)
> **当事者種別（当事者A）** (Score: 0.1652)
<!-- slide -->
![道路形状](plots\道路形状.png)
> **道路形状** (Score: 0.1629)
![昼夜](plots\昼夜.png)
> **昼夜** (Score: 0.1571)
<!-- slide -->
![地形](plots\地形.png)
> **地形** (Score: 0.1542)
<!-- slide -->
![衝突地点](plots\衝突地点.png)
> **衝突地点** (Score: 0.1529)
![車道幅員](plots\車道幅員.png)
> **車道幅員** (Score: 0.1507)
![一時停止規制　標識（当事者A）](plots\一時停止規制　標識（当事者A）.png)
> **一時停止規制　標識（当事者A）** (Score: 0.1501)
<!-- slide -->
![一時停止規制　表示（当事者A）](plots\一時停止規制　表示（当事者A）.png)
> **一時停止規制　表示（当事者A）** (Score: 0.1501)
<!-- slide -->
![都道府県コード](plots\都道府県コード.png)
> **都道府県コード** (Score: 0.1378)
![道路線形](plots\道路線形.png)
> **道路線形** (Score: 0.1297)
![area_id](plots\area_id.png)
> **area_id** (Score: 0.1246)
![hour](plots\hour.png)
> **hour** (Score: 0.1205)
![地点コード](plots\地点コード.png)
> **地点コード** (Score: 0.1118)
<!-- slide -->
![市区町村コード](plots\市区町村コード.png)
> **市区町村コード** (Score: 0.1096)
<!-- slide -->
![警察署等コード](plots\警察署等コード.png)
> **警察署等コード** (Score: 0.1075)
![速度規制（指定のみ）（当事者A）](plots\速度規制（指定のみ）（当事者A）.png)
> **速度規制（指定のみ）（当事者A）** (Score: 0.0989)
````
