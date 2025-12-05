# ロジスティック回帰 実験レポート
- 実行日時: 2025-11-20T14:48:06
- 入力データ: C:\Users\socce\4nd year\software labratory\0.オープンデータを用いた交通事故原因分析\オープンデータ\honhyo_all\honhyo_all_with_datetime.csv
- 総行数: 1,895,275 (学習 1,516,220, テスト 379,055)

## 特徴量
- 数値: hour, month, weekday
- カテゴリ: 天候, 路面状態, 昼夜, 道路形状, 道路線形, 都道府県コード

## ロジスティック回帰設定
```
penalty: l2
C: 1.0
solver: lbfgs
random_state: 42
max_iter: 1000
class_weight: balanced
```

## 評価指標
| 指標 | 値 |
| --- | --- |
| accuracy | 0.7471 |
| precision | 0.0214 |
| recall | 0.6351 |
| f1 | 0.0413 |
| roc_auc | 0.7507 |

## 備考
- 目的変数: 死者数>0 を1、それ以外を0とした2値分類
- 欠損値は前処理パイプライン内で補完
- カテゴリ変数は One-Hot エンコーディング