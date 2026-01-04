
# 閾値最適化 & 戦略分析レポート

## ステップ1: 閾値の最適化 (Max F1 Score)
**バランス重視**: 精度と検知率のバランスが最も良いポイント
- **Threshold**: 0.3085
- **F1 Score**: 0.2577
- Precision: 0.2407
- Recall: 0.2773

## ステップ2: 見逃しを減らしたい (High Recall Strategy)
**警察パトロール重点箇所**: 「怪しい場所は全部検知」
- **Target Recall**: ~98%
- **Threshold**: 0.0100
- **Precision**: 0.0142
- Recall: 0.9801
- F1 Score: 0.0280
*解説: Precisionが低い（1.42%）ため、空振りが多いが、危険な場所の98%を網羅できる設定。*

## ステップ3: 確実な場所だけ知りたい (High Precision Strategy)
**予算限定・集中対策**: 「絶対に事故が起きる場所だけ」
- **Target Precision**: ~80% (または最大)
- **Threshold**: 0.5489
- **Precision**: 0.8011
- Recall: 0.0228
- F1 Score: 0.0444
*解説: 検知数（Recall）は低い（2.28%）が、警報が出た場所の80.11%で実際に事故が発生している高確度設定。*
