# Implementation Plan: Antidote Features for Hard FP Reduction

## Goal
Hard FP 分析で特定された「モデルの誤解（夜間・市街地への過剰反応）」を解くための「解毒剤特徴量」を実装し、Precision @ Recall 99% の向上を目指す。

## 1. Antidote Features (解毒剤特徴量)
`scripts/features/create_interaction_features.py` に以下のロジックを追加する。

### ① `is_safe_night_urban`（安全な夜の市街地）
**目的:** 「夜の市街地」というだけで危険と判定されるバイアスを補正。
- **背景:** Hard FPの75%が夜間、72%が市街地（地形=1,2）だが、実際には死亡事故に至らないケースが多数。
- **Logic:**
  ```python
  condition = (
      (df['地形'].isin([1, 2])) &       # 1:市街地(人口集中), 2:市街地(その他)
      (df['昼夜'] == 22) &              # 22:夜間（照明あり）等 ※要コード確認
      (df['当事者種別（当事者A）'] == 3)   # 3:乗用車（歩行者対車両でない典型例）
  )
  df['is_safe_night_urban'] = condition.astype(int)
  ```

### ② `midnight_activity_flag`（深夜活動係数）
**目的:** 誤検知が集中する深夜帯（平均2.61時）を特別視させる。
- **背景:** 交通量が少なくスピードが出やすい時間帯だが、単なる数値 `hour` ではその「危険な質の変化」を捉えきれていない。
- **Logic:**
  ```python
  # 深夜22時〜早朝4時フラグ
  df['midnight_activity_flag'] = df['hour'].apply(lambda x: 1 if (x >= 22 or x <= 4) else 0)
  ```

### ③ `intersection_safety`（交差点の安全性）
**目的:** 「交差点＝危険」ではなく「交差点＝減速するから（死亡事故は）起きにくい」という逆転の発想を教える。
- **背景:** TP（正解）は交差点が多いが、FPは単路が多い。
- **Logic:**
  ```python
  # 信号機あり( != 7) かつ 交差点(道路形状 != 単路コード)
  condition = (
      (df['信号機'] != 7) &          # 7:信号なし
      (df['道路形状'].isin([1, 2]))  # 交差点コード（要確認）
  )
  df['intersection_safety'] = condition.astype(int)
  ```

## 2. Verification Plan
`experiments/run_interaction_experiment.py` を再実行し、効果測定を行う。

### 比較項目
1.  **Metric:** Precision @ Recall 99% （前回 0.95% からの向上を確認）
2.  **OOF AUC:** 全体精度の変化（大幅な悪化がないか）
3.  **Feature Importance:** 新特徴量が上位（あるいは負の寄与として機能）に入っているか。
4.  **Distribution:** Hard FP の平均確率が低下しているか（「解毒」されているか）。

## Schedule
1. **Implement:** `create_interaction_features.py` に新特徴量ロジックを追加。
2. **Execute:** `run_interaction_experiment.py` を再度実行。
3. **Review:** 生成される `experiment_report.md` を確認。
