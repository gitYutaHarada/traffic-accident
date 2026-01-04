"""
病院データを統合するスクリプト
================================
'honhyo_for_analysis_with_traffic_no_leakage.csv' に
'honhyo_for_analysis_with_hospital.csv' から病院関連データを追加します。

追加する列:
- distance_to_hospital_km: 最寄り病院までの距離 (km)
- nearest_hospital_beds: 最寄り病院の病床数
- nearest_hospital_disaster: 災害拠点病院フラグ
- hospitals_within_5km: 5km圏内の病院数

実行方法:
    python scripts/data_processing/add_hospital_features.py
"""

import pandas as pd
import os


def main():
    # ファイルパス設定
    traffic_path = "data/processed/honhyo_for_analysis_with_traffic_no_leakage.csv"
    hospital_path = "data/processed/honhyo_for_analysis_with_hospital.csv"
    output_path = "data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv"

    # 追加する病院関連列
    hospital_cols = [
        'distance_to_hospital_km',
        'nearest_hospital_beds',
        'nearest_hospital_disaster',
        'hospitals_within_5km'
    ]

    print("=" * 60)
    print("病院データ統合スクリプト")
    print("=" * 60)

    # データ読み込み
    print(f"\n📂 交通量データ読み込み中: {traffic_path}")
    df_traffic = pd.read_csv(traffic_path)
    print(f"   形状: {df_traffic.shape}")

    print(f"\n📂 病院データ読み込み中: {hospital_path}")
    df_hospital = pd.read_csv(hospital_path)
    print(f"   形状: {df_hospital.shape}")

    # 行数チェック
    if len(df_traffic) != len(df_hospital):
        print(f"\n⚠️ 警告: 行数が異なります！")
        print(f"   交通量データ: {len(df_traffic):,} 行")
        print(f"   病院データ:   {len(df_hospital):,} 行")
        print("   行数が一致しないため、処理を中止します。")
        return

    # 病院列の存在確認
    missing_cols = [col for col in hospital_cols if col not in df_hospital.columns]
    if missing_cols:
        print(f"\n❌ エラー: 以下の列が病院データに見つかりません:")
        for col in missing_cols:
            print(f"   - {col}")
        return

    # 病院データを追加
    print(f"\n✅ 病院データを追加中...")
    for col in hospital_cols:
        df_traffic[col] = df_hospital[col].values
        print(f"   + {col}")

    print(f"\n📊 統合後の形状: {df_traffic.shape}")

    # 保存
    print(f"\n💾 保存中: {output_path}")
    df_traffic.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("✅ 完了！")
    print(f"   出力ファイル: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
