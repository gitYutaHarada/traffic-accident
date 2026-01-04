import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import scipy.stats as stats

# WindowsでのUTF-8出力設定
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

DATA_FILE = Path("data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv")
CODEBOOK_FILE = Path("honhyo_all/details/codebook_extracted.txt")
OUTPUT_DIR = Path("results/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_codebook(filepath):
    """
    コードブックを解析し、{カラム名: {コード: 説明}} の辞書を返す関数
    """
    if not filepath.exists():
        return {}
    
    code_map = {}
    current_col = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("【項目名】"):
                current_col = line.replace("【項目名】", "").strip()
                code_map[current_col] = {}
            
            elif current_col and ":" in line and not line.startswith("■"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    code_str = parts[0].strip()
                    desc = parts[1].strip()
                    code_map[current_col][code_str] = desc
    return code_map

def cramers_v(x, y):
    """
    Cramér's V を計算する関数
    x: カテゴリ変数 (Series)
    y: ターゲット変数 (Series)
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2, p, dof, expected = stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    # 補正Cramér's V (Bias correction)
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    if min((kcorr-1), (rcorr-1)) == 0:
        return 0.0
        
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def main():
    print(f"[DATA] Loading {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {DATA_FILE}")
        return
        
    print(f"   Rows: {len(df):,}, Columns: {len(df.columns)}")
    
    target_col = 'fatal'
    if target_col not in df.columns:
        print(f"[ERROR] Target column '{target_col}' not found.")
        return

    print(f"[CODEBOOK] Parsing {CODEBOOK_FILE}...")
    code_map = parse_codebook(CODEBOOK_FILE)
    
    # カラムマッピング作成
    col_mapping = {}
    for df_col in df.columns:
        for cb_col in code_map.keys():
            if cb_col in df_col:
                col_mapping[df_col] = cb_col
                break

    print("[ANALYSIS] Calculating feature importance (Cramér's V & Range)...")
    results = []
    
    # 除外カラム
    exclude_cols = ['fatal', 'latitude', 'longitude', '地点　緯度（北緯）', '地点　経度（東経）', 'index', 'Unnamed: 0', '市区町村コード']
    
    for col in df.columns:
        if col in exclude_cols:
            continue
            
        lookup_col = col_mapping.get(col, col)
        is_in_codebook = lookup_col in code_map
        n_unique = df[col].nunique()
        
        # 連続値とみなされるものは除外（基本的にはカテゴリカルな特徴量間の比較を行いたい）
        if not is_in_codebook and n_unique > 100 and pd.api.types.is_numeric_dtype(df[col]):
             continue

        # データ準備
        temp_df = df[[col, target_col]].copy()
        temp_df[col] = temp_df[col].fillna("Missing")
        temp_df[col] = temp_df[col].astype(str) # カテゴリとして扱うために文字列化
        
        # 1. Cramér's V 計算
        try:
            v_score = cramers_v(temp_df[col], temp_df[target_col])
        except Exception as e:
            # print(f"Error calculating Cramér's V for {col}: {e}")
            v_score = 0.0
            
        # 2. 死亡率レンジ計算 (Max Rate - Min Rate)
        # サンプル数が少ないカテゴリはノイズになるため除外して計算
        stats_df = temp_df.groupby(col)[target_col].agg(['count', 'mean']).reset_index()
        valid_stats = stats_df[stats_df['count'] >= 50] # N>=50のみ
        
        if not valid_stats.empty:
            rate_range = valid_stats['mean'].max() - valid_stats['mean'].min()
            max_rate = valid_stats['mean'].max()
            min_rate = valid_stats['mean'].min()
        else:
            rate_range = 0.0
            max_rate = 0.0
            min_rate = 0.0
            
        results.append({
            'feature': col,
            'cramers_v': v_score,
            'rate_range': rate_range,
            'max_rate': max_rate,
            'min_rate': min_rate,
            'n_unique': n_unique
        })

    # 結果データフレーム
    result_df = pd.DataFrame(results)
    
    # ソート (Cramér's V 優先、次にRange)
    result_df = result_df.sort_values('cramers_v', ascending=False)
    
    # CSV出力
    out_csv = OUTPUT_DIR / "feature_importance_stats.csv"
    result_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"[OUTPUT] Saved CSV to {out_csv}")
    
    # Markdownレポート出力
    out_md = OUTPUT_DIR / "feature_importance_stats.md"
    
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# 特徴量別 死亡事故率関与度ランキング\n\n")
        f.write("## 指標の解説\n")
        f.write("- **Cramér's V (クラメールのV)**: カテゴリ変数間の相関の強さ (0〜1)。値が大きいほど、その特徴量は死亡事故の有無と強く関連しています。\n")
        f.write("- **Rate Range (死亡率の変動幅)**: その特徴量の条件間で、死亡率の最大値と最小値の差。条件によってリスクがどれだけ劇的に変わるかを示します。\n\n")
        
        f.write("## ランキング\n\n")
        f.write("| Rank | 特徴量 | Cramér's V | Rate Range (Max-Min) | Max Rate | Min Rate | カテゴリ数 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for i, row in enumerate(result_df.itertuples(), 1):
            f.write(f"| {i} | {row.feature} | {row.cramers_v:.4f} | {row.rate_range*100:.2f}% | {row.max_rate*100:.2f}% | {row.min_rate*100:.2f}% | {row.n_unique} |\n")
            
    print(f"[OUTPUT] Saved Report to {out_md}")

if __name__ == "__main__":
    main()
