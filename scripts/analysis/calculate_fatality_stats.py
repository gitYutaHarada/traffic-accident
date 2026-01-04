import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io

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
        print(f"[WARNING] Codebook not found at {filepath}")
        return {}
    
    code_map = {}
    current_col = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 項目名の取得
            if line.startswith("【項目名】"):
                current_col = line.replace("【項目名】", "").strip()
                code_map[current_col] = {}
            
            # コード定義行の解析 (例: "1 : 晴")
            elif current_col and ":" in line and not line.startswith("■"):
                # 区切り文字が " : " とは限らないため、最初のコロンで分割
                parts = line.split(":", 1)
                if len(parts) == 2:
                    code_str = parts[0].strip()
                    desc = parts[1].strip()
                    
                    # 範囲指定の場合 (例: 1000~1499) は個別に展開せず、代表として登録するか、
                    # 解析時に数値判定を使うため、ここでは文字列キーとして保存
                    code_map[current_col][code_str] = desc

    return code_map

def get_description(col_name, value, code_map):
    """
    カラム名と値から、コードブックの説明を取得する
    """
    if col_name not in code_map:
        return str(value)
    
    val_str = str(value)
    
    # 完全一致検索
    if val_str in code_map[col_name]:
        return code_map[col_name][val_str]
    
    # 浮動小数点や文字列の "42.0" などを "42" に変換して検索
    try:
        f_val = float(value)
        if f_val.is_integer():
            int_val_str = str(int(f_val))
            if int_val_str in code_map[col_name]:
                return code_map[col_name][int_val_str]
    except (ValueError, TypeError):
        pass
            
    return str(value)

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
    print(f"   Mapped columns: {len(code_map)}")

    print("[ANALYSIS] Calculating fatality rates...")
    results = []
    
    # 分析対象外のカラム
    # 市区町村コードは数が多すぎてランキングを埋め尽くすため除外（または別途分析）
    exclude_cols = ['fatal', 'latitude', 'longitude', '地点　緯度（北緯）', '地点　経度（東経）', 'index', 'Unnamed: 0', '市区町村コード']
    
    # データセットのカラム名とコードブックの項目名のマッピングを作成
    # 例: "当事者種別（当事者A）" -> "当事者種別"
    col_mapping = {}
    for df_col in df.columns:
        for cb_col in code_map.keys():
            # CSVのカラム名にコードブックの項目名が含まれていればマッピング
            if cb_col in df_col:
                col_mapping[df_col] = cb_col
                break
    
    for col in df.columns:
        if col in exclude_cols:
            continue
            
        # ユニーク数が多すぎる数値カラムは連続値とみなしてスキップ
        lookup_col = col_mapping.get(col, col)
        
        # マッピング確認用のログ（最初の数回だけ）
        # print(f"DEBUG: {col} -> {lookup_col}")

        is_in_codebook = lookup_col in code_map
        n_unique = df[col].nunique()
        
        if not is_in_codebook and n_unique > 50 and pd.api.types.is_numeric_dtype(df[col]):
             continue

        # 集計
        temp_df = df[[col, target_col]].copy()
        temp_df[col] = temp_df[col].fillna("Missing")
        
        stats = temp_df.groupby(col)[target_col].agg(['count', 'sum']).reset_index()
        stats.columns = [col, 'total_count', 'fatal_count']
        stats['fatality_rate'] = stats['fatal_count'] / stats['total_count']
        
        for _, row in stats.iterrows():
            val = row[col]
            # マッピングされたカラム名を使って説明を取得
            desc = get_description(lookup_col, val, code_map)
            
            if lookup_col in code_map and desc != str(val):
                display_val = f"{desc} ({val})"
            else:
                display_val = str(val)

            results.append({
                'feature': col,
                'value': str(val),
                'description': display_val,
                'total_count': row['total_count'],
                'fatal_count': row['fatal_count'],
                'fatality_rate': row['fatality_rate']
            })

    # データフレーム化
    result_df = pd.DataFrame(results)
    
    # フィルタリング: サンプルサイズが少なすぎるものは信頼性が低いので除外（またはフラグ立て）
    # ここではN>=50とする
    min_samples = 50
    result_df_filtered = result_df[result_df['total_count'] >= min_samples].copy()
    
    # ソート: 死亡率の高い順
    result_df_filtered = result_df_filtered.sort_values('fatality_rate', ascending=False)
    
    # CSV出力
    out_csv = OUTPUT_DIR / "fatality_stats_ranking.csv"
    result_df_filtered.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"[OUTPUT] Saved CSV to {out_csv}")
    
    # Markdownレポート出力
    out_md = OUTPUT_DIR / "fatality_stats_ranking.md"
    
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# 特徴量条件別 死亡事故率ランキング\n\n")
        f.write(f"データセット: `{DATA_FILE.name}`\n")
        f.write(f"全データ件数: {len(df):,} 件\n")
        f.write(f"集計対象: サンプル数 {min_samples} 件以上の条件\n\n")
        
        f.write("## 死亡事故率トップ50\n\n")
        f.write("| Rank | 特徴量 | 条件 (値) | 死亡率 (%) | 死亡数 | 全件数 |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        for i, row in enumerate(result_df_filtered.head(50).itertuples(), 1):
            rate_pct = row.fatality_rate * 100
            f.write(f"| {i} | {row.feature} | {row.description} | {rate_pct:.2f}% | {row.fatal_count} | {row.total_count} |\n")
            
    print(f"[OUTPUT] Saved Report to {out_md}")

if __name__ == "__main__":
    main()
