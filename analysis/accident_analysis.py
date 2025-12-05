"""
交通事故データ分析スクリプト
pandasを使わず、標準ライブラリのみで分析を実行
"""
import csv
import os
from collections import defaultdict, Counter

# パス設定
data_file = r'c:\Users\socce\4nd year\software labratory\0.オープンデータを用いた交通事故原因分析\オープンデータ\honhyo\honhyo_all\honhyo_all.csv'
output_dir = r'c:\Users\socce\4nd year\software labratory\0.オープンデータを用いた交通事故原因分析\オープンデータ\analysis'

def read_csv_data(file_path, max_rows=None):
    """CSVファイルを読み込む"""
    data = []
    try:
        with open(file_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break
                data.append(row)
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def analyze_yearly_accidents(data):
    """年別の事故件数・死傷者数を分析"""
    yearly_stats = defaultdict(lambda: {'count': 0, 'dead': 0, 'injured': 0})
    
    for row in data:
        year = row.get('発生日時　　年', '')
        if year:
            yearly_stats[year]['count'] += 1
            try:
                yearly_stats[year]['dead'] += int(row.get('死者数', 0) or 0)
                yearly_stats[year]['injured'] += int(row.get('負傷者数', 0) or 0)
            except:
                pass
    
    return dict(sorted(yearly_stats.items()))

def analyze_day_of_week(data):
    """曜日別の事故発生数を分析"""
    dow_map = {'1': '日', '2': '月', '3': '火', '4': '水', '5': '木', '6': '金', '7': '土'}
    dow_count = Counter()
    
    for row in data:
        dow = row.get('曜日(発生年月日)', '')
        if dow in dow_map:
            dow_count[dow_map[dow]] += 1
    
    return dow_count

def analyze_hour_distribution(data):
    """時間帯別の事故発生数を分析"""
    hour_count = Counter()
    
    for row in data:
        hour = row.get('発生日時　　時', '')
        if hour:
            try:
                h = int(hour)
                if 0 <= h <= 23:
                    hour_count[h] += 1
            except:
                pass
    
    return hour_count

def analyze_weather(data):
    """天候別の事故発生数を分析"""
    weather_count = Counter()
    
    for row in data:
        weather = row.get('天候', '')
        if weather:
            weather_count[weather] += 1
    
    return weather_count

def analyze_day_night(data):
    """昼夜別の事故発生数・死傷者数を分析"""
    day_night_stats = defaultdict(lambda: {'count': 0, 'dead': 0, 'injured': 0})
    
    for row in data:
        dn = row.get('昼夜', '')
        if dn:
            day_night_stats[dn]['count'] += 1
            try:
                day_night_stats[dn]['dead'] += int(row.get('死者数', 0) or 0)
                day_night_stats[dn]['injured'] += int(row.get('負傷者数', 0) or 0)
            except:
                pass
    
    return dict(day_night_stats)

def save_results_to_csv(results, filename):
    """分析結果をCSVファイルに保存"""
    filepath = os.path.join(output_dir, filename)
    with open(filepath, mode='w', encoding='utf-8-sig', newline='') as f:
        if isinstance(results, dict):
            # 辞書形式の結果
            first_value = next(iter(results.values()))
            if isinstance(first_value, dict):
                # ネストされた辞書
                keys = ['key'] + list(first_value.keys())
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for key, value in results.items():
                    row = {'key': key}
                    row.update(value)
                    writer.writerow(row)
            else:
                # シンプルなキー:値
                writer = csv.writer(f)
                writer.writerow(['key', 'value'])
                for key, value in results.items():
                    writer.writerow([key, value])
        elif isinstance(results, Counter):
            # Counter形式
            writer = csv.writer(f)
            writer.writerow(['key', 'count'])
            for key, count in results.most_common():
                writer.writerow([key, count])
    
    print(f"Results saved to: {filename}")

def main():
    print("交通事故データ分析を開始します...")
    print("データを読み込んでいます（サンプリング: 最初の100,000件）...")
    
    # 大容量ファイルのため、サンプルデータで分析
    data = read_csv_data(data_file, max_rows=100000)
    
    if not data:
        print("データの読み込みに失敗しました")
        return
    
    print(f"読み込んだデータ件数: {len(data):,}")
    print("\n分析1: 年別の事故件数・死傷者数")
    yearly = analyze_yearly_accidents(data)
    save_results_to_csv(yearly, '01_yearly_accidents.csv')
    
    print("\n分析2: 曜日別の事故発生数")
    dow = analyze_day_of_week(data)
    save_results_to_csv(dow, '02_day_of_week.csv')
    
    print("\n分析3: 時間帯別の事故発生数")
    hour = analyze_hour_distribution(data)
    save_results_to_csv(hour, '03_hourly_distribution.csv')
    
    print("\n分析4: 天候別の事故発生数")
    weather = analyze_weather(data)
    save_results_to_csv(weather, '04_weather_analysis.csv')
    
    print("\n分析5: 昼夜別の事故発生数・死傷者数")
    dn = analyze_day_night(data)
    save_results_to_csv(dn, '05_day_night_analysis.csv')
    
    print("\n分析完了！結果は analysis ディレクトリに保存されました。")

if __name__ == "__main__":
    main()
