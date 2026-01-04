"""
可視化モジュール
===============
PR曲線、ROC曲線、Foliumヒートマップ
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'Hiragino Sans', 'sans-serif']


def plot_pr_curve(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    title: str = "Precision-Recall Curve",
):
    """
    PR曲線のプロット
    
    Args:
        results: {model_name: (y_true, y_pred)}
        output_path: 出力パス
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for (model_name, (y_true, y_pred)), color in zip(results.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        
        ax.plot(recall, precision, color=color, lw=2,
                label=f'{model_name} (AUC = {pr_auc:.4f})')
    
    # ベースライン（ランダム分類器）
    baseline = y_true.sum() / len(y_true) if len(y_true) > 0 else 0
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=1, 
               label=f'Baseline = {baseline:.4f}')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 PR曲線保存: {output_path}")


def plot_roc_curve(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    title: str = "ROC Curve",
):
    """
    ROC曲線のプロット
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for (model_name, (y_true, y_pred)), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # 対角線（ランダム分類器）
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 ROC曲線保存: {output_path}")


def plot_calibration_curve(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    n_bins: int = 10,
    title: str = "Calibration Curve",
):
    """
    キャリブレーション曲線のプロット
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for (model_name, (y_true, y_pred)), color in zip(results.items(), colors):
        # ビンごとの実際の正例率と予測確率の平均
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_means = []
        bin_true = []
        
        for i in range(n_bins):
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_means.append(y_pred[mask].mean())
                bin_true.append(y_true[mask].mean())
        
        ax.plot(bin_means, bin_true, 's-', color=color, lw=2, 
                markersize=8, label=model_name)
    
    # 完璧なキャリブレーション
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect Calibration')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 キャリブレーション曲線保存: {output_path}")


def plot_feature_importance(
    importance_dict: Dict[str, float],
    output_path: str,
    top_n: int = 20,
    title: str = "Feature Importance",
):
    """
    特徴量重要度のプロット
    """
    # ソートしてTop-Nを取得
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importances = zip(*sorted_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, align='center', color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 特徴量重要度保存: {output_path}")


def create_heatmap(
    df: pd.DataFrame,
    output_path: str,
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    pred_col: str = 'prediction',
    title: str = "交通事故リスク予測ヒートマップ",
):
    """
    Foliumを使用した予測確率ヒートマップ生成
    
    Args:
        df: lat, lon, prediction 列を含むデータフレーム
        output_path: 出力HTMLパス
    """
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        print("Warning: folium not installed. Skipping heatmap creation.")
        return
    
    # 日本の中心座標
    center_lat = df[lat_col].mean()
    center_lon = df[lon_col].mean()
    
    # 地図作成
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
    
    # ヒートマップデータ
    heat_data = [[row[lat_col], row[lon_col], row[pred_col]] 
                 for _, row in df.iterrows() if not pd.isna(row[pred_col])]
    
    # ヒートマップ追加
    HeatMap(
        heat_data,
        radius=15,
        blur=10,
        max_zoom=1,
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}
    ).add_to(m)
    
    # タイトル追加
    title_html = f'''
    <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # 保存
    m.save(output_path)
    
    print(f"🗺️ ヒートマップ保存: {output_path}")


def create_top_n_map(
    df: pd.DataFrame,
    output_path: str,
    n: int = 100,
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    pred_col: str = 'prediction',
    title: str = "Top-N 高リスク地点",
):
    """
    Top-N高リスク地点の地図表示
    """
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        print("Warning: folium not installed. Skipping map creation.")
        return
    
    # Top-N取得
    top_n_df = df.nlargest(n, pred_col)
    
    # 地図作成
    center_lat = top_n_df[lat_col].mean()
    center_lon = top_n_df[lon_col].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
    
    # マーカークラスター
    marker_cluster = MarkerCluster().add_to(m)
    
    for idx, row in top_n_df.iterrows():
        popup_text = f"""
        <b>リスクスコア:</b> {row[pred_col]:.4f}<br>
        <b>緯度:</b> {row[lat_col]:.4f}<br>
        <b>経度:</b> {row[lon_col]:.4f}
        """
        
        # リスクレベルに応じた色
        if row[pred_col] >= 0.8:
            color = 'red'
        elif row[pred_col] >= 0.5:
            color = 'orange'
        else:
            color = 'green'
        
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=8,
            popup=folium.Popup(popup_text, max_width=300),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
        ).add_to(marker_cluster)
    
    # タイトル
    title_html = f'''
    <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    m.save(output_path)
    
    print(f"🗺️ Top-N地図保存: {output_path}")


def create_top_n_list(
    df: pd.DataFrame,
    output_path: str,
    n: int = 100,
    pred_col: str = 'prediction',
    additional_cols: List[str] = None,
):
    """
    Top-N高リスク地点リストのCSV/HTML出力
    """
    # Top-N取得
    top_n_df = df.nlargest(n, pred_col).copy()
    top_n_df['rank'] = range(1, len(top_n_df) + 1)
    
    # 列の選択
    cols = ['rank', pred_col]
    if 'lat' in df.columns:
        cols.append('lat')
    if 'lon' in df.columns:
        cols.append('lon')
    if 'geohash' in df.columns:
        cols.append('geohash')
    if additional_cols:
        cols.extend([c for c in additional_cols if c in df.columns])
    
    top_n_df = top_n_df[cols]
    
    # CSV保存
    csv_path = output_path.replace('.html', '.csv')
    top_n_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # HTML保存
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Top-{n} 高リスク地点</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #ddd; }}
        </style>
    </head>
    <body>
        <h2>Top-{n} 高リスク地点リスト</h2>
        {top_n_df.to_html(index=False)}
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📋 Top-Nリスト保存: {output_path}")


class Visualizer:
    """可視化クラス"""
    
    def __init__(self, output_dir: str = "results/spatio_temporal"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_all_curves(
        self,
        results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ):
        """全ての曲線をプロット"""
        plot_pr_curve(results, str(self.output_dir / "pr_curve.png"))
        plot_roc_curve(results, str(self.output_dir / "roc_curve.png"))
        plot_calibration_curve(results, str(self.output_dir / "calibration_curve.png"))
    
    def create_prediction_maps(
        self,
        df: pd.DataFrame,
        pred_col: str = 'prediction',
    ):
        """予測結果の地図を作成"""
        create_heatmap(df, str(self.output_dir / "heatmap.html"), pred_col=pred_col)
        create_top_n_map(df, str(self.output_dir / "top_n_map.html"), pred_col=pred_col)
        create_top_n_list(df, str(self.output_dir / "top_n_list.html"), pred_col=pred_col)
