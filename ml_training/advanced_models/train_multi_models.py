"""Train Random Forest, XGBoost, and Neural Network models on the honhyo dataset."""
from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - handled at runtime
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
DATA_PATH = PROJECT_ROOT / "honhyo_all" / "honhyo_all_with_datetime.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

USE_COLUMNS: List[str] = [
    "発生日時",
    "死者数",
    "天候",
    "路面状態",
    "昼夜",
    "道路形状",
    "道路線形",
    "都道府県コード",
]

NUMERIC_FEATURES = ["hour", "month", "weekday"]
CATEGORICAL_FEATURES = [
    "天候",
    "路面状態",
    "昼夜",
    "道路形状",
    "道路線形",
    "都道府県コード",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2


# データセットを読み込み、ターゲット変数と時間特徴量を整形
def load_dataset() -> pd.DataFrame:
    """指定した列のみを読み込み、ターゲット列と時間特徴量を生成した学習用DataFrameを返す。"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"入力データが見つかりません: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        usecols=USE_COLUMNS,
        parse_dates=["発生日時"],
        low_memory=False,
    )
    df = df.dropna(subset=["発生日時", "死者数"])
    df["target"] = (df["死者数"].fillna(0) > 0).astype(int)
    df["hour"] = df["発生日時"].dt.hour
    df["month"] = df["発生日時"].dt.month
    df["weekday"] = df["発生日時"].dt.weekday
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    dataset = df[features + ["target"]].copy()
    return dataset


# 数値/カテゴリ列向けの前処理パイプラインを構築
def build_preprocessor() -> ColumnTransformer:
    """数値特徴量とカテゴリ特徴量それぞれに適切な前処理を適用するColumnTransformerを構築する。"""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


# 予測結果から主要な評価指標を算出
def evaluate(y_true, y_pred, y_prob) -> Dict[str, float]:
    """予測結果から分類指標を計算し、必要に応じてROC-AUCとPR-AUCも含めて返す。"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    return metrics


# 学習に使用するモデルの設定をまとめる
def get_model_configs(scale_pos_weight: float):
    """学習に利用する各モデルの設定とインスタンスをまとめて返す。"""
    if XGBClassifier is None:
        raise ImportError(
            "XGBoost (xgboost) がインストールされていません。'pip install xgboost' で追加してください。"
        ) from XGBOOST_IMPORT_ERROR

    return [
        {
            "name": "random_forest",
            "display_name": "Random Forest",
            "estimator": RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "supports_sample_weight": True,
        },
        {
            "name": "xgboost",
            "display_name": "XGBoost",
            "estimator": XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                learning_rate=0.1,
                max_depth=6,
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.0,
                min_child_weight=1,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                tree_method="hist",
            ),
            "supports_sample_weight": True,
        },
        {
            "name": "neural_network",
            "display_name": "Neural Network (MLP)",
            "estimator": MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=200,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=RANDOM_STATE,
            ),
            "supports_sample_weight": False,
        },
    ]


# 混同行列をCSV形式で保存
def save_confusion_matrix(cm, path: Path) -> None:
    """混同行列をDataFrame化し、CSVとして保存する。"""
    cm_df = pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )
    cm_df.to_csv(path, encoding="utf-8-sig")


# モデル・指標・レポートなど成果物を出力
def save_model_artifacts(
    model_name: str,
    pipeline: Pipeline,
    metrics: Dict[str, float],
    report_text: str,
    cm,
) -> Dict[str, Path]:
    """モデル・指標・混同行列・レポートをモデル名ごとの成果物ディレクトリに書き出す。"""
    model_dir = ARTIFACT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    metrics_path = model_dir / "metrics.json"
    confusion_path = model_dir / "confusion_matrix.csv"
    report_path = model_dir / "classification_report.txt"

    joblib.dump(pipeline, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    save_confusion_matrix(cm, confusion_path)
    report_path.write_text(report_text, encoding="utf-8")

    return {
        "model": model_path,
        "metrics": metrics_path,
        "confusion": confusion_path,
        "report": report_path,
    }


# すべてのモデル結果をまとめたサマリーファイルを生成
def save_run_summary(
    results: List[Dict],
    total_rows: int,
    train_rows: int,
    test_rows: int,
) -> None:
    """全モデルのメトリクスをMarkdownとJSONにまとめて保存する。"""
    summary_path = ARTIFACT_DIR / "summary.md"
    lines = [
        "# 高度モデル 実験レポート",
        f"- 実行日時: {datetime.now().isoformat(timespec='seconds')}",
        f"- 入力データ: {DATA_PATH}",
        f"- 総行数: {total_rows:,} (学習 {train_rows:,}, テスト {test_rows:,})",
        "",
        "## モデル別評価",
        "| モデル | accuracy | precision | recall | f1 | roc_auc | pr_auc |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for result in results:
        metrics = result["metrics"]
        accuracy = metrics.get("accuracy", float("nan"))
        precision = metrics.get("precision", float("nan"))
        recall = metrics.get("recall", float("nan"))
        f1 = metrics.get("f1", float("nan"))
        roc_auc = metrics.get("roc_auc", float("nan"))
        pr_auc = metrics.get("pr_auc", float("nan"))
        lines.append(
            f"| {result['display_name']} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {roc_auc:.4f} | {pr_auc:.4f} |"
        )

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    metrics_summary_path = ARTIFACT_DIR / "metrics_summary.json"
    summary_dict = {
        result["name"]: result["metrics"] for result in results
    }
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)


# データ前処理からモデル学習・保存まで一連の処理を実行
def train_and_evaluate_models():
    """データ前処理から各モデルの学習・評価・成果物保存までを一括で実行する。"""
    print("データ読み込み中...")
    dataset = load_dataset()
    rows = len(dataset)
    X = dataset.drop(columns=["target"])
    y = dataset["target"]

    print("train/test 分割中...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("前処理パイプラインを構築中...")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("SMOTE によるオーバーサンプリングを実施中...")
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy="auto")
    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train_processed, y_train
    )

    print("クラス不均衡用の重みを計算中...")
    sample_weight = compute_sample_weight(
        class_weight="balanced", y=y_train_balanced
    )
    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    if positive == 0:
        scale_pos_weight = 1.0
    else:
        scale_pos_weight = negative / positive

    model_configs = get_model_configs(scale_pos_weight)
    results = []

    for config in model_configs:
        print("=" * 40)
        print(f"{config['display_name']} を学習中...")
        estimator = config["estimator"]

        fit_kwargs = {}
        if config.get("supports_sample_weight"):
            fit_kwargs["sample_weight"] = sample_weight

        estimator.fit(X_train_balanced, y_train_balanced, **fit_kwargs)

        print("評価中...")
        y_pred = estimator.predict(X_test_processed)
        y_prob = None
        if hasattr(estimator, "predict_proba"):
            y_prob = estimator.predict_proba(X_test_processed)[:, 1]
        metrics = evaluate(y_test, y_pred, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        report_text = classification_report(y_test, y_pred, zero_division=0)

        pipeline = Pipeline([
            ("preprocess", copy.deepcopy(preprocessor)),
            ("model", estimator),
        ])

        artifacts = save_model_artifacts(
            config["name"], pipeline, metrics, report_text, cm
        )

        results.append(
            {
                "name": config["name"],
                "display_name": config["display_name"],
                "metrics": metrics,
                "artifacts": artifacts,
            }
        )

        print(f"{config['display_name']} の指標: {json.dumps(metrics, ensure_ascii=False)}")
        print(f"成果物ディレクトリ: {artifacts['model'].parent}")

    save_run_summary(results, rows, len(X_train), len(X_test))
    print("=" * 40)
    print("全モデルの学習が完了しました。")
    print(f"成果物: {ARTIFACT_DIR}")

    return results


# スクリプトのエントリーポイント
def main():
    """エントリーポイント。全モデルの学習を実行し、主要メトリクスを表示する。"""
    results = train_and_evaluate_models()
    print("サマリー:")
    for result in results:
        name = result["display_name"]
        metrics = result["metrics"]
        metrics_text = ", ".join(
            f"{key}={value:.4f}" for key, value in metrics.items()
        )
        print(f"- {name}: {metrics_text}")


if __name__ == "__main__":
    main()
