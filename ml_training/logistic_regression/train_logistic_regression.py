"""Train a logistic regression model on the honhyo dataset."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
DATA_PATH = PROJECT_ROOT / "honhyo_all" / "honhyo_all_with_datetime.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "logreg_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
CONFUSION_PATH = ARTIFACT_DIR / "confusion_matrix.csv"
REPORT_PATH = ARTIFACT_DIR / "experiment_report.md"

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
LOGREG_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
    "max_iter": 1000,
    "class_weight": "balanced",
}


def load_dataset() -> pd.DataFrame:
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


def build_pipeline() -> Pipeline:
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
    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(**LOGREG_PARAMS)),
        ]
    )
    return clf


def evaluate(y_true, y_pred, y_prob) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics


def save_confusion_matrix(cm) -> None:
    cm_df = pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )
    cm_df.to_csv(CONFUSION_PATH, encoding="utf-8-sig")


def save_metrics(metrics: Dict[str, float], report_text: str) -> None:
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(REPORT_PATH.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(report_text)


def save_markdown_summary(metrics: Dict[str, float], rows: int, train_rows: int, test_rows: int) -> None:
    lines = [
        "# ロジスティック回帰 実験レポート",
        f"- 実行日時: {datetime.now().isoformat(timespec='seconds')}",
        f"- 入力データ: {DATA_PATH}",
        f"- 総行数: {rows:,} (学習 {train_rows:,}, テスト {test_rows:,})",
        "",
        "## 特徴量",
        "- 数値: " + ", ".join(NUMERIC_FEATURES),
        "- カテゴリ: " + ", ".join(CATEGORICAL_FEATURES),
        "",
        "## ロジスティック回帰設定",
        "```",
    ]
    for key, value in LOGREG_PARAMS.items():
        lines.append(f"{key}: {value}")
    lines.append("```")
    lines.append("")
    lines.append("## 評価指標")
    lines.append("| 指標 | 値 |")
    lines.append("| --- | --- |")
    for key, value in metrics.items():
        lines.append(f"| {key} | {value:.4f} |")
    lines.append("")
    lines.append("## 備考")
    lines.append("- 目的変数: 死者数>0 を1、それ以外を0とした2値分類")
    lines.append("- 欠損値は前処理パイプライン内で補完")
    lines.append("- カテゴリ変数は One-Hot エンコーディング")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
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

    print("モデル構築・学習中...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("評価中...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_pred, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    print("成果物を保存中...")
    joblib.dump(pipeline, MODEL_PATH)
    save_confusion_matrix(cm)
    save_metrics(metrics, report_text)
    save_markdown_summary(metrics, rows, len(X_train), len(X_test))

    print("完了しました。")
    print(f"モデル: {MODEL_PATH}")
    print(f"指標: {METRICS_PATH}")
    print(f"レポート: {REPORT_PATH}")


if __name__ == "__main__":
    main()
