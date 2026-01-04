# Spatio-Temporal Stage2 Makefile
# ================================

.PHONY: all preprocess train evaluate visualize optuna clean help

# デフォルト設定
DATA_PATH ?= data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv
OUTPUT_DIR ?= results/spatio_temporal
MODELS ?= mlp,knn_gnn
EPOCHS ?= 100
K ?= 8
N_TRIALS ?= 50

# Python
PYTHON ?= python

help:
	@echo "Spatio-Temporal Stage2 Pipeline"
	@echo "================================"
	@echo ""
	@echo "使用方法:"
	@echo "  make all          - 全工程実行"
	@echo "  make preprocess   - 前処理のみ"
	@echo "  make train        - 学習のみ"
	@echo "  make evaluate     - 評価と可視化"
	@echo "  make optuna       - Optuna探索"
	@echo "  make clean        - チェックポイント削除"
	@echo ""
	@echo "オプション:"
	@echo "  DATA_PATH=...     - データファイルパス"
	@echo "  OUTPUT_DIR=...    - 出力ディレクトリ"
	@echo "  MODELS=...        - モデル (mlp,knn_gnn)"
	@echo "  EPOCHS=...        - エポック数"
	@echo "  K=...             - kNN k値"

all:
	$(PYTHON) run.py --all \
		--data-path $(DATA_PATH) \
		--output-dir $(OUTPUT_DIR) \
		--models $(MODELS) \
		--epochs $(EPOCHS) \
		--k $(K)

preprocess:
	$(PYTHON) preprocess_spatio_temporal.py \
		--data-path $(DATA_PATH) \
		--output-dir data/spatio_temporal

train:
	$(PYTHON) train_spatio_temporal.py \
		--output-dir $(OUTPUT_DIR) \
		--model all \
		--epochs $(EPOCHS)

evaluate:
	$(PYTHON) evaluate.py \
		--output-dir $(OUTPUT_DIR)

visualize:
	$(PYTHON) visualize.py \
		--output-dir $(OUTPUT_DIR)

optuna:
	$(PYTHON) optuna_search.py \
		--output-dir $(OUTPUT_DIR)/optuna \
		--n-trials $(N_TRIALS)

clean:
	rm -rf $(OUTPUT_DIR)/checkpoints
	rm -rf $(OUTPUT_DIR)/logs
	@echo "チェックポイントとログを削除しました"

# Docker
docker-build:
	docker build -t spatio-temporal-stage2 .

docker-run:
	docker run --gpus all -v $(PWD)/data:/app/data -v $(PWD)/results:/app/results \
		spatio-temporal-stage2 python run.py --all
