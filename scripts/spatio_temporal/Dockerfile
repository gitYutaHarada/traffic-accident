# Spatio-Temporal Stage2 Docker Image
# ====================================

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# システム依存パッケージ
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python パッケージ
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# PyTorch Geometric のインストール
RUN pip install --no-cache-dir \
    torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html \
    torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html \
    torch-geometric

# 作業ディレクトリ
WORKDIR /app

# コードをコピー
COPY . /app/scripts/spatio_temporal

# データとresultsディレクトリ（マウント用）
RUN mkdir -p /app/data /app/results

# デフォルトコマンド
CMD ["python", "scripts/spatio_temporal/run.py", "--all"]
