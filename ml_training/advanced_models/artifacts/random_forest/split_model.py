"""
Random Forestモデルファイルの分割・結合スクリプト

GitHubのLFS制限（2GB）を超えるmodel.joblibを分割して管理するためのスクリプト。
"""
import os
from pathlib import Path

# 分割サイズ: 1.5GB (1.5 * 1024 * 1024 * 1024 bytes)
CHUNK_SIZE = 1_610_612_736  # 1.5GB in bytes
MODEL_FILE = "model.joblib"


def split_model(model_path: str = MODEL_FILE) -> list[str]:
    """
    model.joblibを1.5GB単位で分割する
    
    Args:
        model_path: 分割するモデルファイルのパス
        
    Returns:
        作成された分割ファイルのパスリスト
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} が見つかりません")
    
    parts = []
    part_num = 0
    
    with open(model_path, 'rb') as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            
            part_path = f"{model_path}.part{part_num}"
            with open(part_path, 'wb') as part_file:
                part_file.write(chunk)
            
            parts.append(part_path)
            print(f"作成: {part_path} ({len(chunk) / (1024**3):.2f} GB)")
            part_num += 1
    
    print(f"\n分割完了: {len(parts)} 個のパーツファイルを作成しました")
    return parts


def combine_model(output_path: str = MODEL_FILE) -> None:
    """
    分割されたパーツファイルを結合してmodel.joblibを復元する
    
    Args:
        output_path: 出力するモデルファイルのパス
    """
    output_path = Path(output_path)
    
    # パーツファイルを探す
    parts = sorted(Path('.').glob(f"{MODEL_FILE}.part*"))
    if not parts:
        raise FileNotFoundError("パーツファイルが見つかりません")
    
    print(f"結合するパーツ: {[str(p) for p in parts]}")
    
    with open(output_path, 'wb') as out_file:
        for part in parts:
            print(f"結合中: {part}")
            with open(part, 'rb') as part_file:
                out_file.write(part_file.read())
    
    print(f"\n結合完了: {output_path} ({output_path.stat().st_size / (1024**3):.2f} GB)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  分割: python split_model.py split")
        print("  結合: python split_model.py combine")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "split":
        split_model()
    elif command == "combine":
        combine_model()
    else:
        print(f"不明なコマンド: {command}")
        sys.exit(1)
