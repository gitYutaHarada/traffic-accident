from pathlib import Path

CODEBOOK_FILE = Path("honhyo_all/details/codebook_extracted.txt")

def parse_codebook(filepath):
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

cm = parse_codebook(CODEBOOK_FILE)
target = "当事者種別"
print(f"Checking column: {target}")
if target in cm:
    print(f"Keys found: {list(cm[target].keys())[:10]}")
    print(f"Value for '42': {cm[target].get('42')}")
else:
    print("Column not found in codebook.")
