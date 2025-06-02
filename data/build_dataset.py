import pandas as pd
import os

# ========= STEP 1: Gộp các file benign/malicious ========= #

def load_and_label(files_dict):
    dfs = []
    for file, label in files_dict.items():
        df = pd.read_csv(file)
        df['label'] = label
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined

# ========= STEP 2: Gộp các cột trùng nhau ========= #

def merge_duplicate_columns(df):
    df.columns = df.columns.str.strip()
    base_names = {}
    for col in df.columns:
        base = col.split('.')[0]
        base_names.setdefault(base, []).append(col)

    for base, cols in base_names.items():
        if len(cols) > 1:
            print(f"Gộp các cột: {cols} → {base}")
            merged_col = df[cols].bfill(axis=1).iloc[:, 0]
            df.drop(columns=cols, inplace=True)
            df[base] = merged_col
    return df

# ========= Main logic ========= #

def build_dataset(file_map, output_csv='domain_dataset_cleaned.csv'):
    print("Đang gộp các file...")
    df = load_and_label(file_map)

    print("Đang xử lý các cột bị trùng...")
    df = merge_duplicate_columns(df)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"Dataset đã lưu tại: {output_csv}")


if __name__ == '__main__':
    file_map = {
        "mdn_dts/1000mdn_benign_extract.csv": 0,
        "mdn_dts/1000mdn_benign_extract_T.csv": 0,
        "mdn_dts/1000mdn_malicious_extract.csv": 1,
        "mdn_dts/1000mdn_malicious_extract_T.csv": 1,
        "mdn_dts/2000mdn_benign_extract_H.csv": 0,
        "mdn_dts/2000mdn_malicious_extract_H.csv": 1,
    }

    build_dataset(file_map)
