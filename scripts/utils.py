import pandas as pd
import numpy as np
import gzip
import os

OUTPUT_PATH = "../output/output.txt"

def clean_data(data_path: str, required_features: list[str], slice: int = -1) -> pd.DataFrame:
    dataset = []

    with gzip.open(data_path, 'rt', encoding='utf-8') as input_file:
        if slice == -1:
            slice = len(input_file)

        for i, line in enumerate(input_file):
            if i >= slice:
                break
            d = eval(line)
            dataset.append(d)
    df = pd.DataFrame(dataset)

    print(f"Data before cleaning: {df.shape[0]} rows.")

    duplicated_rows = df[df.duplicated(keep=False)]

    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x).replace('', np.nan)

    data_null = df.isnull()
    missing_value_cols = df.columns[data_null.any()]
    missing_value_rows = df[data_null.any(axis=1)]

    missing_cols = [col for col in required_features if col not in df.columns]
    existing_required_features = [col for col in required_features if col in df.columns]

    num_rows_due_to_missing_cols = 0
    if missing_cols:
        pass

    if existing_required_features:
        rows_missing_req_features = df[df[existing_required_features].isnull().any(axis=1)]
        num_rows_missing_req_features = len(rows_missing_req_features)
        df = df.drop(rows_missing_req_features.index)
    else:
        num_rows_missing_req_features = len(df)
        df = df.iloc[0:0]

    df = df.drop_duplicates()
    df = df.dropna()

    print(f"Data after cleaning: {df.shape[0]} rows.")

    desc = df.describe().T
    desc['dtype'] = df.dtypes

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, 'w') as f:
        f.write(f"Quality of dataset with {len(dataset)} records BEFORE cleaning\n\n")
        f.write(f"Found {len(duplicated_rows)} duplicated rows\n")
        f.write(f"Found {len(missing_value_rows)} rows with missing values\n")
        f.write(f"Found {len(missing_value_cols)} columns with missing values\n")

        if missing_cols:
            f.write(f"\nMissing required columns (not in DataFrame): {missing_cols}\n")
            f.write(f"Rows dropped due to missing those columns: {num_rows_due_to_missing_cols}\n")

        f.write(f"Rows dropped due to missing required features: {num_rows_missing_req_features}\n\n")

        f.write("Features description BEFORE cleaning:\n\n")
        original_df = pd.DataFrame(dataset)
        original_desc = original_df.describe().T
        original_desc['dtype'] = original_df.dtypes
        f.write(str(original_desc[['count','min','max','dtype']]))
        f.write("\n\n")

        f.write(f"Quality of dataset with {df.shape[0]} records AFTER cleaning\n\n")
        f.write("Features description AFTER cleaning:\n\n")
        f.write(str(desc[['count','min','max','dtype']]))
        f.write('\n')

    return df

if __name__ == "__main__":
    cleaned_df = clean_data(
        "/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz",
        required_features=["hours", "early_access", "text"],
        slice=1000
    )
