import pandas as pd
import numpy as np
import gzip
import os

OUTPUT_PATH = "../output/output.txt"
    
def clean_data(dataset: list, required_features: list[str], slice_size: int = -1) -> pd.DataFrame:
    # Whole dataset or a slice of it
    df = pd.DataFrame(dataset[:slice_size] if slice_size > 0 else dataset)

    print(f"Data before cleaning: {df.shape[0]} entries.")

    duplicated_entries = df[df.duplicated(keep=False)]
    duplicated_user_entries = df[df.duplicated(subset=['username'])]

    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x).replace('', np.nan)

    data_null = df.isnull()
    missing_value_cols = df.columns[data_null.any()]
    missing_value_entries = df[data_null.any(axis=1)]

    # Check for missing values in required features
    missing_cols = [col for col in required_features if col not in df.columns]

    num_entries_due_to_missing_cols = 0

    entries_missing_req_features = df[df[required_features].isnull().any(axis=1)]
    num_entries_missing_req_features = len(entries_missing_req_features)
    df = df.drop(entries_missing_req_features.index)

    # df = df.drop_duplicates()
    # df = df.dropna()

    print(f"Data after cleaning: {df.shape[0]} entries.")

    desc = df.describe().T
    desc['dtype'] = df.dtypes

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, 'w') as f:
        f.write(f"Quality of dataset with {len(dataset)} records BEFORE cleaning\n\n")
        f.write(f"Found {len(duplicated_entries)} duplicated entries\n")
        f.write(f"Found {len(duplicated_user_entries)} duplicated user entries\n")
        f.write(f"Found {len(missing_value_entries)} entries with missing values\n")
        f.write(f"Found {len(missing_value_cols)} categories with missing values\n")

        if missing_cols:
            f.write(f"\nMissing required categories (not in DataFrame): {missing_cols}\n")
            f.write(f"entries dropped due to missing those categories: {num_entries_due_to_missing_cols}\n")

        f.write(f"entries dropped due to missing required features: {num_entries_missing_req_features}\n\n")

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