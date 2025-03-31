import pandas as pd
import numpy as np
import gzip
import os

OUTPUT_PATH = "../output/output.txt"

def discretize_data(data, n=2, bool=True, scale=0.0):
    # If scale is not zero, then we want to scale the data by scale
    # If scale is less than zero, ignore it
    if (scale > 0.0):
        data *= scale
    

    # Sort data and determine thresholds
    sorted_indices = np.argsort(data)
    thresholds = np.linspace(0, len(data), n+1, dtype=float)
    
    # Create labels based on partition
    labels = np.zeros(len(data), dtype=float)
    for i in range(n):
        labels[sorted_indices[thresholds[i]:thresholds[i+1]]] = i
    
    return labels
    
def clean_data(dataset: list, required_features: list[str], slice_size: int = -1) -> pd.DataFrame:
    # Whole dataset or a slice of it
    df = pd.DataFrame(dataset[:slice_size] if slice_size > 0 else dataset)

    print(f"Data before cleaning: {df.shape[0]} rows.")

    duplicated_rows = df[df.duplicated(keep=False)]

    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x).replace('', np.nan)

    data_null = df.isnull()
    missing_value_cols = df.columns[data_null.any()]
    missing_value_rows = df[data_null.any(axis=1)]

    # Check for missing values in required features
    missing_cols = [col for col in required_features if col not in df.columns]

    num_rows_due_to_missing_cols = 0

    rows_missing_req_features = df[df[required_features].isnull().any(axis=1)]
    num_rows_missing_req_features = len(rows_missing_req_features)
    df = df.drop(rows_missing_req_features.index)

    # df = df.drop_duplicates()
    # df = df.dropna()

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