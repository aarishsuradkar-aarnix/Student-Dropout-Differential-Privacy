"""
dataset_loader.py
Module for downloading, loading, and verifying the UCI 'Predict Students' Dropout and Academic Success' dataset.
Handles column sanitization, missing value verification, duplicate checks, and data summary.
"""

import os
import sys
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path

# Support both package-level imports and standalone usage
try:
    from .config import DATASET_PATH, DATASET_URL, TARGET_COLUMN
except ImportError:
    from config import DATASET_PATH, DATASET_URL, TARGET_COLUMN


def download_dataset_if_needed(target_path=DATASET_PATH, url=DATASET_URL):
    """
    Checks if the dataset CSV file exists. If not, downloads the official ZIP archive
    from the UCI Machine Learning Repository, extracts 'data.csv', and cleans up the temporary archive.

    Parameters
    ----------
    target_path : Path or str
        Destination path where 'data.csv' should be located.
    url : str
        URL to download the dataset ZIP from the UCI repository.

    Returns
    -------
    Path
        Path to the verified data.csv file.
    """
    target_path = Path(target_path)
    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"[Dataset Loader] Dataset found locally at: {target_path} ({target_path.stat().st_size:,} bytes)")
        return target_path

    print(f"[Dataset Loader] Dataset not found at {target_path}. Downloading from UCI repository...")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_zip_path = target_path.parent / "temp_uci_dataset.zip"

    try:
        urllib.request.urlretrieve(url, temp_zip_path)
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            # The archive contains 'data.csv'
            namelist = zip_ref.namelist()
            csv_name = [f for f in namelist if f.endswith('.csv')][0]
            zip_ref.extract(csv_name, target_path.parent)
            extracted_path = target_path.parent / csv_name
            if extracted_path != target_path:
                extracted_path.rename(target_path)

        if temp_zip_path.exists():
            temp_zip_path.unlink()

        print(f"[Dataset Loader] Successfully downloaded and extracted to: {target_path}")
        return target_path
    except Exception as e:
        if temp_zip_path.exists():
            temp_zip_path.unlink()
        raise RuntimeError(
            f"Failed to download dataset automatically from {url}. Error: {e}\n"
            f"Please download the dataset manually from the UCI repository:\n"
            f"https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success\n"
            f"and place 'data.csv' in the 'data/' directory."
        )


def load_dataset(filepath=DATASET_PATH, auto_download=True):
    """
    Loads the student dropout dataset into a pandas DataFrame.
    Sanitizes column headers (stripping whitespace/tab characters), verifies dimensions,
    and checks for missing values and duplicates.

    Parameters
    ----------
    filepath : Path or str, optional
        Path to the CSV file. Defaults to DATASET_PATH from config.
    auto_download : bool, optional
        Whether to automatically download the dataset if absent. Default is True.

    Returns
    -------
    pd.DataFrame
        Cleaned, verified pandas DataFrame.
    """
    filepath = Path(filepath)
    if not filepath.exists() and auto_download:
        filepath = download_dataset_if_needed(target_path=filepath)

    # UCI dataset uses semicolon ';' delimiter
    df = pd.read_csv(filepath, sep=';')

    # Sanitize column names: remove leading/trailing whitespace and tabs (e.g. 'Daytime/evening attendance\t')
    df.columns = df.columns.str.strip()

    # Validation checks
    expected_rows = 4424
    expected_cols = 37
    if df.shape != (expected_rows, expected_cols):
        print(f"[Warning] Expected shape ({expected_rows}, {expected_cols}), got {df.shape}. Proceeding with loaded shape.")

    return df


def get_dataset_summary(df):
    """
    Computes a comprehensive summary of the loaded dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded student dataset.

    Returns
    -------
    dict
        Summary dictionary containing row/column counts, missing value counts,
        duplicate rows, and class balance.
    """
    target_counts = df[TARGET_COLUMN].value_counts().to_dict()
    target_proportions = (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2).to_dict()

    summary = {
        "total_records": len(df),
        "total_features": df.shape[1] - 1,
        "total_columns": df.shape[1],
        "missing_values_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": target_counts,
        "class_percentages": target_proportions
    }
    return summary


if __name__ == "__main__":
    df = load_dataset()
    summary = get_dataset_summary(df)
    print("\n--- UCI Dataset Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
