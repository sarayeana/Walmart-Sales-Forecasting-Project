"""
Data loading module for the SaaS Product Analytics & Churn Prediction project.

This module provides reusable functions for loading the four raw datasets:

    - train.csv
    - stores.csv
    - features.csv
    - test.csv

The functions return Pandas DataFrames and do not perform business
transformations. Data cleaning and validation are handled separately.
"""

from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Project Paths
# --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT /"data"/"raw"


# -------------------------------------------------------------------
# Dataset Paths
# -------------------------------------------------------------------


TRAIN_PATH = RAW_DATA_DIR / "train.csv"
STORES_PATH = RAW_DATA_DIR / "stores.csv"
FEATURES_PATH = RAW_DATA_DIR / "features.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"


# -------------------------------------------------------------------
# Generic CSV Loader
# -------------------------------------------------------------------

def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the specified CSV file does not exist.
    ValueError
        If the CSV file is empty.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            f"Dataset is empty: {file_path}"
        )

    return df


# -------------------------------------------------------------------
# Individual Dataset Loaders
# -------------------------------------------------------------------

def load_train() -> pd.DataFrame:
    """
    Load the train dataset.

    Returns
    -------
    pd.DataFrame
        Train data.
    """

    return load_csv(TRAIN_PATH)


def load_stores() -> pd.DataFrame:
    """
    Load the strores events dataset.

    Returns
    -------
    pd.DataFrame
        Stores usage event data.
    """

    return load_csv(STORES_PATH)


def load_features() -> pd.DataFrame:
    """
    Load the features dataset.

    Returns
    -------
    pd.DataFrame
        Features data.
    """

    return load_csv(FEATURES_PATH)


def load_test() -> pd.DataFrame:
    """
    Load the test dataset.

    Returns
    -------
    pd.DataFrame
        Test data.
    """

    return load_csv(TEST_PATH)


# -------------------------------------------------------------------
# Load All Datasets
# -------------------------------------------------------------------

def load_all_data() -> dict[str, pd.DataFrame]:
    """
    Load all project datasets.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing all four datasets.

    Example
    -------
    >>> data = load_all_data()
    >>> train = data["train"]
    >>> stores = data["stores"]
    """

    return {
        "train": load_train(),
        "stores": load_stores(),
        "features": load_features(),
        "test": load_test(),
    }


# -------------------------------------------------------------------
# Dataset Summary
# -------------------------------------------------------------------

def get_dataset_summary(
    data: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Create a high-level summary of loaded datasets.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Dictionary returned by load_all_data().

    Returns
    -------
    pd.DataFrame
        Summary containing dataset names, rows, columns,
        and missing-value counts.
    """

    summary = []

    for name, df in data.items():

        summary.append(
            {
                "dataset": name,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "missing_values": int(df.isna().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
            }
        )

    return pd.DataFrame(summary)


# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("Walmart sales forcasting project")
    print("Data Loading Test")
    print("=" * 70)

    datasets = load_all_data()

    summary = get_dataset_summary(datasets)

    print("\nDataset Summary:")
    print(summary.to_string(index=False))

    print("\nData loading completed successfully.")