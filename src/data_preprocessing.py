from pathlib import Path

import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
FEATURES_PATH = RAW_DATA_DIR / "features.csv"
STORES_PATH = RAW_DATA_DIR / "stores.csv"


# ============================================================
# REQUIRED COLUMNS
# ============================================================

TRAIN_REQUIRED_COLUMNS = [
    "Store",
    "Dept",
    "Date",
    "Weekly_Sales",
    "IsHoliday"
]

TEST_REQUIRED_COLUMNS = [
    "Store",
    "Dept",
    "Date",
    "IsHoliday"
]

FEATURES_REQUIRED_COLUMNS = [
    "Store",
    "Date",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "IsHoliday"
]

STORES_REQUIRED_COLUMNS = [
    "Store",
    "Type",
    "Size"
]


# ============================================================
# DATA LOADING
# ============================================================

def load_csv(path):
    """
    Load a CSV file into a pandas DataFrame.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return pd.read_csv(path)


def load_walmart_data(
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    features_path=FEATURES_PATH,
    stores_path=STORES_PATH
):
    """
    Load all raw Walmart datasets.

    Returns:
        train, test, features, stores
    """

    train = load_csv(train_path)
    test = load_csv(test_path)
    features = load_csv(features_path)
    stores = load_csv(stores_path)

    return train, test, features, stores


# ============================================================
# DATE PROCESSING
# ============================================================

def convert_date_column(df, column="Date"):
    """
    Convert a date column to pandas datetime.
    """

    df = df.copy()

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found."
        )

    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )

    return df


def convert_walmart_dates(
    train,
    test,
    features
):
    """
    Convert Date columns in Walmart datasets.
    """

    train = convert_date_column(train)
    test = convert_date_column(test)
    features = convert_date_column(features)

    return train, test, features


# ============================================================
# DATASET MERGING
# ============================================================

def merge_walmart_data(
    train,
    test,
    features,
    stores
):
    """
    Merge Walmart sales data with external features
    and store information.
    """

    train = train.merge(
        features,
        on=["Store", "Date", "IsHoliday"],
        how="left"
    )

    test = test.merge(
        features,
        on=["Store", "Date", "IsHoliday"],
        how="left"
    )

    train = train.merge(
        stores,
        on="Store",
        how="left"
    )

    test = test.merge(
        stores,
        on="Store",
        how="left"
    )

    return train, test


# ============================================================
# COLUMN VALIDATION
# ============================================================

def validate_columns(
    df,
    required_columns,
    dataset_name="dataset"
):
    """
    Validate required columns.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: "
            f"{missing_columns}"
        )

    return True


def validate_walmart_columns(
    train,
    test,
    features,
    stores
):
    """
    Validate required columns for all raw Walmart datasets.
    """

    validate_columns(
        train,
        TRAIN_REQUIRED_COLUMNS,
        "train.csv"
    )

    validate_columns(
        test,
        TEST_REQUIRED_COLUMNS,
        "test.csv"
    )

    validate_columns(
        features,
        FEATURES_REQUIRED_COLUMNS,
        "features.csv"
    )

    validate_columns(
        stores,
        STORES_REQUIRED_COLUMNS,
        "stores.csv"
    )

    return True


# ============================================================
# MISSING VALUE VALIDATION
# ============================================================

def check_missing_values(df):
    """
    Return missing-value counts by column.
    """

    return df.isnull().sum()


def get_missing_value_summary(df):
    """
    Return only columns containing missing values.
    """

    missing = check_missing_values(df)

    return missing[
        missing > 0
    ].sort_values(
        ascending=False
    )


# ============================================================
# DUPLICATE VALIDATION
# ============================================================

def check_duplicates(
    df,
    subset=None
):
    """
    Return the number of duplicate records.
    """

    return int(
        df.duplicated(
            subset=subset
        ).sum()
    )


def check_walmart_duplicates(df):
    """
    Check duplicate Store-Dept-Date combinations.
    """

    key_columns = [
        "Store",
        "Dept",
        "Date"
    ]

    return check_duplicates(
        df,
        subset=key_columns
    )


# ============================================================
# SALES VALIDATION
# ============================================================

def check_negative_sales(
    df,
    column="Weekly_Sales"
):
    """
    Return records with negative weekly sales.
    """

    if column not in df.columns:
        return pd.DataFrame()

    return df[
        df[column] < 0
    ].copy()


# ============================================================
# DATE VALIDATION
# ============================================================

def check_invalid_dates(
    df,
    column="Date"
):
    """
    Return records containing invalid dates.
    """

    if column not in df.columns:
        return pd.DataFrame()

    return df[
        df[column].isna()
    ].copy()


def get_date_range(
    df,
    column="Date"
):
    """
    Return minimum and maximum date.
    """

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found."
        )

    return {
        "min_date": df[column].min(),
        "max_date": df[column].max()
    }


# ============================================================
# DATA TYPE VALIDATION
# ============================================================

def validate_numeric_columns(
    df,
    columns
):
    """
    Check whether specified columns are numeric.
    """

    invalid_columns = []

    for column in columns:

        if column not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            invalid_columns.append(column)

    return invalid_columns


# ============================================================
# WALMART DATA VALIDATION
# ============================================================

def validate_walmart_data(
    df,
    is_train=True
):
    """
    Run the main validation checks on a Walmart dataset.
    """

    required_columns = (
        TRAIN_REQUIRED_COLUMNS
        if is_train
        else TEST_REQUIRED_COLUMNS
    )

    validate_columns(
        df,
        required_columns,
        "train.csv" if is_train else "test.csv"
    )

    result = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(
            df.isnull().sum().sum()
        ),
        "duplicate_store_dept_date": (
            check_walmart_duplicates(df)
        ),
        "invalid_dates": int(
            df["Date"].isna().sum()
        )
    }

    if is_train:
        result["negative_sales"] = int(
            (df["Weekly_Sales"] < 0).sum()
        )

    return result


# ============================================================
# DATASET SUMMARY
# ============================================================

def get_dataset_summary(df):
    """
    Return a complete basic dataset summary.
    """

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(
            df.isnull().sum().sum()
        ),
        "duplicates": int(
            df.duplicated().sum()
        )
    }

    if "Date" in df.columns:
        summary["start_date"] = df["Date"].min()
        summary["end_date"] = df["Date"].max()

    if "Store" in df.columns:
        summary["stores"] = df["Store"].nunique()

    if "Dept" in df.columns:
        summary["departments"] = df["Dept"].nunique()

    if "Weekly_Sales" in df.columns:
        summary["total_sales"] = df[
            "Weekly_Sales"
        ].sum()

    return summary
