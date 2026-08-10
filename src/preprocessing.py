import pandas as pd


# ============================================================
# WALMART SALES FORECASTING
# DATA PREPROCESSING
# ============================================================


# ============================================================
# 1. LOAD DATASETS
# ============================================================

def load_walmart_data(
    train_path="data/raw/train.csv",
    test_path="data/raw/test.csv",
    features_path="data/raw/features.csv",
    stores_path="data/raw/stores.csv"
):
    """
    Load all four Walmart datasets.

    Returns:
        train
        test
        features
        stores
    """

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    features = pd.read_csv(features_path)
    stores = pd.read_csv(stores_path)

    return train, test, features, stores


# ============================================================
# 2. CONVERT DATE COLUMNS
# ============================================================

def convert_dates(
    train,
    test,
    features
):
    """
    Convert Date columns to datetime format.
    """

    train = train.copy()
    test = test.copy()
    features = features.copy()

    train["Date"] = pd.to_datetime(
        train["Date"]
    )

    test["Date"] = pd.to_datetime(
        test["Date"]
    )

    features["Date"] = pd.to_datetime(
        features["Date"]
    )

    return train, test, features


# ============================================================
# 3. TRAIN DATA VALIDATION
# ============================================================

def validate_train_data(train):
    """
    Validate Walmart train dataset.
    """

    required_columns = [
        "Store",
        "Dept",
        "Date",
        "Weekly_Sales",
        "IsHoliday"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in train.columns
    ]

    return missing_columns


# ============================================================
# 4. TEST DATA VALIDATION
# ============================================================

def validate_test_data(test):
    """
    Validate Walmart test dataset.
    """

    required_columns = [
        "Store",
        "Dept",
        "Date",
        "IsHoliday"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in test.columns
    ]

    return missing_columns


# ============================================================
# 5. FEATURES DATA VALIDATION
# ============================================================

def validate_features_data(features):
    """
    Validate Walmart features dataset.
    """

    required_columns = [
        "Store",
        "Date",
        "Temperature",
        "Fuel_Price",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "CPI",
        "Unemployment",
        "IsHoliday"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in features.columns
    ]

    return missing_columns


# ============================================================
# 6. STORES DATA VALIDATION
# ============================================================

def validate_stores_data(stores):
    """
    Validate Walmart stores dataset.
    """

    required_columns = [
        "Store",
        "Type",
        "Size"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in stores.columns
    ]

    return missing_columns


# ============================================================
# 7. MISSING VALUE ANALYSIS
# ============================================================

def check_missing_values(df):
    """
    Return missing-value counts.
    """

    return (
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )


# ============================================================
# 8. MARKDOWN MISSING VALUES
# ============================================================

def fill_markdown_values(features):
    """
    Fill missing Walmart markdown values with zero.

    Markdown columns:
        MarkDown1
        MarkDown2
        MarkDown3
        MarkDown4
        MarkDown5
    """

    features = features.copy()

    markdown_columns = [
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5"
    ]

    features[markdown_columns] = (
        features[markdown_columns]
        .fillna(0)
    )

    return features


# ============================================================
# 9. CPI AND UNEMPLOYMENT
# ============================================================

def fill_economic_features(features):
    """
    Handle missing CPI and Unemployment values.
    """

    features = features.copy()

    economic_columns = [
        "CPI",
        "Unemployment"
    ]

    features[economic_columns] = (
        features[economic_columns]
        .ffill()
        .bfill()
    )

    return features


# ============================================================
# 10. DUPLICATE CHECK
# ============================================================

def check_duplicates(df):
    """
    Count duplicate rows.
    """

    return df.duplicated().sum()


# ============================================================
# 11. BUSINESS KEY DUPLICATES
# ============================================================

def check_train_duplicates(train):
    """
    Check duplicate Store + Dept + Date records.
    """

    return train.duplicated(
        subset=[
            "Store",
            "Dept",
            "Date"
        ]
    ).sum()


def check_test_duplicates(test):
    """
    Check duplicate Store + Dept + Date records.
    """

    return test.duplicated(
        subset=[
            "Store",
            "Dept",
            "Date"
        ]
    ).sum()


# ============================================================
# 12. NEGATIVE WEEKLY SALES
# ============================================================

def check_negative_sales(train):
    """
    Return records with negative Weekly_Sales.
    """

    return train[
        train["Weekly_Sales"] < 0
    ].copy()


# ============================================================
# 13. MERGE TRAIN WITH FEATURES
# ============================================================

def merge_train_features(
    train,
    features
):
    """
    Merge train data with Walmart external features.

    Join keys:
        Store
        Date
    """

    merged = train.merge(
        features,
        on=[
            "Store",
            "Date"
        ],
        how="left",
        suffixes=(
            "",
            "_feature"
        )
    )

    return merged


# ============================================================
# 14. MERGE TEST WITH FEATURES
# ============================================================

def merge_test_features(
    test,
    features
):
    """
    Merge test data with Walmart external features.

    Join keys:
        Store
        Date
    """

    merged = test.merge(
        features,
        on=[
            "Store",
            "Date"
        ],
        how="left",
        suffixes=(
            "",
            "_feature"
        )
    )

    return merged


# ============================================================
# 15. MERGE WITH STORE INFORMATION
# ============================================================

def merge_store_information(
    df,
    stores
):
    """
    Add store Type and Size information.
    """

    df = df.merge(
        stores[
            [
                "Store",
                "Type",
                "Size"
            ]
        ],
        on="Store",
        how="left"
    )

    return df


# ============================================================
# 16. FINAL DATA QUALITY CHECK
# ============================================================

def data_quality_report(df):
    """
    Generate a complete data quality report.
    """

    report = {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Duplicate_Rows": int(
            df.duplicated().sum()
        ),
        "Missing_Values": int(
            df.isnull().sum().sum()
        )
    }

    if "Weekly_Sales" in df.columns:

        report["Negative_Sales"] = int(
            (df["Weekly_Sales"] < 0).sum()
        )

    return report


# ============================================================
# 17. DATASET SHAPE
# ============================================================

def dataset_shapes(
    train,
    test,
    features,
    stores
):
    """
    Return shapes of all Walmart datasets.
    """

    return {
        "Train": train.shape,
        "Test": test.shape,
        "Features": features.shape,
        "Stores": stores.shape
    }


# ============================================================
# 18. DATE RANGE
# ============================================================

def get_date_range(
    df,
    date_column="Date"
):
    """
    Return minimum and maximum dates.
    """

    return {
        "Start_Date": df[date_column].min(),
        "End_Date": df[date_column].max()
    }


# ============================================================
# 19. FINAL TRAIN / TEST INFORMATION
# ============================================================

def get_train_test_info(
    train,
    test
):
    """
    Return important train/test information.
    """

    return {
        "Train_Shape": train.shape,
        "Test_Shape": test.shape,

        "Train_Start_Date": train["Date"].min(),
        "Train_End_Date": train["Date"].max(),

        "Test_Start_Date": test["Date"].min(),
        "Test_End_Date": test["Date"].max()
    }