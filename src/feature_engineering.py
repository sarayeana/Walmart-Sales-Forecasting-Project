from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================

# PATH CONFIGURATION

# ============================================================

PROJECT_ROOT = Path(**file**).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ============================================================

# FEATURE COLUMN CONFIGURATION

# ============================================================

LAG_COLUMNS = [
"Lag_1",
"Lag_2",
"Lag_4",
"Lag_8",
"Lag_12",
"Lag_52"
]

ROLLING_COLUMNS = [
"Rolling_Mean_4",
"Rolling_Mean_8",
"Rolling_Mean_12",
"Rolling_Std_4",
"Rolling_Std_8",
"Rolling_Std_12",
"Rolling_Min_4",
"Rolling_Max_4"
]

EXPANDING_COLUMNS = [
"Expanding_Mean",
"Expanding_Std",
"Expanding_Min",
"Expanding_Max"
]

# ============================================================

# DATE FEATURES

# ============================================================

def create_date_features(df):
"""
Create date-based features from the Date column.
"""

```
df = df.copy()

df["Date"] = pd.to_datetime(
    df["Date"]
)

df["Year"] = df["Date"].dt.year

df["Quarter"] = (
    df["Date"]
    .dt
    .quarter
)

df["Month"] = (
    df["Date"]
    .dt
    .month
)

df["Week"] = (
    df["Date"]
    .dt
    .isocalendar()
    .week
    .astype(int)
)

df["Day"] = (
    df["Date"]
    .dt
    .day
)

df["DayOfWeek"] = (
    df["Date"]
    .dt
    .dayofweek
)

df["IsWeekend"] = (
    df["DayOfWeek"]
    .isin([5, 6])
    .astype(int)
)

return df
```

# ============================================================

# HOLIDAY FEATURES

# ============================================================

def create_holiday_features(df):
"""
Create holiday-related features.
"""

```
df = df.copy()

df["Holiday_Flag"] = (
    df["IsHoliday"]
    .astype(int)
)

df = df.sort_values(
    ["Store", "Dept", "Date"]
)

df["Previous_Holiday"] = (
    df
    .groupby(
        ["Store", "Dept"]
    )["Holiday_Flag"]
    .shift(1)
    .fillna(0)
)

df["Next_Holiday"] = (
    df
    .groupby(
        ["Store", "Dept"]
    )["Holiday_Flag"]
    .shift(-1)
    .fillna(0)
)

df["Near_Holiday"] = (
    (
        df["Holiday_Flag"] == 1
    )
    |
    (
        df["Previous_Holiday"] == 1
    )
    |
    (
        df["Next_Holiday"] == 1
    )
).astype(int)

return df
```

# ============================================================

# LAG FEATURES

# ============================================================

def create_lag_features(
df,
target_column="Weekly_Sales",
lags=(1, 2, 4, 8, 12, 52)
):
"""
Create lag features using Store and Department history.

```
This function is intended primarily for training data
where Weekly_Sales is available.
"""

df = df.copy()

df = df.sort_values(
    ["Store", "Dept", "Date"]
)

grouped_sales = (
    df
    .groupby(
        ["Store", "Dept"]
    )[target_column]
)

for lag in lags:

    df[f"Lag_{lag}"] = (
        grouped_sales
        .shift(lag)
    )

return df
```

# ============================================================

# TEST LAG FEATURES

# ============================================================

def create_test_lag_features(
test,
train,
target_column="Weekly_Sales",
lags=(1, 2, 4, 8, 12, 52)
):
"""
Create lag features for test data using historical
Weekly_Sales from the training dataset.

```
The test dataset is expected to not contain actual sales.
"""

test = test.copy()
train = train.copy()

train["Date"] = pd.to_datetime(
    train["Date"]
)

test["Date"] = pd.to_datetime(
    test["Date"]
)

train_history = train[
    [
        "Store",
        "Dept",
        "Date",
        target_column
    ]
].copy()

test[target_column] = np.nan

combined = pd.concat(
    [
        train_history,
        test[
            [
                "Store",
                "Dept",
                "Date",
                target_column
            ]
        ]
    ],
    ignore_index=True
)

combined = combined.sort_values(
    ["Store", "Dept", "Date"]
)

grouped_sales = (
    combined
    .groupby(
        ["Store", "Dept"]
    )[target_column]
)

for lag in lags:

    combined[f"Lag_{lag}"] = (
        grouped_sales
        .shift(lag)
    )

test_features = combined[
    combined[target_column].isna()
].copy()

return test_features[
    [
        "Store",
        "Dept",
        "Date"
    ]
    +
    [
        f"Lag_{lag}"
        for lag in lags
    ]
]
```

# ============================================================

# ROLLING FEATURES

# ============================================================

def create_rolling_features(
df
):
"""
Create rolling statistics from lag features.

```
Uses historical lag values rather than the current
Weekly_Sales value to avoid target leakage.
"""

df = df.copy()

df["Rolling_Mean_4"] = (
    df[
        ["Lag_1", "Lag_2", "Lag_4"]
    ]
    .mean(axis=1)
)

df["Rolling_Mean_8"] = (
    df[
        [
            "Lag_1",
            "Lag_2",
            "Lag_4",
            "Lag_8"
        ]
    ]
    .mean(axis=1)
)

df["Rolling_Mean_12"] = (
    df[
        [
            "Lag_1",
            "Lag_2",
            "Lag_4",
            "Lag_8",
            "Lag_12"
        ]
    ]
    .mean(axis=1)
)

df["Rolling_Std_4"] = (
    df[
        ["Lag_1", "Lag_2", "Lag_4"]
    ]
    .std(axis=1)
)

df["Rolling_Std_8"] = (
    df[
        [
            "Lag_1",
            "Lag_2",
            "Lag_4",
            "Lag_8"
        ]
    ]
    .std(axis=1)
)

df["Rolling_Std_12"] = (
    df[
        [
            "Lag_1",
            "Lag_2",
            "Lag_4",
            "Lag_8",
            "Lag_12"
        ]
    ]
    .std(axis=1)
)

df["Rolling_Min_4"] = (
    df[
        [
            "Lag_1",
            "Lag_2",
            "Lag_4",
            "Lag_8"
        ]
    ]
    .min(axis=1)
)

df["Rolling_Max_4"] = (
    df[
        [
            "Lag_1",
            "Lag_2",
            "Lag_4",
            "Lag_8"
        ]
    ]
    .max(axis=1)
)

return df
```

# ============================================================

# EXPANDING FEATURES

# ============================================================

def create_expanding_features(
df,
target_column="Weekly_Sales"
):
"""
Create expanding historical statistics for each
Store and Department combination.
"""

```
df = df.copy()

df = df.sort_values(
    ["Store", "Dept", "Date"]
)

grouped = (
    df
    .groupby(
        ["Store", "Dept"]
    )[target_column]
)

df["Expanding_Mean"] = (
    grouped
    .transform(
        lambda x: (
            x.shift(1)
            .expanding()
            .mean()
        )
    )
)

df["Expanding_Std"] = (
    grouped
    .transform(
        lambda x: (
            x.shift(1)
            .expanding()
            .std()
        )
    )
)

df["Expanding_Min"] = (
    grouped
    .transform(
        lambda x: (
            x.shift(1)
            .expanding()
            .min()
        )
    )
)

df["Expanding_Max"] = (
    grouped
    .transform(
        lambda x: (
            x.shift(1)
            .expanding()
            .max()
        )
    )
)

return df
```

# ============================================================

# STORE FEATURES

# ============================================================

def create_store_features(
df,
reference_df=None,
target_column="Weekly_Sales"
):
"""
Create Store-level sales statistics.

```
For test data, provide the training dataframe as
reference_df so statistics are calculated from
historical training sales.
"""

df = df.copy()

if reference_df is None:

    reference_df = df

store_stats = (
    reference_df
    .groupby("Store")[target_column]
    .agg(
        Store_Avg_Sales="mean",
        Store_Std_Sales="std",
        Store_Min_Sales="min",
        Store_Max_Sales="max"
    )
    .reset_index()
)

df = df.merge(
    store_stats,
    on="Store",
    how="left"
)

store_rank = (
    store_stats
    .sort_values(
        "Store_Avg_Sales",
        ascending=False
    )
    .reset_index(drop=True)
)

store_rank["Store_Sales_Rank"] = (
    store_rank.index + 1
)

df = df.merge(
    store_rank[
        [
            "Store",
            "Store_Sales_Rank"
        ]
    ],
    on="Store",
    how="left"
)

return df
```

# ============================================================

# STORE STRUCTURAL FEATURES

# ============================================================

def create_store_structure_features(
df
):
"""
Create numerical features based on Walmart
store Type and Size.
"""

```
df = df.copy()

if "Type" in df.columns:

    type_mapping = {
        "A": 1,
        "B": 2,
        "C": 3
    }

    df["Store_Type"] = (
        df["Type"]
        .map(type_mapping)
    )

if "Size" in df.columns:

    average_size = (
        df["Size"]
        .mean()
    )

    df["Store_Size_Relative"] = (
        df["Size"]
        /
        average_size
    )

return df
```

# ============================================================

# DEPARTMENT FEATURES

# ============================================================

def create_department_features(
df,
reference_df=None,
target_column="Weekly_Sales"
):
"""
Create Department-level sales statistics.

```
For test data, provide training data as reference_df.
"""

df = df.copy()

if reference_df is None:

    reference_df = df

department_stats = (
    reference_df
    .groupby("Dept")[target_column]
    .agg(
        Dept_Avg_Sales="mean",
        Dept_Std_Sales="std",
        Dept_Min_Sales="min",
        Dept_Max_Sales="max"
    )
    .reset_index()
)

df = df.merge(
    department_stats,
    on="Dept",
    how="left"
)

department_rank = (
    department_stats
    .sort_values(
        "Dept_Avg_Sales",
        ascending=False
    )
    .reset_index(drop=True)
)

department_rank["Dept_Sales_Rank"] = (
    department_rank.index + 1
)

df = df.merge(
    department_rank[
        [
            "Dept",
            "Dept_Sales_Rank"
        ]
    ],
    on="Dept",
    how="left"
)

total_sales = (
    reference_df[target_column]
    .sum()
)

department_contribution = (
    reference_df
    .groupby("Dept")[target_column]
    .sum()
    /
    total_sales
)

department_contribution = (
    department_contribution
    .reset_index()
    .rename(
        columns={
            target_column:
            "Dept_Sales_Contribution"
        }
    )
)

df = df.merge(
    department_contribution,
    on="Dept",
    how="left"
)

return df
```

# ============================================================

# CYCLICAL FEATURES

# ============================================================

def create_cyclical_features(
df
):
"""
Create cyclical encodings for calendar features.
"""

```
df = df.copy()

df["Month_Sin"] = np.sin(
    2
    *
    np.pi
    *
    df["Month"]
    /
    12
)

df["Month_Cos"] = np.cos(
    2
    *
    np.pi
    *
    df["Month"]
    /
    12
)

df["Week_Sin"] = np.sin(
    2
    *
    np.pi
    *
    df["Week"]
    /
    52
)

df["Week_Cos"] = np.cos(
    2
    *
    np.pi
    *
    df["Week"]
    /
    52
)

df["Quarter_Sin"] = np.sin(
    2
    *
    np.pi
    *
    df["Quarter"]
    /
    4
)

df["Quarter_Cos"] = np.cos(
    2
    *
    np.pi
    *
    df["Quarter"]
    /
    4
)

df["DayOfWeek_Sin"] = np.sin(
    2
    *
    np.pi
    *
    df["DayOfWeek"]
    /
    7
)

df["DayOfWeek_Cos"] = np.cos(
    2
    *
    np.pi
    *
    df["DayOfWeek"]
    /
    7
)

return df
```

# ============================================================

# MISSING VALUE HANDLING

# ============================================================

def handle_feature_missing_values(
df
):
"""
Handle missing values in engineered features.
"""

```
df = df.copy()

numeric_columns = (
    df
    .select_dtypes(
        include=[
            np.number
        ]
    )
    .columns
)

for column in numeric_columns:

    df[column] = (
        df[column]
        .fillna(
            df[column].median()
        )
    )

return df
```

# ============================================================

# MODEL FEATURES

# ============================================================

def get_model_features():
"""
Return the exact feature columns used by the
Walmart XGBoost forecasting model.
"""

```
return [
    "Store",
    "Dept",

    "Year",
    "Quarter",
    "Month",
    "Week",
    "Day",
    "DayOfWeek",
    "IsWeekend",

    "Holiday_Flag",
    "Previous_Holiday",
    "Next_Holiday",
    "Near_Holiday",

    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",

    "Lag_1",
    "Lag_2",
    "Lag_4",
    "Lag_8",
    "Lag_12",
    "Lag_52",

    "Rolling_Mean_4",
    "Rolling_Mean_8",
    "Rolling_Mean_12",

    "Rolling_Std_4",
    "Rolling_Std_8",
    "Rolling_Std_12",

    "Rolling_Min_4",
    "Rolling_Max_4",

    "Expanding_Mean",
    "Expanding_Std",
    "Expanding_Min",
    "Expanding_Max",

    "Store_Avg_Sales",
    "Store_Std_Sales",
    "Store_Min_Sales",
    "Store_Max_Sales",
    "Store_Sales_Rank",

    "Store_Type",
    "Store_Size_Relative",

    "Dept_Avg_Sales",
    "Dept_Std_Sales",
    "Dept_Min_Sales",
    "Dept_Max_Sales",
    "Dept_Sales_Rank",
    "Dept_Sales_Contribution",

    "Month_Sin",
    "Month_Cos",

    "Week_Sin",
    "Week_Cos",

    "Quarter_Sin",
    "Quarter_Cos",

    "DayOfWeek_Sin",
    "DayOfWeek_Cos"
]
```

# ============================================================

# FEATURE VALIDATION

# ============================================================

def validate_model_features(
df,
feature_columns=None
):
"""
Check that all required model features
exist in a dataframe.
"""

```
if feature_columns is None:

    feature_columns = (
        get_model_features()
    )

missing_features = [
    feature
    for feature in feature_columns
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing required model features: "
        f"{missing_features}"
    )

return True
```

# ============================================================

# FEATURE PREPARATION

# ============================================================

def prepare_model_features(
df,
feature_columns=None
):
"""
Select and prepare features for model input.
"""

```
if feature_columns is None:

    feature_columns = (
        get_model_features()
    )

validate_model_features(
    df,
    feature_columns
)

X = df[
    feature_columns
].copy()

return X
```

# ============================================================

# SAVE FEATURE DATASETS

# ============================================================

def save_feature_dataset(
df,
filename
):
"""
Save an engineered dataset to data/processed.
"""

```
PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

file_path = (
    PROCESSED_DATA_DIR
    /
    filename
)

df.to_csv(
    file_path,
    index=False
)

return file_path
```

# ============================================================

# COMPLETE TRAIN FEATURE PIPELINE

# ============================================================

def create_train_features(
train
):
"""
Run the complete feature engineering pipeline
for Walmart training data.
"""

```
train = train.copy()

train = create_date_features(
    train
)

train = create_holiday_features(
    train
)

train = create_lag_features(
    train
)

train = create_rolling_features(
    train
)

train = create_expanding_features(
    train
)

train = create_store_features(
    train
)

train = create_store_structure_features(
    train
)

train = create_department_features(
    train
)

train = create_cyclical_features(
    train
)

return train
```

# ============================================================

# COMPLETE TEST FEATURE PIPELINE

# ============================================================

def create_test_features(
test,
train
):
"""
Run the feature engineering pipeline for
Walmart test data using training data
as historical reference.
"""

```
test = test.copy()
train = train.copy()

test = create_date_features(
    test
)

test = create_holiday_features(
    test
)

lag_features = create_test_lag_features(
    test,
    train
)

test = test.merge(
    lag_features,
    on=[
        "Store",
        "Dept",
        "Date"
    ],
    how="left"
)

test = create_rolling_features(
    test
)

test = create_store_features(
    test,
    reference_df=train
)

test = create_store_structure_features(
    test
)

test = create_department_features(
    test,
    reference_df=train
)

test = create_cyclical_features(
    test
)

return test
```
