import numpy as np
import pandas as pd


# ============================================================
# WALMART SALES FORECASTING
# FEATURE ENGINEERING
# ============================================================

# These are the exact 56 features used by the final model.
MODEL_FEATURES = [
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


# ============================================================
# 1. DATE FEATURES
# ============================================================

def create_date_features(df):
    """
    Create calendar-based features from Date.
    """

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Quarter"] = df["Date"].dt.quarter
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # Monday-Friday = 0-4
    df["IsWeekend"] = (
        df["DayOfWeek"] >= 5
    ).astype(int)

    return df


# ============================================================
# 2. HOLIDAY FEATURES
# ============================================================

def create_holiday_features(df):
    """
    Create holiday-related features.

    IsHoliday comes from the Walmart dataset.
    """

    df = df.copy()

    if "IsHoliday" in df.columns:
        df["Holiday_Flag"] = (
            df["IsHoliday"]
            .astype(int)
        )
    else:
        df["Holiday_Flag"] = 0

    # Create a date-level holiday table.
    holiday_dates = (
        df.groupby("Date")["Holiday_Flag"]
        .max()
        .sort_index()
    )

    previous_holiday = (
        holiday_dates
        .shift(1)
        .fillna(0)
    )

    next_holiday = (
        holiday_dates
        .shift(-1)
        .fillna(0)
    )

    df["Previous_Holiday"] = (
        df["Date"]
        .map(previous_holiday)
        .fillna(0)
        .astype(int)
    )

    df["Next_Holiday"] = (
        df["Date"]
        .map(next_holiday)
        .fillna(0)
        .astype(int)
    )

    # Holiday itself or immediately before/after holiday.
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


# ============================================================
# 3. LAG FEATURES
# ============================================================

def create_lag_features(
    df,
    history=None
):
    """
    Create historical Weekly_Sales lag features.

    Lags:
        1
        2
        4
        8
        12
        52 weeks

    For test data, history should contain the
    available training Weekly_Sales.
    """

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    if "Weekly_Sales" not in df.columns:
        df["Weekly_Sales"] = np.nan

    if history is not None:

        history = history.copy()

        history["Date"] = pd.to_datetime(
            history["Date"]
        )

        combined = pd.concat(
            [
                history[
                    [
                        "Store",
                        "Dept",
                        "Date",
                        "Weekly_Sales"
                    ]
                ],
                df[
                    [
                        "Store",
                        "Dept",
                        "Date",
                        "Weekly_Sales"
                    ]
                ]
            ],
            ignore_index=True
        )

    else:
        combined = df[
            [
                "Store",
                "Dept",
                "Date",
                "Weekly_Sales"
            ]
        ].copy()

    combined = combined.sort_values(
        [
            "Store",
            "Dept",
            "Date"
        ]
    )

    grouped = combined.groupby(
        ["Store", "Dept"]
    )["Weekly_Sales"]

    for lag in [1, 2, 4, 8, 12, 52]:

        combined[f"Lag_{lag}"] = (
            grouped.shift(lag)
        )

    # Keep only the rows belonging to df.
    df = df.merge(
        combined[
            [
                "Store",
                "Dept",
                "Date",
                "Lag_1",
                "Lag_2",
                "Lag_4",
                "Lag_8",
                "Lag_12",
                "Lag_52"
            ]
        ],
        on=[
            "Store",
            "Dept",
            "Date"
        ],
        how="left"
    )

    return df


# ============================================================
# 4. ROLLING FEATURES
# ============================================================

def create_rolling_features(df):
    """
    Create rolling statistics from lagged sales.

    Uses only previous sales values.
    """

    df = df.copy()

    df = df.sort_values(
        [
            "Store",
            "Dept",
            "Date"
        ]
    )

    # Previous sales history
    sales = df.groupby(
        ["Store", "Dept"]
    )["Weekly_Sales"]

    shifted_sales = sales.shift(1)

    df["_Previous_Sales"] = shifted_sales

    grouped = df.groupby(
        ["Store", "Dept"]
    )["_Previous_Sales"]

    for window in [4, 8, 12]:

        df[
            f"Rolling_Mean_{window}"
        ] = grouped.transform(
            lambda x:
            x.rolling(
                window,
                min_periods=1
            ).mean()
        )

        df[
            f"Rolling_Std_{window}"
        ] = grouped.transform(
            lambda x:
            x.rolling(
                window,
                min_periods=2
            ).std()
        )

    df["Rolling_Min_4"] = grouped.transform(
        lambda x:
        x.rolling(
            4,
            min_periods=1
        ).min()
    )

    df["Rolling_Max_4"] = grouped.transform(
        lambda x:
        x.rolling(
            4,
            min_periods=1
        ).max()
    )

    df = df.drop(
        columns=["_Previous_Sales"]
    )

    return df


# ============================================================
# 5. EXPANDING FEATURES
# ============================================================

def create_expanding_features(df):
    """
    Create expanding historical sales statistics.
    """

    df = df.copy()

    df = df.sort_values(
        [
            "Store",
            "Dept",
            "Date"
        ]
    )

    previous_sales = (
        df.groupby(
            ["Store", "Dept"]
        )["Weekly_Sales"]
        .shift(1)
    )

    temp = df.copy()

    temp["_Previous_Sales"] = previous_sales

    grouped = temp.groupby(
        ["Store", "Dept"]
    )["_Previous_Sales"]

    df["Expanding_Mean"] = grouped.transform(
        lambda x:
        x.expanding(
            min_periods=1
        ).mean()
    )

    df["Expanding_Std"] = grouped.transform(
        lambda x:
        x.expanding(
            min_periods=2
        ).std()
    )

    df["Expanding_Min"] = grouped.transform(
        lambda x:
        x.expanding(
            min_periods=1
        ).min()
    )

    df["Expanding_Max"] = grouped.transform(
        lambda x:
        x.expanding(
            min_periods=1
        ).max()
    )

    return df


# ============================================================
# 6. STORE FEATURES
# ============================================================

def create_store_features(
    df,
    reference_data=None
):
    """
    Create store-level historical statistics.
    """

    df = df.copy()

    if reference_data is None:
        reference_data = df

    reference_data = reference_data.copy()

    store_stats = (
        reference_data
        .groupby("Store")["Weekly_Sales"]
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

    # Rank stores by average sales.
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

    # Store type.
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

    else:
        df["Store_Type"] = np.nan

    # Store size relative to average store size.
    if "Size" in df.columns:

        average_size = (
            reference_data["Size"].mean()
            if "Size" in reference_data.columns
            else df["Size"].mean()
        )

        df["Store_Size_Relative"] = (
            df["Size"] /
            average_size
        )

    else:
        df["Store_Size_Relative"] = np.nan

    return df


# ============================================================
# 7. DEPARTMENT FEATURES
# ============================================================

def create_department_features(
    df,
    reference_data=None
):
    """
    Create department-level sales statistics.
    """

    df = df.copy()

    if reference_data is None:
        reference_data = df

    reference_data = reference_data.copy()

    dept_stats = (
        reference_data
        .groupby("Dept")["Weekly_Sales"]
        .agg(
            Dept_Avg_Sales="mean",
            Dept_Std_Sales="std",
            Dept_Min_Sales="min",
            Dept_Max_Sales="max"
        )
        .reset_index()
    )

    df = df.merge(
        dept_stats,
        on="Dept",
        how="left"
    )

    # Rank departments.
    dept_rank = (
        dept_stats
        .sort_values(
            "Dept_Avg_Sales",
            ascending=False
        )
        .reset_index(drop=True)
    )

    dept_rank["Dept_Sales_Rank"] = (
        dept_rank.index + 1
    )

    df = df.merge(
        dept_rank[
            [
                "Dept",
                "Dept_Sales_Rank"
            ]
        ],
        on="Dept",
        how="left"
    )

    # Department contribution to total sales.
    total_sales = (
        reference_data["Weekly_Sales"]
        .sum()
    )

    dept_sales = (
        reference_data
        .groupby("Dept")["Weekly_Sales"]
        .sum()
        .reset_index(
            name="Dept_Total_Sales"
        )
    )

    if total_sales != 0:

        dept_sales[
            "Dept_Sales_Contribution"
        ] = (
            dept_sales["Dept_Total_Sales"]
            / total_sales
        )

    else:

        dept_sales[
            "Dept_Sales_Contribution"
        ] = 0

    df = df.merge(
        dept_sales[
            [
                "Dept",
                "Dept_Sales_Contribution"
            ]
        ],
        on="Dept",
        how="left"
    )

    return df


# ============================================================
# 8. CYCLICAL FEATURES
# ============================================================

def create_cyclical_features(df):
    """
    Encode calendar variables using sine/cosine.
    """

    df = df.copy()

    # Month
    df["Month_Sin"] = np.sin(
        2 * np.pi * df["Month"] / 12
    )

    df["Month_Cos"] = np.cos(
        2 * np.pi * df["Month"] / 12
    )

    # Week
    df["Week_Sin"] = np.sin(
        2 * np.pi * df["Week"] / 52
    )

    df["Week_Cos"] = np.cos(
        2 * np.pi * df["Week"] / 52
    )

    # Quarter
    df["Quarter_Sin"] = np.sin(
        2 * np.pi * df["Quarter"] / 4
    )

    df["Quarter_Cos"] = np.cos(
        2 * np.pi * df["Quarter"] / 4
    )

    # Day of week
    df["DayOfWeek_Sin"] = np.sin(
        2 * np.pi * df["DayOfWeek"] / 7
    )

    df["DayOfWeek_Cos"] = np.cos(
        2 * np.pi * df["DayOfWeek"] / 7
    )

    return df


# ============================================================
# 9. CLEAN FEATURE VALUES
# ============================================================

def clean_feature_values(df):
    """
    Clean infinite and missing feature values.
    """

    df = df.copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    return df


# ============================================================
# 10. CREATE TRAIN FEATURES
# ============================================================

def create_train_features(
    train,
    features=None,
    stores=None
):
    """
    Complete feature engineering pipeline
    for training data.
    """

    df = train.copy()

    # --------------------------------------------------------
    # Merge external Walmart features
    # --------------------------------------------------------

    if features is not None:

        features = features.copy()

        features["Date"] = pd.to_datetime(
            features["Date"]
        )

        df["Date"] = pd.to_datetime(
            df["Date"]
        )

        feature_columns = [
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
            "Unemployment"
        ]

        available = [
            col
            for col in feature_columns
            if col in features.columns
        ]

        df = df.merge(
            features[available],
            on=["Store", "Date"],
            how="left",
            suffixes=("", "_feature")
        )

    # --------------------------------------------------------
    # Merge store information
    # --------------------------------------------------------

    if stores is not None:

        store_columns = [
            "Store",
            "Type",
            "Size"
        ]

        available = [
            col
            for col in store_columns
            if col in stores.columns
        ]

        df = df.merge(
            stores[available],
            on="Store",
            how="left",
            suffixes=("", "_store")
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    df = create_date_features(df)

    df = create_holiday_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = create_expanding_features(df)

    df = create_store_features(
        df,
        reference_data=df
    )

    df = create_department_features(
        df,
        reference_data=df
    )

    df = create_cyclical_features(df)

    df = clean_feature_values(df)

    return df


# ============================================================
# 11. CREATE TEST FEATURES
# ============================================================

def create_test_features(
    test,
    train_history,
    features=None,
    stores=None
):
    """
    Complete feature engineering pipeline
    for Walmart test data.

    train_history must contain historical
    Weekly_Sales from train.csv.
    """

    df = test.copy()

    train_history = train_history.copy()

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    train_history["Date"] = pd.to_datetime(
        train_history["Date"]
    )

    # Test does not have Weekly_Sales.
    df["Weekly_Sales"] = np.nan

    # --------------------------------------------------------
    # Merge Walmart external features
    # --------------------------------------------------------

    if features is not None:

        features = features.copy()

        features["Date"] = pd.to_datetime(
            features["Date"]
        )

        feature_columns = [
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
            "Unemployment"
        ]

        available = [
            col
            for col in feature_columns
            if col in features.columns
        ]

        df = df.merge(
            features[available],
            on=["Store", "Date"],
            how="left",
            suffixes=("", "_feature")
        )

    # --------------------------------------------------------
    # Merge store information
    # --------------------------------------------------------

    if stores is not None:

        store_columns = [
            "Store",
            "Type",
            "Size"
        ]

        available = [
            col
            for col in store_columns
            if col in stores.columns
        ]

        df = df.merge(
            stores[available],
            on="Store",
            how="left",
            suffixes=("", "_store")
        )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df = create_date_features(df)

    # --------------------------------------------------------
    # Holiday
    # --------------------------------------------------------

    combined_dates = pd.concat(
        [
            train_history[
                ["Store", "Date", "IsHoliday"]
            ],
            df[
                ["Store", "Date", "IsHoliday"]
            ]
        ],
        ignore_index=True
    )

    df = create_holiday_features(
        pd.concat(
            [
                train_history[
                    [
                        "Store",
                        "Dept",
                        "Date",
                        "IsHoliday"
                    ]
                ],
                df[
                    [
                        "Store",
                        "Dept",
                        "Date",
                        "IsHoliday"
                    ]
                ]
            ],
            ignore_index=True
        )
    )

    # Keep only test dates.
    df = df[
        df["Date"].isin(
            test["Date"]
        )
    ].copy()

    # --------------------------------------------------------
    # Restore external columns if needed
    # --------------------------------------------------------

    if features is not None:

        features = features.copy()

        features["Date"] = pd.to_datetime(
            features["Date"]
        )

        merge_columns = [
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
            "Unemployment"
        ]

        available = [
            col
            for col in merge_columns
            if col in features.columns
        ]

        # Remove duplicate columns before merge.
        for col in available:
            if col not in [
                "Store",
                "Date"
            ] and col in df.columns:
                df = df.drop(
                    columns=[col]
                )

        df = df.merge(
            features[available],
            on=["Store", "Date"],
            how="left"
        )

    # --------------------------------------------------------
    # Restore store information
    # --------------------------------------------------------

    if stores is not None:

        for col in ["Type", "Size"]:

            if col in df.columns:
                df = df.drop(
                    columns=[col]
                )

        df = df.merge(
            stores[
                ["Store", "Type", "Size"]
            ],
            on="Store",
            how="left"
        )

    # --------------------------------------------------------
    # Lags using historical training sales
    # --------------------------------------------------------

    df = create_lag_features(
        df,
        history=train_history
    )

    # --------------------------------------------------------
    # Rolling features
    #
    # Combine historical train data with test rows
    # so test features can use previous sales.
    # --------------------------------------------------------

    history_for_rolling = train_history[
        [
            "Store",
            "Dept",
            "Date",
            "Weekly_Sales"
        ]
    ].copy()

    combined = pd.concat(
        [
            history_for_rolling,
            df[
                [
                    "Store",
                    "Dept",
                    "Date",
                    "Weekly_Sales"
                ]
            ]
        ],
        ignore_index=True
    )

    combined = combined.sort_values(
        [
            "Store",
            "Dept",
            "Date"
        ]
    )

    combined = create_rolling_features(
        combined
    )

    combined = create_expanding_features(
        combined
    )

    df = df.drop(
        columns=[
            col
            for col in [
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
                "Expanding_Max"
            ]
            if col in df.columns
        ]
    )

    rolling_columns = [
        "Store",
        "Dept",
        "Date",
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
        "Expanding_Max"
    ]

    df = df.merge(
        combined[rolling_columns],
        on=[
            "Store",
            "Dept",
            "Date"
        ],
        how="left"
    )

    # --------------------------------------------------------
    # Store and department historical statistics
    # --------------------------------------------------------

    df = create_store_features(
        df,
        reference_data=train_history
    )

    df = create_department_features(
        df,
        reference_data=train_history
    )

    # --------------------------------------------------------
    # Cyclical features
    # --------------------------------------------------------

    df = create_cyclical_features(df)

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = clean_feature_values(df)

    return df


# ============================================================
# 12. SELECT MODEL FEATURES
# ============================================================

def select_model_features(df):
    """
    Return exactly the 56 features required
    by the XGBoost model.
    """

    missing = [
        col
        for col in MODEL_FEATURES
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required model features:\n"
            + "\n".join(missing)
        )

    return df[
        MODEL_FEATURES
    ].copy()


# ============================================================
# 13. SAVE FEATURE-ENGINEERED DATA
# ============================================================

def save_feature_engineered_data(
    df,
    path
):
    """
    Save feature-engineered dataset.
    """

    df.to_csv(
        path,
        index=False
    )


# ============================================================
# 14. FEATURE ENGINEERING SUMMARY
# ============================================================

def feature_engineering_summary(df):
    """
    Return useful information about
    the engineered dataset.
    """

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "required_model_features": len(
            MODEL_FEATURES
        ),
        "available_model_features": sum(
            col in df.columns
            for col in MODEL_FEATURES
        ),
        "missing_model_features": [
            col
            for col in MODEL_FEATURES
            if col not in df.columns
        ]
    }