import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# WALMART SALES FORECASTING
# PREDICTION MODULE
# ============================================================

# The exact feature columns used by the final XGBoost model
FEATURE_COLUMNS = [
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
# LOAD MODEL
# ============================================================

def load_model(model_path):
    """
    Load the trained XGBoost model.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = joblib.load(model_path)

    print("Model loaded successfully.")

    return model


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):
    """
    Prepare the dataframe for prediction.

    Uses the exact 56 features used during
    XGBoost model training.
    """

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features:\n"
            + "\n".join(missing_features)
        )

    X = df[FEATURE_COLUMNS].copy()

    # Replace infinite values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # XGBoost can handle missing values,
    # so NaNs are intentionally retained.

    return X


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(model, df):
    """
    Generate Weekly Sales predictions
    using the trained XGBoost model.
    """

    X = prepare_features(df)

    predictions = model.predict(X)

    return predictions


# ============================================================
# CREATE FORECAST DATAFRAME
# ============================================================

def create_forecast_dataframe(
    df,
    predictions
):
    """
    Create the final forecast dataframe.

    Keeps Store, Dept and Date together
    with predicted Weekly Sales.
    """

    forecast = df[
        [
            "Store",
            "Dept",
            "Date"
        ]
    ].copy()

    forecast["Predicted_Weekly_Sales"] = predictions

    return forecast


# ============================================================
# ADD PREDICTIONS TO DATAFRAME
# ============================================================

def add_predictions(
    df,
    predictions
):
    """
    Add predictions directly to the
    original dataframe.
    """

    result = df.copy()

    result["Predicted_Weekly_Sales"] = predictions

    return result


# ============================================================
# FORECAST SUMMARY
# ============================================================

def create_forecast_summary(
    forecast
):
    """
    Create a summary of predicted sales.
    """

    summary = {
        "Total_Predicted_Sales":
            forecast[
                "Predicted_Weekly_Sales"
            ].sum(),

        "Average_Predicted_Sales":
            forecast[
                "Predicted_Weekly_Sales"
            ].mean(),

        "Minimum_Predicted_Sales":
            forecast[
                "Predicted_Weekly_Sales"
            ].min(),

        "Maximum_Predicted_Sales":
            forecast[
                "Predicted_Weekly_Sales"
            ].max(),

        "Forecast_Rows":
            len(forecast)
    }

    return pd.DataFrame(
        [summary]
    )


# ============================================================
# STORE-LEVEL FORECAST
# ============================================================

def store_forecast_summary(
    forecast
):
    """
    Calculate predicted sales by store.
    """

    return (
        forecast
        .groupby("Store", as_index=False)
        ["Predicted_Weekly_Sales"]
        .sum()
        .sort_values(
            "Predicted_Weekly_Sales",
            ascending=False
        )
    )


# ============================================================
# DEPARTMENT-LEVEL FORECAST
# ============================================================

def department_forecast_summary(
    forecast
):
    """
    Calculate predicted sales by department.
    """

    return (
        forecast
        .groupby("Dept", as_index=False)
        ["Predicted_Weekly_Sales"]
        .sum()
        .sort_values(
            "Predicted_Weekly_Sales",
            ascending=False
        )
    )


# ============================================================
# DATE-LEVEL FORECAST
# ============================================================

def date_forecast_summary(
    forecast
):
    """
    Calculate predicted sales by date.
    """

    return (
        forecast
        .groupby("Date", as_index=False)
        ["Predicted_Weekly_Sales"]
        .sum()
        .sort_values("Date")
    )


# ============================================================
# SAVE FORECAST
# ============================================================

def save_forecast(
    forecast,
    path="outputs/final_forecast.csv"
):
    """
    Save the final forecast to CSV.
    """

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    forecast.to_csv(
        path,
        index=False
    )

    print(
        f"Forecast saved to: {path}"
    )


# ============================================================
# COMPLETE PREDICTION PIPELINE
# ============================================================

def run_prediction(
    model_path,
    test_features_path,
    output_path="outputs/final_forecast.csv"
):
    """
    Complete prediction pipeline:

    1. Load trained XGBoost model
    2. Load test feature-engineered data
    3. Select the 56 model features
    4. Generate predictions
    5. Create forecast dataframe
    6. Save final forecast
    """

    # Load model
    model = load_model(model_path)

    # Load test features
    test = pd.read_csv(
        test_features_path
    )

    # Convert date
    if "Date" in test.columns:
        test["Date"] = pd.to_datetime(
            test["Date"]
        )

    print(
        f"Test data loaded: {test.shape}"
    )

    # Generate predictions
    predictions = generate_predictions(
        model,
        test
    )

    # Create forecast
    forecast = create_forecast_dataframe(
        test,
        predictions
    )

    # Save forecast
    save_forecast(
        forecast,
        output_path
    )

    print(
        f"Predictions generated: "
        f"{len(forecast):,}"
    )

    return forecast


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    MODEL_PATH = (
        "models/xgboost_model.pkl"
    )

    TEST_FEATURES_PATH = (
        "data/processed/"
        "test_feature_engineered.csv"
    )

    OUTPUT_PATH = (
        "outputs/final_forecast.csv"
    )

    forecast = run_prediction(
        model_path=MODEL_PATH,
        test_features_path=TEST_FEATURES_PATH,
        output_path=OUTPUT_PATH
    )

    print("\nForecast Preview:")
    print(
        forecast.head()
    )

    print("\nForecast Summary:")
    print(
        create_forecast_summary(
            forecast
        )
    )