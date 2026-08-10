import os
import joblib
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# ============================================================
# WALMART SALES FORECASTING
# MODEL TRAINING
# ============================================================


# Exact features used in the final XGBoost model
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
# LOAD TRAINING DATA
# ============================================================

def load_training_data(path):
    """
    Load the processed training dataset.
    """

    train = pd.read_csv(path)

    if "Date" in train.columns:
        train["Date"] = pd.to_datetime(
            train["Date"]
        )

    print(
        f"Training data loaded: {train.shape}"
    )

    return train


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_training_data(train):
    """
    Separate features (X) and target (y).
    """

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in train.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features:\n"
            + "\n".join(missing_features)
        )

    if "Weekly_Sales" not in train.columns:
        raise ValueError(
            "Weekly_Sales column not found."
        )

    X = train[
        FEATURE_COLUMNS
    ].copy()

    y = train[
        "Weekly_Sales"
    ].copy()

    # Replace infinite values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return X, y


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def split_data(
    X,
    y,
    test_size=0.20,
    random_state=42
):
    """
    Split training data into training
    and validation datasets.
    """

    X_train, X_valid, y_train, y_valid = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )
    )

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_valid: {X_valid.shape}"
    )

    return (
        X_train,
        X_valid,
        y_train,
        y_valid
    )


# ============================================================
# CREATE XGBOOST MODEL
# ============================================================

def create_xgboost_model():
    """
    Create the XGBoost regression model.
    """

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    X_train,
    y_train
):
    """
    Train the XGBoost model.
    """

    print(
        "\nTraining XGBoost model..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "XGBoost training completed."
    )

    return model


# ============================================================
# VALIDATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_valid,
    y_valid
):
    """
    Evaluate XGBoost on validation data.
    """

    predictions = model.predict(
        X_valid
    )

    mae = mean_absolute_error(
        y_valid,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_valid,
            predictions
        )
    )

    print("\nValidation Results")
    print("-" * 40)
    print(
        f"MAE  : {mae:,.2f}"
    )
    print(
        f"RMSE : {rmse:,.2f}"
    )

    return {
        "MAE": mae,
        "RMSE": rmse
    }


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    path="models/xgboost_model.pkl"
):
    """
    Save the trained XGBoost model.
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    joblib.dump(
        model,
        path
    )

    print(
        f"\nModel saved to: {path}"
    )


# ============================================================
# COMPLETE TRAINING PIPELINE
# ============================================================

def run_training(
    train_path,
    model_path="models/xgboost_model.pkl"
):
    """
    Complete model-training pipeline.

    Steps:
    1. Load processed training data
    2. Select 56 features
    3. Separate target
    4. Split training/validation data
    5. Create XGBoost model
    6. Train model
    7. Evaluate model
    8. Save model
    """

    # Load data
    train = load_training_data(
        train_path
    )

    # Prepare features and target
    X, y = prepare_training_data(
        train
    )

    print(
        f"\nNumber of features: {X.shape[1]}"
    )

    # Split data
    (
        X_train,
        X_valid,
        y_train,
        y_valid
    ) = split_data(
        X,
        y
    )

    # Create model
    model = create_xgboost_model()

    # Train
    model = train_model(
        model,
        X_train,
        y_train
    )

    # Evaluate
    metrics = evaluate_model(
        model,
        X_valid,
        y_valid
    )

    # Save
    save_model(
        model,
        model_path
    )

    return model, metrics


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    TRAIN_PATH = (
        "data/processed/"
        "train_feature_engineered.csv"
    )

    MODEL_PATH = (
        "models/"
        "xgboost_model.pkl"
    )

    model, metrics = run_training(
        train_path=TRAIN_PATH,
        model_path=MODEL_PATH
    )

    print("\nTraining complete.")