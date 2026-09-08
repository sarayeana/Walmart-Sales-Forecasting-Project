"""
Prediction module for Walmart Sales Forecasting.

Loads the trained model and processed test data, generates predictions,
validates the prediction output, and saves the final forecast.
"""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd


# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TEST_DATA_PATH = PROCESSED_DIR / "test_feature_engineered.csv"
MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
OUTPUT_PATH = OUTPUT_DIR / "final_forecast.csv"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TARGET_COLUMN = "Weekly_Sales"

IDENTIFIER_COLUMNS = [
    "Store",
    "Dept",
    "Date",
]


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_prediction_input(df: pd.DataFrame) -> None:
    """
    Validate the processed test dataset before prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Processed test dataset.

    Raises
    ------
    TypeError
        If input is not a DataFrame.

    ValueError
        If required columns are missing or data is empty.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Prediction input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Prediction input dataset is empty.")

    missing_columns = [
        column
        for column in IDENTIFIER_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Prediction input is missing required identifier columns: "
            f"{missing_columns}"
        )

    if TARGET_COLUMN in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' should not be present "
            "in the prediction feature dataset."
        )


def validate_model(model) -> None:
    """
    Validate the loaded model object.

    Parameters
    ----------
    model
        Trained machine-learning model.

    Raises
    ------
    ValueError
        If the model is missing or does not provide predict().
    """

    if model is None:
        raise ValueError("Loaded model is None.")

    if not hasattr(model, "predict"):
        raise ValueError(
            "Loaded model does not provide a 'predict' method."
        )


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_test_data(
    test_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load processed test data.

    Parameters
    ----------
    test_path : Path, optional
        Path to processed test dataset.

    Returns
    -------
    pd.DataFrame
        Processed test dataset.
    """

    path = Path(test_path) if test_path else TEST_DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Processed test dataset not found: {path}"
        )

    df = pd.read_csv(path)

    validate_prediction_input(df)

    return df


def load_model(
    model_path: Optional[Path] = None,
):
    """
    Load the trained machine-learning model.

    Parameters
    ----------
    model_path : Path, optional
        Path to saved model.

    Returns
    -------
    object
        Loaded trained model.
    """

    path = Path(model_path) if model_path else MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {path}"
        )

    model = joblib.load(path)

    validate_model(model)

    return model


# ---------------------------------------------------------------------
# Feature Preparation
# ---------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """
    Prepare test features for model prediction.

    The function removes identifier columns and ensures that the
    feature columns match the columns expected by the trained model.

    Parameters
    ----------
    df : pd.DataFrame
        Processed test dataset.

    model
        Trained model.

    Returns
    -------
    pd.DataFrame
        Feature matrix ready for prediction.
    """

    features = df.drop(
        columns=IDENTIFIER_COLUMNS,
        errors="ignore",
    ).copy()

    # Remove target if it accidentally exists.
    features = features.drop(
        columns=[TARGET_COLUMN],
        errors="ignore",
    )

    # If the model stores feature names, align the dataframe to them.
    if hasattr(model, "feature_names_in_"):

        expected_features = list(model.feature_names_in_)

        missing_features = [
            column
            for column in expected_features
            if column not in features.columns
        ]

        if missing_features:
            raise ValueError(
                "Test dataset is missing model features: "
                f"{missing_features}"
            )

        # Ignore unexpected columns and preserve training order.
        features = features[expected_features]

    return features


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def generate_predictions(
    model,
    features: pd.DataFrame,
) -> pd.Series:
    """
    Generate sales predictions.

    Parameters
    ----------
    model
        Trained forecasting model.

    features : pd.DataFrame
        Feature matrix.

    Returns
    -------
    pd.Series
        Predicted Weekly_Sales values.
    """

    if features.empty:
        raise ValueError("Feature dataset is empty.")

    predictions = model.predict(features)

    predictions = pd.Series(
        predictions,
        index=features.index,
        name=TARGET_COLUMN,
    )

    if predictions.isna().any():
        raise ValueError(
            "Prediction contains missing values."
        )

    return predictions


# ---------------------------------------------------------------------
# Forecast Output
# ---------------------------------------------------------------------

def create_forecast(
    test_df: pd.DataFrame,
    predictions: pd.Series,
) -> pd.DataFrame:
    """
    Create the final forecast dataframe.

    Parameters
    ----------
    test_df : pd.DataFrame
        Original processed test dataset.

    predictions : pd.Series
        Model predictions.

    Returns
    -------
    pd.DataFrame
        Forecast containing Store, Dept, Date, and Weekly_Sales.
    """

    if len(test_df) != len(predictions):
        raise ValueError(
            "Number of predictions does not match "
            "number of test records."
        )

    forecast = test_df[
        IDENTIFIER_COLUMNS
    ].copy()

    forecast[TARGET_COLUMN] = predictions.values

    # Sales cannot logically be negative.
    forecast[TARGET_COLUMN] = forecast[TARGET_COLUMN].clip(
        lower=0
    )

    return forecast


def validate_forecast(
    forecast: pd.DataFrame,
) -> None:
    """
    Validate the final forecast dataframe.
    """

    required_columns = [
        "Store",
        "Dept",
        "Date",
        TARGET_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in forecast.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Forecast is missing required columns: "
            f"{missing_columns}"
        )

    if forecast.empty:
        raise ValueError("Forecast dataframe is empty.")

    if forecast[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Forecast contains missing Weekly_Sales values."
        )

    if (forecast[TARGET_COLUMN] < 0).any():
        raise ValueError(
            "Forecast contains negative Weekly_Sales values."
        )


def save_forecast(
    forecast: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save the final forecast to CSV.

    Parameters
    ----------
    forecast : pd.DataFrame
        Final forecast dataframe.

    output_path : Path, optional
        Destination path.

    Returns
    -------
    Path
        Saved file path.
    """

    path = Path(output_path) if output_path else OUTPUT_PATH

    validate_forecast(forecast)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast.to_csv(
        path,
        index=False,
    )

    return path


# ---------------------------------------------------------------------
# Main Prediction Pipeline
# ---------------------------------------------------------------------

def run_prediction_pipeline(
    test_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run the complete prediction pipeline.

    Steps
    -----
    1. Load processed test data.
    2. Load trained model.
    3. Prepare prediction features.
    4. Generate predictions.
    5. Create final forecast.
    6. Validate forecast.
    7. Save forecast.

    Returns
    -------
    pd.DataFrame
        Final forecast dataframe.
    """

    print("Loading processed test data...")
    test_df = load_test_data(test_path)

    print(f"Test records loaded: {len(test_df):,}")

    print("Loading trained model...")
    model = load_model(model_path)

    print("Preparing prediction features...")
    features = prepare_features(
        test_df,
        model,
    )

    print(
        f"Prediction features prepared: "
        f"{features.shape[1]} features"
    )

    print("Generating predictions...")
    predictions = generate_predictions(
        model,
        features,
    )

    print("Creating forecast...")
    forecast = create_forecast(
        test_df,
        predictions,
    )

    print("Validating forecast...")
    validate_forecast(forecast)

    print("Saving final forecast...")
    saved_path = save_forecast(
        forecast,
        output_path,
    )

    print(
        f"Forecast saved successfully: {saved_path}"
    )

    return forecast


# ---------------------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    forecast_df = run_prediction_pipeline()

    print("\nPrediction completed successfully.")
    print(f"Forecast shape: {forecast_df.shape}")
    print("\nForecast preview:")
    print(forecast_df.head())
