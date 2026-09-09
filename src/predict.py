"""
Prediction module for Walmart Sales Forecasting.

Loads the trained model and processed test data, prepares the exact
model features, generates predictions, validates the forecast output,
and saves the final forecast.
"""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from src.feature_engineering import (
get_model_features,
prepare_model_features,
)

# ============================================================

# PROJECT PATHS

# ============================================================

PROJECT_ROOT = Path(**file**).resolve().parents[1]

PROCESSED_DIR = (
PROJECT_ROOT
/ "data"
/ "processed"
)

MODEL_DIR = (
PROJECT_ROOT
/ "models"
)

OUTPUT_DIR = (
PROJECT_ROOT
/ "outputs"
)

TEST_DATA_PATH = (
PROCESSED_DIR
/ "test_feature_engineered.csv"
)

MODEL_PATH = (
MODEL_DIR
/ "best_model.joblib"
)

OUTPUT_PATH = (
OUTPUT_DIR
/ "final_forecast.csv"
)

# ============================================================

# CONFIGURATION

# ============================================================

TARGET_COLUMN = "Weekly_Sales"

IDENTIFIER_COLUMNS = [
"Store",
"Dept",
"Date",
]

# ============================================================

# INPUT VALIDATION

# ============================================================

def validate_prediction_input(
df: pd.DataFrame,
) -> None:
"""
Validate the processed test dataset.
"""

```
if not isinstance(
    df,
    pd.DataFrame
):
    raise TypeError(
        "Prediction input must be a pandas DataFrame."
    )

if df.empty:
    raise ValueError(
        "Prediction input dataset is empty."
    )

missing_columns = [
    column
    for column in IDENTIFIER_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Prediction input is missing required "
        f"identifier columns: {missing_columns}"
    )
```

def validate_model(
model,
) -> None:
"""
Validate the loaded trained model.
"""

```
if model is None:
    raise ValueError(
        "Loaded model is None."
    )

if not hasattr(
    model,
    "predict"
):
    raise ValueError(
        "Loaded model does not provide "
        "a 'predict' method."
    )
```

# ============================================================

# LOAD TEST DATA

# ============================================================

def load_test_data(
test_path: Optional[Path] = None,
) -> pd.DataFrame:
"""
Load processed test feature data.
"""

```
path = (
    Path(test_path)
    if test_path
    else TEST_DATA_PATH
)

if not path.exists():

    raise FileNotFoundError(
        "Processed test dataset not found: "
        f"{path}"
    )

df = pd.read_csv(
    path
)

validate_prediction_input(
    df
)

if "Date" in df.columns:

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

return df
```

# ============================================================

# LOAD MODEL

# ============================================================

def load_trained_model(
model_path: Optional[Path] = None,
):
"""
Load the trained forecasting model.
"""

```
path = (
    Path(model_path)
    if model_path
    else MODEL_PATH
)

if not path.exists():

    raise FileNotFoundError(
        "Trained model not found: "
        f"{path}"
    )

model = joblib.load(
    path
)

validate_model(
    model
)

return model
```

# ============================================================

# PREPARE MODEL FEATURES

# ============================================================

def prepare_prediction_features(
df: pd.DataFrame,
) -> pd.DataFrame:
"""
Prepare the exact feature columns required
by the trained Walmart forecasting model.

```
Uses the same feature configuration used
during model training.
"""

feature_columns = (
    get_model_features()
)

features = (
    prepare_model_features(
        df,
        feature_columns
    )
)

return features
```

# ============================================================

# GENERATE PREDICTIONS

# ============================================================

def generate_predictions(
model,
features: pd.DataFrame,
) -> pd.Series:
"""
Generate Weekly_Sales predictions.
"""

```
if features.empty:

    raise ValueError(
        "Prediction feature dataset is empty."
    )

predictions = model.predict(
    features
)

predictions = pd.Series(
    predictions,
    index=features.index,
    name=TARGET_COLUMN
)

if predictions.isna().any():

    raise ValueError(
        "Prediction contains missing values."
    )

return predictions
```

# ============================================================

# CREATE FORECAST

# ============================================================

def create_forecast(
test_df: pd.DataFrame,
predictions: pd.Series,
) -> pd.DataFrame:
"""
Create the final forecast dataframe.

```
Output columns:

Store
Dept
Date
Weekly_Sales
"""

if len(test_df) != len(predictions):

    raise ValueError(
        "Number of predictions does not match "
        "the number of test records."
    )

forecast = test_df[
    IDENTIFIER_COLUMNS
].copy()

forecast[TARGET_COLUMN] = (
    predictions.values
)

# Weekly sales should not be negative.
forecast[TARGET_COLUMN] = (
    forecast[TARGET_COLUMN]
    .clip(
        lower=0
    )
)

return forecast
```

# ============================================================

# FORECAST VALIDATION

# ============================================================

def validate_forecast(
forecast: pd.DataFrame,
) -> None:
"""
Validate the final forecast dataframe.
"""

```
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
        "Forecast is missing required columns: "
        f"{missing_columns}"
    )

if forecast.empty:

    raise ValueError(
        "Forecast dataframe is empty."
    )

if forecast[
    TARGET_COLUMN
].isna().any():

    raise ValueError(
        "Forecast contains missing "
        "Weekly_Sales values."
    )

if (
    forecast[TARGET_COLUMN] < 0
).any():

    raise ValueError(
        "Forecast contains negative "
        "Weekly_Sales values."
    )
```

# ============================================================

# SAVE FORECAST

# ============================================================

def save_forecast(
forecast: pd.DataFrame,
output_path: Optional[Path] = None,
) -> Path:
"""
Save the final forecast CSV.
"""

```
path = (
    Path(output_path)
    if output_path
    else OUTPUT_PATH
)

validate_forecast(
    forecast
)

path.parent.mkdir(
    parents=True,
    exist_ok=True
)

forecast.to_csv(
    path,
    index=False
)

return path
```

# ============================================================

# COMPLETE PREDICTION PIPELINE

# ============================================================

def run_prediction_pipeline(
test_path: Optional[Path] = None,
model_path: Optional[Path] = None,
output_path: Optional[Path] = None,
) -> pd.DataFrame:
"""
Run the complete Walmart forecasting pipeline.

```
Steps:
1. Load processed test data.
2. Validate test data.
3. Load trained model.
4. Prepare exact model features.
5. Generate predictions.
6. Create forecast.
7. Validate forecast.
8. Save forecast.

Returns
-------
pd.DataFrame
    Final forecast dataframe.
"""

# --------------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------------

print(
    "Loading processed test data..."
)

test_df = load_test_data(
    test_path
)

print(
    f"Test records loaded: "
    f"{len(test_df):,}"
)


# --------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------

print(
    "Loading trained model..."
)

model = load_trained_model(
    model_path
)


# --------------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------------

print(
    "Preparing model features..."
)

features = (
    prepare_prediction_features(
        test_df
    )
)

print(
    f"Prediction features prepared: "
    f"{features.shape[1]}"
)


# --------------------------------------------------------
# GENERATE PREDICTIONS
# --------------------------------------------------------

print(
    "Generating predictions..."
)

predictions = (
    generate_predictions(
        model,
        features
    )
)


# --------------------------------------------------------
# CREATE FORECAST
# --------------------------------------------------------

print(
    "Creating final forecast..."
)

forecast = create_forecast(
    test_df,
    predictions
)


# --------------------------------------------------------
# VALIDATE FORECAST
# --------------------------------------------------------

print(
    "Validating forecast..."
)

validate_forecast(
    forecast
)


# --------------------------------------------------------
# SAVE FORECAST
# --------------------------------------------------------

print(
    "Saving final forecast..."
)

saved_path = save_forecast(
    forecast,
    output_path
)

print(
    "Forecast saved successfully:"
)

print(
    saved_path
)


return forecast
```

# ============================================================

# SCRIPT ENTRY POINT

# ============================================================

if **name** == "**main**":

```
forecast_df = (
    run_prediction_pipeline()
)

print(
    "\nPrediction completed successfully."
)

print(
    f"Forecast shape: "
    f"{forecast_df.shape}"
)

print(
    "\nForecast preview:"
)

print(
    forecast_df.head()
)
```

