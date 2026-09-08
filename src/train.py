
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

from src.feature_engineering import (
get_model_features,
prepare_model_features
)

# ============================================================

# PATH CONFIGURATION

# ============================================================

PROJECT_ROOT = Path(**file**).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_FEATURE_FILE = (
PROCESSED_DATA_DIR
/ "train_feature_engineered.csv"
)

MODEL_FILE = (
MODEL_DIR
/ "best_model.joblib"
)

# ============================================================

# LOAD TRAINING DATA

# ============================================================

def load_training_data(
file_path=TRAIN_FEATURE_FILE
):
"""
Load the processed training dataset.
"""

```
df = pd.read_csv(
    file_path
)

return df
```

# ============================================================

# PREPARE TRAINING DATA

# ============================================================

def prepare_training_data(
df,
target_column="Weekly_Sales"
):
"""
Prepare X and y for model training.
"""

```
feature_columns = (
    get_model_features()
)

X = prepare_model_features(
    df,
    feature_columns
)

y = df[
    target_column
].copy()

return X, y
```

# ============================================================

# TRAIN VALIDATION SPLIT

# ============================================================

def split_training_data(
X,
y,
test_size=0.2,
random_state=42
):
"""
Split data into training and validation sets.
"""

```
X_train, X_valid, y_train, y_valid = (
    train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
)

return (
    X_train,
    X_valid,
    y_train,
    y_valid
)
```

# ============================================================

# TRAIN BASELINE MODEL

# ============================================================

def train_random_forest(
X_train,
y_train,
random_state=42
):
"""
Train a Random Forest baseline model.
"""

```
model = RandomForestRegressor(
    n_estimators=100,
    random_state=random_state,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

return model
```

# ============================================================

# TRAIN XGBOOST MODEL

# ============================================================

def train_xgboost(
X_train,
y_train,
random_state=42
):
"""
Train the XGBoost regression model.
"""

```
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

return model
```

# ============================================================

# COMPARE MODELS

# ============================================================

def compare_models(
models,
X_valid,
y_valid
):
"""
Compare multiple trained models using MAE.
"""

```
results = []

for model_name, model in models.items():

    predictions = model.predict(
        X_valid
    )

    mae = mean_absolute_error(
        y_valid,
        predictions
    )

    results.append(
        {
            "Model": model_name,
            "MAE": mae
        }
    )

results_df = pd.DataFrame(
    results
)

results_df = (
    results_df
    .sort_values(
        "MAE"
    )
    .reset_index(
        drop=True
    )
)

return results_df
```

# ============================================================

# SELECT BEST MODEL

# ============================================================

def select_best_model(
models,
comparison_results
):
"""
Select the best model based on the
lowest validation MAE.
"""

```
best_model_name = (
    comparison_results
    .iloc[0]["Model"]
)

best_model = models[
    best_model_name
]

return (
    best_model_name,
    best_model
)
```

# ============================================================

# SAVE MODEL

# ============================================================

def save_model(
model,
file_path=MODEL_FILE
):
"""
Save the trained model.
"""

```
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    file_path
)

return file_path
```

# ============================================================

# LOAD SAVED MODEL

# ============================================================

def load_model(
file_path=MODEL_FILE
):
"""
Load a previously trained model.
"""

```
model = joblib.load(
    file_path
)

return model
```

# ============================================================

# COMPLETE MODEL TRAINING PIPELINE

# ============================================================

def train_models(
df=None,
target_column="Weekly_Sales"
):
"""
Complete Walmart forecasting model
training pipeline.

```
Steps:
1. Load processed training data.
2. Prepare model features.
3. Split training and validation data.
4. Train Random Forest.
5. Train XGBoost.
6. Compare models.
7. Select best model.
8. Save best model.
"""

# --------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------

if df is None:

    df = load_training_data()


# --------------------------------------------------------
# PREPARE DATA
# --------------------------------------------------------

X, y = prepare_training_data(
    df,
    target_column
)


# --------------------------------------------------------
# SPLIT DATA
# --------------------------------------------------------

(
    X_train,
    X_valid,
    y_train,
    y_valid
) = split_training_data(
    X,
    y
)


# --------------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------------

random_forest_model = (
    train_random_forest(
        X_train,
        y_train
    )
)

xgboost_model = (
    train_xgboost(
        X_train,
        y_train
    )
)


# --------------------------------------------------------
# MODEL DICTIONARY
# --------------------------------------------------------

models = {
    "Random Forest":
    random_forest_model,

    "XGBoost":
    xgboost_model
}


# --------------------------------------------------------
# COMPARE MODELS
# --------------------------------------------------------

comparison_results = (
    compare_models(
        models,
        X_valid,
        y_valid
    )
)


# --------------------------------------------------------
# SELECT BEST MODEL
# --------------------------------------------------------

(
    best_model_name,
    best_model
) = select_best_model(
    models,
    comparison_results
)


# --------------------------------------------------------
# SAVE BEST MODEL
# --------------------------------------------------------

model_path = save_model(
    best_model
)


# --------------------------------------------------------
# RETURN RESULTS
# --------------------------------------------------------

return {
    "best_model_name":
    best_model_name,

    "best_model":
    best_model,

    "comparison_results":
    comparison_results,

    "X_valid":
    X_valid,

    "y_valid":
    y_valid,

    "model_path":
    model_path
}
```

# ============================================================

# RUN DIRECTLY

# ============================================================

if **name** == "**main**":

```
results = train_models()

print(
    "\nModel Comparison:"
)

print(
    results[
        "comparison_results"
    ]
)

print(
    f"\nBest Model: "
    f"{results['best_model_name']}"
)

print(
    f"\nModel Saved To: "
    f"{results['model_path']}"
)
```
