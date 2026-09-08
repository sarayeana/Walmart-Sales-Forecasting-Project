
import numpy as np
import pandas as pd

from sklearn.metrics import (
mean_absolute_error,
mean_squared_error
)

# ============================================================

# MEAN ABSOLUTE ERROR

# ============================================================

def calculate_mae(
y_true,
y_pred
):
"""
Calculate Mean Absolute Error.
"""

```
return mean_absolute_error(
    y_true,
    y_pred
)
```

# ============================================================

# ROOT MEAN SQUARED ERROR

# ============================================================

def calculate_rmse(
y_true,
y_pred
):
"""
Calculate Root Mean Squared Error.
"""

```
mse = mean_squared_error(
    y_true,
    y_pred
)

rmse = np.sqrt(
    mse
)

return rmse
```

# ============================================================

# MEAN ABSOLUTE PERCENTAGE ERROR

# ============================================================

def calculate_mape(
y_true,
y_pred
):
"""
Calculate Mean Absolute Percentage Error.

```
Zero actual sales values are excluded to prevent
division by zero.
"""

y_true = np.asarray(
    y_true
)

y_pred = np.asarray(
    y_pred
)

non_zero_mask = (
    y_true != 0
)

if non_zero_mask.sum() == 0:

    return np.nan

y_true_non_zero = (
    y_true[
        non_zero_mask
    ]
)

y_pred_non_zero = (
    y_pred[
        non_zero_mask
    ]
)

mape = np.mean(
    np.abs(
        (
            y_true_non_zero
            -
            y_pred_non_zero
        )
        /
        y_true_non_zero
    )
)

return (
    mape
    *
    100
)
```

# ============================================================

# BASELINE PREDICTIONS

# ============================================================

def create_baseline_predictions(
y_train,
number_of_predictions
):
"""
Create a simple baseline prediction.

```
Uses the average training sales value as
the prediction for every validation record.
"""

baseline_value = (
    np.mean(
        y_train
    )
)

predictions = np.full(
    shape=number_of_predictions,
    fill_value=baseline_value
)

return predictions
```

# ============================================================

# BASELINE MAE

# ============================================================

def calculate_baseline_mae(
y_true,
y_train
):
"""
Calculate MAE for the baseline model.
"""

```
baseline_predictions = (
    create_baseline_predictions(
        y_train,
        len(y_true)
    )
)

baseline_mae = calculate_mae(
    y_true,
    baseline_predictions
)

return baseline_mae
```

# ============================================================

# MAE IMPROVEMENT

# ============================================================

def calculate_mae_improvement(
baseline_mae,
model_mae
):
"""
Calculate MAE improvement percentage compared
with the baseline.
"""

```
if baseline_mae == 0:

    return np.nan

improvement = (
    (
        baseline_mae
        -
        model_mae
    )
    /
    baseline_mae
)

return (
    improvement
    *
    100
)
```

# ============================================================

# COMPLETE MODEL EVALUATION

# ============================================================

def evaluate_model(
model,
X,
y_true,
y_train=None
):
"""
Evaluate a trained model.

```
Calculates:
- MAE
- RMSE
- MAPE
- Baseline MAE
- MAE Improvement Percentage
"""

# --------------------------------------------------------
# MODEL PREDICTIONS
# --------------------------------------------------------

predictions = model.predict(
    X
)


# --------------------------------------------------------
# MODEL METRICS
# --------------------------------------------------------

mae = calculate_mae(
    y_true,
    predictions
)

rmse = calculate_rmse(
    y_true,
    predictions
)

mape = calculate_mape(
    y_true,
    predictions
)


# --------------------------------------------------------
# BASELINE METRICS
# --------------------------------------------------------

baseline_mae = np.nan
mae_improvement_percent = np.nan

if y_train is not None:

    baseline_mae = (
        calculate_baseline_mae(
            y_true,
            y_train
        )
    )

    mae_improvement_percent = (
        calculate_mae_improvement(
            baseline_mae,
            mae
        )
    )


# --------------------------------------------------------
# RETURN RESULTS
# --------------------------------------------------------

results = {
    "MAE":
    mae,

    "RMSE":
    rmse,

    "MAPE":
    mape,

    "Baseline_MAE":
    baseline_mae,

    "MAE_Improvement_Percent":
    mae_improvement_percent
}

return results
```

# ============================================================

# EVALUATION SUMMARY

# ============================================================

def create_evaluation_summary(
model_name,
evaluation_results
):
"""
Create a one-row evaluation DataFrame.
"""

```
summary = pd.DataFrame(
    [
        {
            "Selected_Model":
            model_name,

            "MAE":
            evaluation_results[
                "MAE"
            ],

            "RMSE":
            evaluation_results[
                "RMSE"
            ],

            "MAPE":
            evaluation_results[
                "MAPE"
            ],

            "Baseline_MAE":
            evaluation_results[
                "Baseline_MAE"
            ],

            "MAE_Improvement_Percent":
            evaluation_results[
                "MAE_Improvement_Percent"
            ]
        }
    ]
)

return summary
```

# ============================================================

# EVALUATE MULTIPLE MODELS

# ============================================================

def evaluate_multiple_models(
models,
X,
y_true,
y_train=None
):
"""
Evaluate multiple models and return
a comparison DataFrame.
"""

```
results = []

for model_name, model in models.items():

    evaluation = evaluate_model(
        model=model,
        X=X,
        y_true=y_true,
        y_train=y_train
    )

    results.append(
        {
            "Model":
            model_name,

            "MAE":
            evaluation[
                "MAE"
            ],

            "RMSE":
            evaluation[
                "RMSE"
            ],

            "MAPE":
            evaluation[
                "MAPE"
            ],

            "Baseline_MAE":
            evaluation[
                "Baseline_MAE"
            ],

            "MAE_Improvement_Percent":
            evaluation[
                "MAE_Improvement_Percent"
            ]
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

# PRINT EVALUATION RESULTS

# ============================================================

def print_evaluation_results(
evaluation_results
):
"""
Print model evaluation metrics.
"""

```
print(
    "\nModel Evaluation Results"
)

print(
    "-" * 40
)

for metric, value in (
    evaluation_results
    .items()
):

    if pd.isna(value):

        print(
            f"{metric}: N/A"
        )

    else:

        print(
            f"{metric}: "
            f"{value:,.4f}"
        )
```

# ============================================================

# SAVE EVALUATION RESULTS

# ============================================================

def save_evaluation_results(
evaluation_summary,
file_path
):
"""
Save evaluation results to CSV.
"""

```
evaluation_summary.to_csv(
    file_path,
    index=False
)

return file_path
```
