import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# WALMART SALES FORECASTING
# MODEL EVALUATION
# ============================================================


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE).

    Lower MAE = better model.
    """

    return mean_absolute_error(
        y_true,
        y_pred
    )


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE).

    Lower RMSE = better model.
    """

    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Zero actual sales values are excluded to avoid
    division-by-zero problems.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0

    if mask.sum() == 0:
        return np.nan

    return (
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        )
        * 100
    )


def evaluate_model(
    y_true,
    y_pred,
    model_name
):
    """
    Calculate the main evaluation metrics
    for a forecasting model.

    Returns:
        pandas DataFrame
    """

    mae = calculate_mae(
        y_true,
        y_pred
    )

    rmse = calculate_rmse(
        y_true,
        y_pred
    )

    mape = calculate_mape(
        y_true,
        y_pred
    )

    return pd.DataFrame(
        [{
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        }]
    )


# ============================================================
# BASELINE
# ============================================================

def calculate_baseline_mae(
    y_true,
    baseline_prediction
):
    """
    Calculate MAE for the baseline forecast.

    In this project, the baseline is used as the
    benchmark for evaluating XGBoost.
    """

    return calculate_mae(
        y_true,
        baseline_prediction
    )


def calculate_mae_improvement(
    baseline_mae,
    model_mae
):
    """
    Calculate percentage improvement in MAE
    compared with the baseline.

    Positive value = improvement.
    """

    if baseline_mae == 0:
        return np.nan

    return (
        (
            baseline_mae - model_mae
        )
        / baseline_mae
    ) * 100


# ============================================================
# XGBOOST EVALUATION
# ============================================================

def evaluate_xgboost(
    y_true,
    xgb_predictions,
    baseline_mae
):
    """
    Evaluate the selected XGBoost model.

    Returns the same KPI structure used
    in the project.
    """

    mae = calculate_mae(
        y_true,
        xgb_predictions
    )

    rmse = calculate_rmse(
        y_true,
        xgb_predictions
    )

    mape = calculate_mape(
        y_true,
        xgb_predictions
    )

    improvement = calculate_mae_improvement(
        baseline_mae,
        mae
    )

    return pd.DataFrame(
        [{
            "Selected_Model": "XGBoost",
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "Baseline_MAE": baseline_mae,
            "MAE_Improvement_Percent": improvement
        }]
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

def create_model_comparison(
    baseline_results,
    prophet_summary,
    xgb_summary
):
    """
    Combine results from:

    1. Baseline
    2. Prophet
    3. XGBoost

    into one comparison table.
    """

    results = pd.concat(
        [
            baseline_results,
            prophet_summary,
            xgb_summary
        ],
        ignore_index=True
    )

    return results


def select_best_model(
    model_comparison,
    metric="MAE"
):
    """
    Select the best model based on the
    lowest evaluation metric.

    Default:
        MAE
    """

    if metric not in model_comparison.columns:
        raise ValueError(
            f"{metric} is not available "
            "in model comparison."
        )

    best_index = (
        model_comparison[metric]
        .idxmin()
    )

    return model_comparison.loc[
        best_index
    ]


# ============================================================
# FINAL FORECAST SUMMARY
# ============================================================

def create_final_summary(
    selected_model,
    mae,
    rmse,
    mape,
    baseline_mae
):
    """
    Create the final executive-level
    model evaluation summary.
    """

    improvement = calculate_mae_improvement(
        baseline_mae,
        mae
    )

    return pd.DataFrame(
        [{
            "Selected_Model": selected_model,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "Baseline_MAE": baseline_mae,
            "MAE_Improvement_Percent": improvement
        }]
    )


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

def save_evaluation_results(
    results,
    path="outputs/model_evaluation.csv"
):
    """
    Save model evaluation results.
    """

    results.to_csv(
        path,
        index=False
    )


# ============================================================
# PRINT MODEL SUMMARY
# ============================================================

def print_model_summary(
    results
):
    """
    Print a clean model evaluation summary.
    """

    print("=" * 60)
    print("WALMART SALES FORECASTING")
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)

    if "Selected_Model" in results.columns:

        print(
            f"Selected Model: "
            f"{results['Selected_Model'].iloc[0]}"
        )

    if "MAE" in results.columns:

        print(
            f"MAE: "
            f"{results['MAE'].iloc[0]:,.2f}"
        )

    if "RMSE" in results.columns:

        print(
            f"RMSE: "
            f"{results['RMSE'].iloc[0]:,.2f}"
        )

    if "MAPE" in results.columns:

        print(
            f"MAPE: "
            f"{results['MAPE'].iloc[0]:,.2f}%"
        )

    if "Baseline_MAE" in results.columns:

        print(
            f"Baseline MAE: "
            f"{results['Baseline_MAE'].iloc[0]:,.2f}"
        )

    if "MAE_Improvement_Percent" in results.columns:

        print(
            f"MAE Improvement: "
            f"{results['MAE_Improvement_Percent'].iloc[0]:,.2f}%"
        )

    print("=" * 60)