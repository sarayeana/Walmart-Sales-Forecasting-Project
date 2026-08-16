"""
Data validation module for the SaaS Product Analytics & Churn Prediction project.

This module validates:

    - Dataset structure
    - Required columns
    - Primary keys
    - Duplicate records
    - Missing values
    - Data types
    - Date fields
    - Numeric business rules
    - Referential integrity between datasets

The module is intentionally separate from preprocessing so that
data quality problems can be identified before transformations
are applied.
"""

from pathlib import Path

import pandas as pd

from data_loader import load_all_data


# -------------------------------------------------------------------
# Project Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# -------------------------------------------------------------------
# Expected Dataset Structure
# -------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "train": [
        "Store",
        "Dept",
        "Date",
        "Weekly_Sales",
        "IsHoliday",
    ],
    "stores": [
        "Store",
        "Type",
        "Size",
    ],
    "features": [
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
        "IsHoliday",
    ],
    "test": [
        "Store",
        "Dept",
        "Date",
        "IsHoliday",
    ],
}


# -------------------------------------------------------------------
# Primary Keys
# -------------------------------------------------------------------

PRIMARY_KEYS = {
    "train": ["Store", "Dept", "Date"],
    "stores": ["Store"],
    "features": ["Store", "Date"],
    "test": ["Store", "Dept", "Date"],
}


# -------------------------------------------------------------------
# Date Columns
# -------------------------------------------------------------------

DATE_COLUMNS = {
    "train": [
        "Date",
    ],
    "stores": [],
    "features": [
        "Date",
    ],
    "test": [
        "Date",
    ],
}


# -------------------------------------------------------------------
# Required Non-Null Columns
# -------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "train": [
        "Store",
        "Dept",
        "Date",
        "Weekly_Sales",
        "IsHoliday",
    ],
    "stores": [
        "Store",
        "Type",
        "Size",
    ],
    "features": [
        "Store",
        "Date",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "IsHoliday",
    ],
    "test": [
        "Store",
        "Dept",
        "Date",
        "IsHoliday",
    ],
}


# -------------------------------------------------------------------
# Required Column Validation
# -------------------------------------------------------------------

def validate_columns(
    data: dict[str, pd.DataFrame]
) -> dict[str, dict]:
    """
    Validate that all expected columns exist.

    Returns
    -------
    dict
        Validation results for each dataset.
    """

    results = {}

    for dataset_name, expected_columns in EXPECTED_COLUMNS.items():

        df = data[dataset_name]

        actual_columns = set(df.columns)
        expected_columns_set = set(expected_columns)

        missing_columns = sorted(
            expected_columns_set - actual_columns
        )

        unexpected_columns = sorted(
            actual_columns - expected_columns_set
        )

        results[dataset_name] = {
            "passed": len(missing_columns) == 0,
            "missing_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
        }

    return results



# -------------------------------------------------------------------
# Primary Key Validation
# -------------------------------------------------------------------

def validate_primary_keys(
    data: dict[str, pd.DataFrame]
) -> dict[str, dict]:
    """
    Validate primary keys for all Walmart datasets.

    Supports both single-column and composite primary keys.

    Returns
    -------
    dict[str, dict]
        Validation results for each dataset.
    """

    results = {}

    for dataset_name, primary_key in PRIMARY_KEYS.items():

        df = data[dataset_name]

        # -----------------------------------------------------------
        # Normalize primary key to a list
        # -----------------------------------------------------------

        if isinstance(primary_key, str):
            primary_key_columns = [primary_key]
        else:
            primary_key_columns = primary_key

        # -----------------------------------------------------------
        # Count null values across primary-key columns
        # -----------------------------------------------------------

        null_count = int(
            df[primary_key_columns]
            .isna()
            .any(axis=1)
            .sum()
        )

        # -----------------------------------------------------------
        # Count duplicate primary-key combinations
        # -----------------------------------------------------------

        duplicate_count = int(
            df.duplicated(
                subset=primary_key_columns
            ).sum()
        )

        # -----------------------------------------------------------
        # Validation status
        # -----------------------------------------------------------

        passed = (
            null_count == 0
            and duplicate_count == 0
        )

        results[dataset_name] = {
            "primary_key": primary_key_columns,
            "duplicate_count": duplicate_count,
            "null_count": null_count,
            "passed": passed,
        }

    return results

# -------------------------------------------------------------------
# Missing Value Validation
# -------------------------------------------------------------------

def validate_missing_values(
    data: dict[str, pd.DataFrame]
) -> dict[str, dict]:
    """
    Calculate missing values for every dataset and
    identify missing values in required columns.
    """

    results = {}

    for dataset_name, df in data.items():

        total_missing = int(
            df.isna().sum().sum()
        )

        required_missing = {}

        for column in REQUIRED_COLUMNS[dataset_name]:

            missing_count = int(
                df[column].isna().sum()
            )

            required_missing[column] = missing_count

        required_columns_passed = all(
            count == 0
            for count in required_missing.values()
        )

        results[dataset_name] = {
            "total_missing_values": total_missing,
            "required_column_missing": required_missing,
            "passed": required_columns_passed,
        }

    return results


# -------------------------------------------------------------------
# Date Validation
# -------------------------------------------------------------------

def validate_dates(
    data: dict[str, pd.DataFrame]
) -> dict[str, dict]:
    """
    Validate date columns.

    Checks:

        - Invalid date values
        - Future dates
        - Negative date relationships
    """

    results = {}

    current_date = pd.Timestamp.now()

    for dataset_name, columns in DATE_COLUMNS.items():

        df = data[dataset_name]

        dataset_results = {}

        for column in columns:

            converted_dates = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            invalid_dates = int(
                converted_dates.isna().sum()
                - df[column].isna().sum()
            )

            future_dates = int(
                (converted_dates > current_date).sum()
            )

            dataset_results[column] = {
                "invalid_dates": invalid_dates,
                "future_dates": future_dates,
                "passed": (
                    invalid_dates == 0
                    and future_dates == 0
                ),
            }

        results[dataset_name] = dataset_results

    return results


# -------------------------------------------------------------------
# Numeric Business Rule Validation
# -------------------------------------------------------------------

def validate_numeric_rules(
    data: dict[str, pd.DataFrame]
) -> dict[str, dict]:
    """
    Validate numeric business rules for Walmart datasets.

    Returns
    -------
    dict[str, dict]
        Validation results containing counts of invalid values.
    """

    results = {}

    # ---------------------------------------------------------------
    # Train Dataset
    # ---------------------------------------------------------------

    train = data["train"]

    negative_weekly_sales = int(
        (train["Weekly_Sales"] < 0).sum()
    )

    invalid_store = int(
        (train["Store"] <= 0).sum()
    )

    invalid_department = int(
        (train["Dept"] <= 0).sum()
    )

    train_result = {
        "negative_weekly_sales": negative_weekly_sales,
        "invalid_store": invalid_store,
        "invalid_department": invalid_department,
    }

    # ---------------------------------------------------------------
    # Stores Dataset
    # ---------------------------------------------------------------

    stores = data["stores"]

    invalid_store_id = int(
        (stores["Store"] <= 0).sum()
    )

    invalid_store_size = int(
        (stores["Size"] <= 0).sum()
    )

    stores_result = {
        "invalid_store_id": invalid_store_id,
        "invalid_store_size": invalid_store_size,
    }

    # ---------------------------------------------------------------
    # Features Dataset
    # ---------------------------------------------------------------

    features = data["features"]

    negative_temperature = int(
        (features["Temperature"] < 0).sum()
    )

    negative_fuel_price = int(
        (features["Fuel_Price"] < 0).sum()
    )

    negative_cpi = int(
        (features["CPI"] < 0).sum()
    )

    negative_unemployment = int(
        (features["Unemployment"] < 0).sum()
    )

    markdown_columns = [
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
    ]

    negative_markdowns = {
        column: int(
            (features[column] < 0).sum()
        )
        for column in markdown_columns
    }

    features_result = {
        "negative_temperature": negative_temperature,
        "negative_fuel_price": negative_fuel_price,
        "negative_cpi": negative_cpi,
        "negative_unemployment": negative_unemployment,
        "negative_markdowns": negative_markdowns,
    }

    # ---------------------------------------------------------------
    # Test Dataset
    # ---------------------------------------------------------------

    test = data["test"]

    invalid_test_store = int(
        (test["Store"] <= 0).sum()
    )

    invalid_test_department = int(
        (test["Dept"] <= 0).sum()
    )

    test_result = {
        "invalid_store": invalid_test_store,
        "invalid_department": invalid_test_department,
    }

    # ---------------------------------------------------------------
    # Store Results
    # ---------------------------------------------------------------

    results["train"] = train_result
    results["stores"] = stores_result
    results["features"] = features_result
    results["test"] = test_result

    return results



# -------------------------------------------------------------------
# Referential Integrity Validation
# -------------------------------------------------------------------

def validate_referential_integrity(
    data: dict[str, pd.DataFrame]
) -> dict[str, dict]:
    """
    Validate relationships between Walmart datasets.

    Checks whether Store values appearing in train, features,
    and test exist in the stores dataset.
    """

    stores = data["stores"]

    store_ids = set(
        stores["Store"].dropna()
    )

    results = {}

    for dataset_name in [
        "train",
        "features",
        "test",
    ]:

        df = data[dataset_name]

        missing_store_ids = int(
            (~df["Store"].isin(store_ids)).sum()
        )

        results[dataset_name] = {
            "missing_store_ids": missing_store_ids,
            "passed": missing_store_ids == 0,
        }

    return results


# -------------------------------------------------------------------
# Duplicate Row Validation
# -------------------------------------------------------------------

def validate_duplicate_rows(
    data: dict[str, pd.DataFrame]
) -> dict[str, int]:
    """
    Count completely duplicated rows in every dataset.
    """

    return {
        dataset_name: int(df.duplicated().sum())
        for dataset_name, df in data.items()
    }


# -------------------------------------------------------------------
# Full Validation
# -------------------------------------------------------------------

def run_validation(
    data: dict[str, pd.DataFrame]
) -> dict:
    """
    Run the complete Walmart dataset validation suite.

    Returns
    -------
    dict
        Complete validation results.
    """

    return {
        "columns": validate_columns(data),
        "primary_keys": validate_primary_keys(data),
        "missing_values": validate_missing_values(data),
        "dates": validate_dates(data),
        "numeric_rules": validate_numeric_rules(data),
        "referential_integrity": validate_referential_integrity(data),
        "duplicates": validate_duplicate_rows(data),
    }


# -------------------------------------------------------------------
# Validation Report
# -------------------------------------------------------------------

def print_validation_report(
    results: dict
) -> None:
    """
    Print a readable Walmart dataset validation report.
    """

    print("\n")
    print("=" * 75)
    print("WALMART SALES DATA VALIDATION REPORT")
    print("=" * 75)

    # ---------------------------------------------------------------
    # Column Validation
    # ---------------------------------------------------------------

    print("\n1. COLUMN VALIDATION")
    print("-" * 75)

    for dataset, result in results["columns"].items():

        status = "PASS" if result["passed"] else "FAIL"

        print(f"{dataset:<25} {status}")

        if result["missing_columns"]:
            print(
                f"  Missing: {result['missing_columns']}"
            )

        if result["unexpected_columns"]:
            print(
                f"  Unexpected: {result['unexpected_columns']}"
            )

    # ---------------------------------------------------------------
    # Primary Keys
    # ---------------------------------------------------------------

    print("\n2. PRIMARY KEY VALIDATION")
    print("-" * 75)

    for dataset, result in results["primary_keys"].items():

        status = "PASS" if result["passed"] else "FAIL"

        print(
            f"{dataset:<25} {status} "
            f"| duplicates={result['duplicate_count']} "
            f"| nulls={result['null_count']}"
        )

    # ---------------------------------------------------------------
    # Missing Values
    # ---------------------------------------------------------------

    print("\n3. MISSING VALUE VALIDATION")
    print("-" * 75)

    for dataset, result in results["missing_values"].items():

        status = "PASS" if result["passed"] else "CHECK"

        print(
            f"{dataset:<25} {status} "
            f"| total_missing={result['total_missing_values']}"
        )

        for column, count in result[
            "required_column_missing"
        ].items():

            if count > 0:
                print(
                    f"  {column}: {count} missing"
                )

    # ---------------------------------------------------------------
    # Date Validation
    # ---------------------------------------------------------------

    print("\n4. DATE VALIDATION")
    print("-" * 75)

    for dataset, columns in results["dates"].items():

        for column, result in columns.items():

            status = (
                "PASS"
                if result["passed"]
                else "FAIL"
            )

            print(
                f"{dataset}.{column:<35} {status} "
                f"| invalid={result['invalid_dates']} "
                f"| future={result['future_dates']}"
            )

    # ---------------------------------------------------------------
    # Numeric Rules
    # ---------------------------------------------------------------

    print("\n5. NUMERIC BUSINESS RULES")
    print("-" * 75)

    for dataset, rules in results["numeric_rules"].items():

        for rule, count in rules.items():

            status = (
                "PASS"
                if count == 0
                else "FAIL"
            )

            print(
                f"{dataset}.{rule:<45} "
                f"{status} | violations={count}"
            )

    # ---------------------------------------------------------------
    # Referential Integrity
    # ---------------------------------------------------------------

    print("\n6. REFERENTIAL INTEGRITY")
    print("-" * 75)

    for dataset, result in results[
        "referential_integrity"
    ].items():

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"{dataset:<25} {status} "
            f"| missing_stores={result['missing_store_ids']}"
        )

    # ---------------------------------------------------------------
    # Duplicate Rows
    # ---------------------------------------------------------------

    print("\n7. DUPLICATE ROWS")
    print("-" * 75)

    for dataset, count in results[
        "duplicates"
    ].items():

        status = (
            "PASS"
            if count == 0
            else "CHECK"
        )

        print(
            f"{dataset:<25} {status} "
            f"| duplicates={count}"
        )

    print("\n" + "=" * 75)
    print("VALIDATION COMPLETE")
    print("=" * 75)


# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------

if __name__ == "__main__":

    print(
        "Loading Walmart datasets..."
    )

    data = load_all_data()

    results = run_validation(data)

    print_validation_report(results)