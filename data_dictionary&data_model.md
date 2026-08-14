# Walmart Sales Forecasting — Data Dictionary & Data Model

## 1. Project Overview

This project uses the **Walmart Recruiting – Store Sales Forecasting** dataset.

The objective is to forecast weekly sales for individual Walmart departments within individual stores.

The original dataset contains historical sales information for **45 Walmart stores**, with multiple departments per store. Additional store-level and economic information is provided through the `features.csv` and `stores.csv` datasets.

The core datasets are:

```text
train.csv
test.csv
features.csv
stores.csv
```

The project combines these datasets and creates additional time-series and machine-learning features for the XGBoost forecasting model.

---

# 2. Dataset Architecture

The original dataset can be represented as:

```text
                    ┌──────────────────┐
                    │    stores.csv    │
                    │                  │
                    │ Store (PK)       │
                    │ Type             │
                    │ Size             │
                    └────────┬─────────┘
                             │
                             │ 1
                             │
                             │ N
                    ┌────────▼─────────┐
                    │    features.csv  │
                    │                  │
                    │ Store            │
                    │ Date             │
                    │ Temperature      │
                    │ Fuel_Price       │
                    │ MarkDown1-5      │
                    │ CPI              │
                    │ Unemployment     │
                    │ IsHoliday        │
                    └────────┬─────────┘
                             │
                             │
              ┌──────────────┴──────────────┐
              │                             │
              │ Store + Date                │
              │                             │
       ┌──────▼───────┐              ┌──────▼───────┐
       │   train.csv  │              │    test.csv  │
       │              │              │              │
       │ Store        │              │ Store        │
       │ Dept         │              │ Dept         │
       │ Date         │              │ Date         │
       │ Weekly_Sales │              │ IsHoliday    │
       │ IsHoliday    │              │              │
       └──────────────┘              └──────────────┘
```

---

# 3. Dataset Summary

| Dataset        | Purpose                                    |  Rows in Project | Main Key              |
| -------------- | ------------------------------------------ | ---------------: | --------------------- |
| `train.csv`    | Historical sales used for model training   |          421,570 | `Store + Dept + Date` |
| `test.csv`     | Future periods requiring predictions       |          115,064 | `Store + Dept + Date` |
| `features.csv` | Weather, economic and markdown information | Store/date-level | `Store + Date`        |
| `stores.csv`   | Store-level information                    |               45 | `Store`               |

The standard Walmart forecasting dataset contains 45 stores, and the forecasting task is to predict sales for each store/department/date combination in the test data.

---

# 4. `stores.csv`

## 4.1 Purpose

`stores.csv` contains information about the Walmart stores.

It describes the store type and physical/operational size of each store. The dataset contains information for 45 stores.

---

## 4.2 Columns

| Column  | Data Type   | Key    | Description                     |
| ------- | ----------- | ------ | ------------------------------- |
| `Store` | Integer     | **PK** | Unique Walmart store identifier |
| `Type`  | Categorical | —      | Store type: A, B, or C          |
| `Size`  | Integer     | —      | Store size                      |

---

## 4.3 Primary Key

```text
Store
```

Example:

```text
Store
-----
1
2
3
...
45
```

Each store appears once in `stores.csv`.

---

## 4.4 Column Details

### `Store`

Identifies the Walmart store.

Example:

```text
1
2
3
45
```

**Role:** Primary Key.

---

### `Type`

Categorizes the store.

Possible values:

```text
A
B
C
```

Store types represent different store formats.

---

### `Size`

Represents the size of the Walmart store.

A larger store generally has greater physical capacity and potentially different sales behavior.

---

# 5. `train.csv`

## 5.1 Purpose

`train.csv` contains historical weekly sales.

It is the primary transactional/time-series dataset used to train the forecasting model.

Your processed training dataset contains:

```text
421,570 rows
```

---

## 5.2 Columns

| Column         | Data Type | Key          | Target  | Description                   |
| -------------- | --------- | ------------ | ------- | ----------------------------- |
| `Store`        | Integer   | Composite PK | No      | Store identifier              |
| `Dept`         | Integer   | Composite PK | No      | Department identifier         |
| `Date`         | Date      | Composite PK | No      | Week of observation           |
| `Weekly_Sales` | Float     | —            | **Yes** | Weekly sales                  |
| `IsHoliday`    | Boolean   | —            | No      | Whether the week is a holiday |

---

# 6. Primary Key of `train.csv`

There is no single-column primary key.

The practical unique identifier is:

```text
(Store, Dept, Date)
```

Therefore:

```text
PRIMARY KEY = Store + Dept + Date
```

Example:

```text
Store   Dept   Date
1       1      2010-02-05
1       1      2010-02-12
1       1      2010-02-19
1       2      2010-02-05
1       2      2010-02-12
```

A store can have many departments.

A department can appear across many stores.

Therefore:

```text
Store + Dept + Date
```

uniquely identifies a sales observation.

---

# 7. `train.csv` Column Details

## `Store`

Identifies the Walmart store.

```text
1 → Store 1
2 → Store 2
...
45 → Store 45
```

---

## `Dept`

Identifies the department within the store.

The same department number can exist in multiple stores.

Therefore:

```text
Dept = 1
```

does **not** uniquely identify a record.

Instead:

```text
Store + Dept + Date
```

is required.

---

## `Date`

Represents the week associated with the sales observation.

Example:

```text
2010-02-05
2010-02-12
2010-02-19
```

This column is critical for time-series analysis.

It was used to create:

```text
Year
Quarter
Month
Week
Day
DayOfWeek
IsWeekend
```

and cyclical features.

---

## `Weekly_Sales`

Historical weekly sales for a specific:

```text
Store
+
Department
+
Week
```

This is the **target variable** for the machine-learning model.

In Python:

```python
y = train["Weekly_Sales"]
```

---

## `IsHoliday`

Boolean indicator identifying whether the week contains a special holiday.

Example:

```text
False
True
```

Holiday information is important because holiday periods can significantly affect Walmart sales.

---

# 8. `test.csv`

## 8.1 Purpose

`test.csv` contains the future observations for which weekly sales must be predicted.

It has the identifying information but does not contain the target:

```text
Weekly_Sales
```

The original competition requires predictions for each:

```text
Store + Department + Date
```

combination in the test dataset.

---

## 8.2 Columns

| Column      | Data Type | Key          | Description           |
| ----------- | --------- | ------------ | --------------------- |
| `Store`     | Integer   | Composite PK | Store identifier      |
| `Dept`      | Integer   | Composite PK | Department identifier |
| `Date`      | Date      | Composite PK | Forecast week         |
| `IsHoliday` | Boolean   | —            | Holiday indicator     |

---

## 8.3 Primary Key

```text
Store + Dept + Date
```

Example:

```text
Store   Dept   Date
1       1      2012-11-02
1       1      2012-11-09
1       1      2012-11-16
```

---

# 9. Test Dataset in This Project

Your project contains:

```text
115,064 rows
```

The forecast period is:

```text
2012-11-02
        ↓
2013-07-26
```

The important point is that `Weekly_Sales` is unavailable for these observations.

Therefore:

```text
Historical Sales
      ↓
Feature Engineering
      ↓
XGBoost
      ↓
Predicted Weekly Sales
```

---

# 10. `features.csv`

## 10.1 Purpose

`features.csv` provides additional information associated with a store and week.

It contains:

* Temperature
* Fuel price
* Markdown information
* CPI
* Unemployment
* Holiday information

These variables provide external factors that may influence sales.

---

# 11. `features.csv` Columns

| Column         | Data Type | Key          | Description                  |
| -------------- | --------- | ------------ | ---------------------------- |
| `Store`        | Integer   | Composite PK | Store identifier             |
| `Date`         | Date      | Composite PK | Week                         |
| `Temperature`  | Float     | —            | Regional average temperature |
| `Fuel_Price`   | Float     | —            | Regional fuel price          |
| `MarkDown1`    | Float     | —            | Promotional markdown measure |
| `MarkDown2`    | Float     | —            | Promotional markdown measure |
| `MarkDown3`    | Float     | —            | Promotional markdown measure |
| `MarkDown4`    | Float     | —            | Promotional markdown measure |
| `MarkDown5`    | Float     | —            | Promotional markdown measure |
| `CPI`          | Float     | —            | Consumer Price Index         |
| `Unemployment` | Float     | —            | Regional unemployment rate   |
| `IsHoliday`    | Boolean   | —            | Holiday indicator            |

---

# 12. Primary Key of `features.csv`

The appropriate composite key is:

```text
Store + Date
```

Example:

```text
Store   Date
1       2010-02-05
1       2010-02-12
1       2010-02-19
2       2010-02-05
2       2010-02-12
```

Each record represents a store/week combination.

---

# 13. `features.csv` Column Details

## `Store`

Connects the feature record to a Walmart store.

Relationship:

```text
features.Store
       ↓
stores.Store
```

---

## `Date`

Identifies the week to which the external features apply.

This allows:

```text
Store + Date
```

to connect the features to the sales data.

---

## `Temperature`

Average regional temperature for the week.

It can capture weather-related changes in customer behavior and demand.

---

## `Fuel_Price`

Fuel price for the relevant region/week.

Fuel prices can potentially influence consumer spending and transportation-related behavior.

---

## `MarkDown1`

An anonymized promotional markdown variable.

---

## `MarkDown2`

An anonymized promotional markdown variable.

---

## `MarkDown3`

An anonymized promotional markdown variable.

---

## `MarkDown4`

An anonymized promotional markdown variable.

---

## `MarkDown5`

An anonymized promotional markdown variable.

The markdown fields represent promotional activity. The original dataset notes that markdown information is incomplete and contains many missing values.

---

## `CPI`

Consumer Price Index.

This represents the broader economic price environment.

Higher CPI can indicate increased price levels/inflation.

---

## `Unemployment`

Regional unemployment rate.

This is an economic indicator that may influence consumer spending.

---

## `IsHoliday`

Indicates whether the week is a holiday week.

---

# 14. Relationships

The core data model contains three major relationships.

---

## 14.1 Stores → Train

```text
stores.Store
     │
     │ 1
     │
     │ N
     ▼
train.Store
```

### Cardinality

```text
One-to-Many (1:N)
```

One store can have many sales records.

Example:

```text
Store 1
   │
   ├── Dept 1
   ├── Dept 2
   ├── Dept 3
   ├── Dept 4
   └── ...
```

---

# 15. Stores → Features

```text
stores.Store
     │
     │ 1
     │
     │ N
     ▼
features.Store
```

### Cardinality

```text
One-to-Many (1:N)
```

One store has many weekly feature records.

---

# 16. Features → Train

This relationship requires both:

```text
Store
Date
```

because the same store has many dates.

Conceptually:

```text
features
-----------
Store
Date
   │
   │
   │ Store + Date
   ▼
train
-----------
Store
Dept
Date
```

The relationship is:

```text
features.Store + features.Date
                 ↓
train.Store + train.Date
```

The `Dept` column is not part of the feature key because the external feature data is store/date-level rather than department-level.

---

# 17. Features → Test

The same relationship applies to the test dataset:

```text
features.Store + features.Date
                 ↓
test.Store + test.Date
```

This allows future external variables to be attached to future store/department observations.

---

# 18. Complete Relationship Model

```text
                         ┌──────────────────┐
                         │    STORES        │
                         │──────────────────│
                         │ PK Store         │
                         │ Type             │
                         │ Size             │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                   1│                          1│
                    │                           │
                   N│                          N│
                    ▼                           ▼
          ┌──────────────────┐        ┌──────────────────┐
          │     TRAIN        │        │    FEATURES      │
          │──────────────────│        │──────────────────│
          │ PK Store         │        │ PK Store         │
          │ PK Dept          │        │ PK Date          │
          │ PK Date          │        │ Temperature      │
          │ Weekly_Sales     │        │ Fuel_Price       │
          │ IsHoliday        │        │ MarkDown1-5      │
          └──────────────────┘        │ CPI              │
                                      │ Unemployment     │
                                      │ IsHoliday        │
                                      └────────┬─────────┘
                                               │
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │      TEST        │
                                      │──────────────────│
                                      │ PK Store         │
                                      │ PK Dept          │
                                      │ PK Date          │
                                      │ IsHoliday        │
                                      └──────────────────┘
```

---

# 19. Star Schema Interpretation

For the machine-learning project, the data can also be understood using a dimensional model.

## Fact Table

```text
fact_sales
```

Based primarily on:

```text
train.csv
```

Important fields:

```text
Store
Dept
Date
Weekly_Sales
IsHoliday
```

---

## Store Dimension

```text
dim_store
```

Based on:

```text
stores.csv
```

Fields:

```text
Store
Type
Size
```

---

## Date Dimension

Created during feature engineering.

Fields include:

```text
Date
Year
Quarter
Month
Week
Day
DayOfWeek
IsWeekend
```

---

## Feature / Economic Dimension

Based on:

```text
features.csv
```

Fields include:

```text
Store
Date
Temperature
Fuel_Price
MarkDown1
MarkDown2
MarkDown3
MarkDown4
MarkDown5
CPI
Unemployment
IsHoliday
```

---

# 20. Processed Training Dataset

After merging the original datasets and performing feature engineering, your project produced:

```text
train_feature_engineered.csv
```

Shape:

```text
421,570 rows
66 columns
```

The dataset contains the original target:

```text
Weekly_Sales
```

plus engineered predictors.

---

# 21. Final Model Features

Your XGBoost model uses **56 predictive features**.

## Feature List

```text
Store
Dept
Year
Quarter
Month
Week
Day
DayOfWeek
IsWeekend
Holiday_Flag
Previous_Holiday
Next_Holiday
Near_Holiday
Temperature
Fuel_Price
CPI
Unemployment
Lag_1
Lag_2
Lag_4
Lag_8
Lag_12
Lag_52
Rolling_Mean_4
Rolling_Mean_8
Rolling_Mean_12
Rolling_Std_4
Rolling_Std_8
Rolling_Std_12
Rolling_Min_4
Rolling_Max_4
Expanding_Mean
Expanding_Std
Expanding_Min
Expanding_Max
Store_Avg_Sales
Store_Std_Sales
Store_Min_Sales
Store_Max_Sales
Store_Sales_Rank
Store_Type
Store_Size_Relative
Dept_Avg_Sales
Dept_Std_Sales
Dept_Min_Sales
Dept_Max_Sales
Dept_Sales_Rank
Dept_Sales_Contribution
Month_Sin
Month_Cos
Week_Sin
Week_Cos
Quarter_Sin
Quarter_Cos
DayOfWeek_Sin
DayOfWeek_Cos
```

---

# 22. Feature Categories

| Category              | Features                                                           |
| --------------------- | ------------------------------------------------------------------ |
| Store                 | `Store`, `Store_Type`, `Store_Size_Relative`                       |
| Department            | `Dept`, department statistics                                      |
| Time                  | `Year`, `Quarter`, `Month`, `Week`, `Day`, `DayOfWeek`             |
| Weekend               | `IsWeekend`                                                        |
| Holiday               | `Holiday_Flag`, `Previous_Holiday`, `Next_Holiday`, `Near_Holiday` |
| Economic              | `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`                 |
| Lag                   | `Lag_1`, `Lag_2`, `Lag_4`, `Lag_8`, `Lag_12`, `Lag_52`             |
| Rolling               | Rolling means/std/min/max                                          |
| Expanding             | Expanding mean/std/min/max                                         |
| Store Statistics      | Store average/std/min/max/rank                                     |
| Department Statistics | Department average/std/min/max/rank/contribution                   |
| Cyclical              | Sin/Cos transformations                                            |

---

# 23. Lag Feature Relationships

Lag features are generated within each:

```text
Store + Dept
```

group.

For example:

```text
Store = 1
Dept  = 1
```

Historical sales:

```text
Week 1 → 24,924
Week 2 → 46,039
Week 3 → 41,595
Week 4 → 19,403
```

Then:

```text
Lag_1
```

represents the previous week's sales.

```text
Lag_4
```

represents sales approximately four weeks earlier.

```text
Lag_52
```

represents sales approximately one year earlier.

---

# 24. Rolling Feature Relationships

Rolling features are also calculated within:

```text
Store + Dept
```

Examples:

```text
Rolling_Mean_4
```

= average of recent historical sales.

```text
Rolling_Std_4
```

= volatility of recent sales.

```text
Rolling_Min_4
```

= minimum recent sales.

```text
Rolling_Max_4
```

= maximum recent sales.

---

# 25. Store-Level Features

Store-level statistics are calculated from historical sales.

Examples:

```text
Store_Avg_Sales
Store_Std_Sales
Store_Min_Sales
Store_Max_Sales
Store_Sales_Rank
```

These allow the model to distinguish high-volume and low-volume stores.

---

# 26. Department-Level Features

Department statistics include:

```text
Dept_Avg_Sales
Dept_Std_Sales
Dept_Min_Sales
Dept_Max_Sales
Dept_Sales_Rank
Dept_Sales_Contribution
```

These features help the model understand the historical importance and behavior of individual departments.

---

# 27. Cyclical Features

Time is cyclical.

For example:

```text
December → January
```

are close in time even though their numerical month values are:

```text
12 → 1
```

To represent this correctly, the project uses:

```text
Month_Sin
Month_Cos
```

Similarly:

```text
Week_Sin
Week_Cos

Quarter_Sin
Quarter_Cos

DayOfWeek_Sin
DayOfWeek_Cos
```

---

# 28. Target Variable

The target variable is:

```text
Weekly_Sales
```

The machine-learning problem can therefore be represented as:

```text
X = 56 engineered features

y = Weekly_Sales
```

The model learns:

```text
Historical Features
        ↓
     XGBoost
        ↓
Predicted Weekly Sales
```

---

# 29. Train/Test Relationship

The two datasets share the same identification structure:

```text
Store
Dept
Date
```

Training:

```text
Store + Dept + Date
          ↓
Weekly_Sales
```

Testing:

```text
Store + Dept + Date
          ↓
Weekly_Sales = Unknown
```

Therefore:

```text
TRAIN
┌───────────────────────────────┐
│ Store                         │
│ Dept                          │
│ Date                          │
│ IsHoliday                     │
│ Weekly_Sales ← TARGET         │
└───────────────────────────────┘

TEST
┌───────────────────────────────┐
│ Store                         │
│ Dept                          │
│ Date                          │
│ IsHoliday                     │
│ Weekly_Sales ← TO PREDICT     │
└───────────────────────────────┘
```

---

# 30. Data Flow in This Project

```text
stores.csv
     │
     ├───────────────┐
     │               │
     ▼               ▼
train.csv       features.csv
     │               │
     └───────┬───────┘
             │
             ▼
          MERGE
             │
             ▼
     Combined Training Data
             │
             ▼
      Feature Engineering
             │
             ├── Time Features
             ├── Holiday Features
             ├── Lag Features
             ├── Rolling Features
             ├── Expanding Features
             ├── Store Features
             ├── Department Features
             └── Cyclical Features
             │
             ▼
train_feature_engineered.csv
             │
             ▼
          XGBoost
             │
             ▼
      Trained Model
             │
             ▼
test_feature_engineered.csv
             │
             ▼
          Prediction
             │
             ▼
    final_forecast.csv
```

---

# 31. Data Quality Considerations

## Missing Values

The original `features.csv` contains substantial missing values, especially in:

```text
MarkDown1
MarkDown2
MarkDown3
MarkDown4
MarkDown5
```

Markdown data is not available for all periods/stores.

Therefore, missing-value handling is an important preprocessing step.

---

# 32. Duplicate Records

The following combination should uniquely identify a sales observation:

```text
Store + Dept + Date
```

Therefore, duplicate checks should be performed using:

```python
train.duplicated(
    subset=["Store", "Dept", "Date"]
)
```

---

# 33. Referential Integrity

The following conditions should hold:

### Store Integrity

Every:

```text
train.Store
test.Store
features.Store
```

should correspond to:

```text
stores.Store
```

### Date Integrity

`features.Date` should correspond to the appropriate weekly dates in the sales datasets.

### Sales Key Integrity

```text
Store + Dept + Date
```

should uniquely identify sales records.

---

# 34. Important Modeling Consideration

Not every column in the processed dataset should automatically be used as a model feature.

For example:

```text
Weekly_Sales
```

is the target and must not be included in `X`.

The final model should use only the predefined feature list.

This prevents accidental target leakage.

---

# 35. Final Model Architecture

```text
                 INPUT DATA
                     │
                     ▼
          ┌────────────────────┐
          │ Store Information  │
          └────────────────────┘
                     │
                     ▼
          ┌────────────────────┐
          │ Economic Features  │
          └────────────────────┘
                     │
                     ▼
          ┌────────────────────┐
          │ Time Features      │
          └────────────────────┘
                     │
                     ▼
          ┌────────────────────┐
          │ Lag Features       │
          └────────────────────┘
                     │
                     ▼
          ┌────────────────────┐
          │ Rolling Features   │
          └────────────────────┘
                     │
                     ▼
          ┌────────────────────┐
          │ Store/Dept Stats   │
          └────────────────────┘
                     │
                     ▼
          ┌────────────────────┐
          │ Cyclical Features  │
          └────────────────────┘
                     │
                     ▼
                XGBoost
                     │
                     ▼
          Predicted Weekly Sales
```

---

# 36. Final Project Dataset Summary

| Layer     | Dataset                        | Role                       |
| --------- | ------------------------------ | -------------------------- |
| Raw       | `stores.csv`                   | Store dimension            |
| Raw       | `features.csv`                 | External/economic features |
| Raw       | `train.csv`                    | Historical sales           |
| Raw       | `test.csv`                     | Future observations        |
| Processed | `train_feature_engineered.csv` | Model training data        |
| Processed | `test_feature_engineered.csv`  | Model prediction data      |
| Output    | `final_forecast.csv`           | Final predictions          |

---

# 37. Key Relationships Summary

```text
stores
PK: Store
      │
      │ 1:N
      ▼
features
PK: Store + Date
      │
      │ Store + Date
      ▼
train
PK: Store + Dept + Date
```

And for forecasting:

```text
stores
      │
features
      │
      ▼
test
PK: Store + Dept + Date
      │
      ▼
XGBoost
      │
      ▼
Predicted Weekly Sales
```

---

# 38. Primary Key Summary

| Dataset                        | Primary Key           |
| ------------------------------ | --------------------- |
| `stores.csv`                   | `Store`               |
| `features.csv`                 | `Store + Date`        |
| `train.csv`                    | `Store + Dept + Date` |
| `test.csv`                     | `Store + Dept + Date` |
| `train_feature_engineered.csv` | `Store + Dept + Date` |
| `test_feature_engineered.csv`  | `Store + Dept + Date` |

---

# 39. Foreign Key Summary

| Child Dataset  | Foreign Key    | Parent Dataset |
| -------------- | -------------- | -------------- |
| `train.csv`    | `Store`        | `stores.csv`   |
| `test.csv`     | `Store`        | `stores.csv`   |
| `features.csv` | `Store`        | `stores.csv`   |
| `train.csv`    | `Store + Date` | `features.csv` |
| `test.csv`     | `Store + Date` | `features.csv` |

---

# 40. Final Business Interpretation

The Walmart forecasting dataset combines three major types of information:

### 1. Sales Behavior

```text
Weekly_Sales
Lag Features
Rolling Statistics
Expanding Statistics
```

These describe what happened historically.

### 2. Store and Department Characteristics

```text
Store
Dept
Store Type
Store Size
Store Statistics
Department Statistics
```

These describe where the sales occurred.

### 3. External Conditions

```text
Temperature
Fuel Price
CPI
Unemployment
Markdowns
Holidays
```

These describe external factors that may influence demand.

The final machine-learning model combines these three categories:

```text
Historical Sales
        +
Store / Department Characteristics
        +
External Conditions
        +
Time & Seasonal Patterns
        ↓
      XGBoost
        ↓
Future Weekly Sales Forecast
```

---

# 41. Project Outcome

The completed project produced:

```text
Training Data
421,570 rows

Test Data
115,064 rows

Final Engineered Features
56

Selected Model
XGBoost

MAE
1,200.49

RMSE
2,602.18

Baseline MAE
1,540.86

MAE Improvement
22.09%
```

The model is then used to forecast future weekly sales for Walmart stores and departments.

---

## 42. Short Data Model Summary

```text
                         STORES
                   ┌─────────────────┐
                   │ Store (PK)      │
                   │ Type            │
                   │ Size            │
                   └────────┬────────┘
                            │
                            │ 1:N
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
        FEATURES                         SALES
   ┌─────────────────┐          ┌─────────────────────┐
   │ Store (FK)      │          │ Store (PK/FK)       │
   │ Date (PK)       │          │ Dept (PK)           │
   │ Temperature     │          │ Date (PK/FK)        │
   │ Fuel_Price      │          │ Weekly_Sales        │
   │ MarkDown1-5     │          │ IsHoliday           │
   │ CPI             │          └─────────────────────┘
   │ Unemployment    │
   │ IsHoliday       │
   └─────────────────┘
```

**Core grain of the project:**

```text
ONE ROW = ONE STORE + ONE DEPARTMENT + ONE WEEK
```

This grain is the most important concept to remember when working with this dataset. It explains the primary key, lag calculations, rolling calculations, joins, model target, and final forecast structure.
