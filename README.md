# Walmart Sales Forecasting

## 📌 Project Overview

This project develops a machine learning solution to forecast weekly sales for Walmart stores and departments.

The goal is to use historical sales patterns, time-based features, store and department characteristics, economic indicators, holidays, and lag/rolling statistics to predict future weekly sales.

The final model selected for the project is **XGBoost Regressor**.

---

## 🎯 Business Problem

Walmart needs accurate sales forecasts to support:

* Inventory planning
* Store staffing
* Supply chain management
* Promotion planning
* Seasonal demand planning
* Store-level decision making

The key business question is:

> **Can historical sales patterns and external factors be used to accurately forecast future weekly sales?**

---

## 🎯 Project Objectives

The main objectives are:

1. Load and understand the Walmart sales dataset.
2. Clean and preprocess the data.
3. Perform exploratory data analysis.
4. Identify sales trends and seasonality.
5. Create time-series features.
6. Create lag and rolling features.
7. Create store-level and department-level features.
8. Train machine learning models.
9. Compare model performance against a baseline.
10. Select the best-performing model.
11. Generate future sales forecasts.
12. Save the trained model and forecast results.

---

## 📊 Dataset

The project uses the **Walmart Store Sales Forecasting** dataset.

The main datasets contain information about:

### Training Data

* Store
* Dept
* Date
* Weekly_Sales
* IsHoliday

### Store Information

* Store
* Type
* Size

### Test Data

* Store
* Dept
* Date
* IsHoliday
* Temperature
* Fuel_Price
* MarkDown1
* MarkDown2
* MarkDown3
* MarkDown4
* MarkDown5
* CPI
* Unemployment

---

## 📅 Dataset Time Period

### Training Period

```text
2010-02-05 → 2012-10-26
```

### Forecast/Test Period

```text
2012-11-02 → 2013-07-26
```

### Dataset Size

```text
Training rows: 421,570
Test rows:     115,064
Combined rows: 536,634
```

---

# 🔄 Project Workflow

```text
Raw Walmart Dataset
        ↓
Data Loading
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train / Validation Split
        ↓
Baseline Model
        ↓
XGBoost Model
        ↓
Model Evaluation
        ↓
Model Selection
        ↓
Future Sales Prediction
        ↓
Final Forecast
```

---

# 🔍 Exploratory Data Analysis

The exploratory analysis investigates:

* Weekly sales distribution
* Weekly sales trends
* Sales by store
* Sales by department
* Sales by store type
* Holiday impact
* Temperature vs. sales
* Fuel price vs. sales
* CPI vs. sales
* Unemployment vs. sales
* Monthly sales trends
* Seasonal patterns

The purpose of EDA is to understand the factors influencing Walmart's weekly sales before building the forecasting model.

---

# ⚙️ Feature Engineering

Feature engineering is one of the most important parts of this project.

The final model uses **56 engineered features**.

## Time-Based Features

```text
Year
Quarter
Month
Week
Day
DayOfWeek
IsWeekend
```

## Holiday Features

```text
Holiday_Flag
Previous_Holiday
Next_Holiday
Near_Holiday
```

## Economic Features

```text
Temperature
Fuel_Price
CPI
Unemployment
```

## Lag Features

Historical sales are used to create lag variables:

```text
Lag_1
Lag_2
Lag_4
Lag_8
Lag_12
Lag_52
```

These features capture recent and seasonal sales patterns.

For example:

```text
Lag_1  → previous week's sales
Lag_4  → sales approximately one month earlier
Lag_52 → sales approximately one year earlier
```

## Rolling Features

Rolling statistics capture recent sales behavior:

```text
Rolling_Mean_4
Rolling_Mean_8
Rolling_Mean_12

Rolling_Std_4
Rolling_Std_8
Rolling_Std_12

Rolling_Min_4
Rolling_Max_4
```

## Expanding Features

```text
Expanding_Mean
Expanding_Std
Expanding_Min
Expanding_Max
```

These capture the long-term historical behavior of sales.

## Store-Level Features

```text
Store_Avg_Sales
Store_Std_Sales
Store_Min_Sales
Store_Max_Sales
Store_Sales_Rank
Store_Type
Store_Size_Relative
```

## Department-Level Features

```text
Dept_Avg_Sales
Dept_Std_Sales
Dept_Min_Sales
Dept_Max_Sales
Dept_Sales_Rank
Dept_Sales_Contribution
```

## Cyclical Features

Cyclical encoding was used to represent seasonal patterns:

```text
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

# 🤖 Model

## Selected Model: XGBoost

The final selected model is:

```text
XGBRegressor
```

XGBoost was selected because it performs well with:

* Non-linear relationships
* Large datasets
* Numerical features
* Interaction effects
* Complex feature relationships
* High-dimensional engineered features

The final training dataset contains:

```text
421,570 rows
66 columns
```

The model uses the engineered predictive features while excluding the target variable from the input.

---

# 📈 Model Evaluation

The model was evaluated using:

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted sales.

```text
MAE = 1,200.49
```

### RMSE

Root Mean Squared Error gives greater weight to large prediction errors.

```text
RMSE = 2,602.18
```

### Baseline MAE

The baseline model achieved:

```text
Baseline MAE = 1,540.86
```

### MAE Improvement

XGBoost improved the baseline MAE by approximately:

```text
22.09%
```

### Final Model Result

| Metric          |   Result |
| --------------- | -------: |
| Selected Model  |  XGBoost |
| MAE             | 1,200.49 |
| RMSE            | 2,602.18 |
| Baseline MAE    | 1,540.86 |
| MAE Improvement |   22.09% |

> **Note:** The original notebook contained an extremely large MAPE value caused by near-zero actual values. MAPE should therefore be interpreted carefully for this dataset, and MAE/RMSE are treated as the primary evaluation metrics.

---

# 🔮 Forecasting

After training the final XGBoost model, predictions are generated for the test period:

```text
2012-11-02 → 2013-07-26
```

The final forecast contains:

```text
Store
Dept
Date
Predicted_Weekly_Sales
```

Example:

```text
Store    Dept    Date          Predicted_Weekly_Sales
1        1       2012-11-02    ...
1        1       2012-11-09    ...
1        1       2012-11-16    ...
1        1       2012-11-23    ...
```

---

# 📁 Project Structure

```text
Walmart-Sales-Forecasting/
│
├── data/
│   ├── raw/
│   │
│   └── processed/
│       ├── train_feature_engineered.csv
│       └── test_feature_engineered.csv
│
├── notebooks/
│   ├── 01_Data_Loading.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Exploratory_Data_Analysis.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   └── 10_Final_Forecast.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── evaluation.py
│
├── models/
│   └── xgboost_model.pkl
│
├── outputs/
│   ├── model_evaluation.csv
│   └── final_forecast.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Model Persistence

* Joblib

### Development Environment

* Jupyter Notebook

---

# 📦 Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Navigate to the project:

```bash
cd Walmart-Sales-Forecasting
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## 1. Run the notebooks

Start with:

```text
notebooks/01_Data_Loading.ipynb
```

and proceed through the project workflow.

---

## 2. Train the model

The reusable training pipeline is located at:

```text
src/train.py
```

It loads:

```text
data/processed/train_feature_engineered.csv
```

and saves:

```text
models/xgboost_model.pkl
```

---

## 3. Generate predictions

The prediction pipeline is located at:

```text
src/predict.py
```

It loads the trained XGBoost model and:

```text
test_feature_engineered.csv
        ↓
XGBoost Model
        ↓
Predictions
        ↓
final_forecast.csv
```

---

# 💼 Business Insights

The project demonstrates several important forecasting concepts:

### 1. Historical sales are highly informative

Lag features allow the model to learn from previous sales behavior.

### 2. Seasonal patterns matter

The 52-week lag captures yearly seasonality, while month and week features capture shorter seasonal patterns.

### 3. Store and department behavior differs

Store-level and department-level statistics help the model understand differences in sales volume and performance.

### 4. Holidays influence demand

Holiday indicators and holiday proximity features allow the model to account for unusual sales periods.

### 5. Machine learning improves the baseline

The final XGBoost model reduced MAE from approximately:

```text
1,540.86
```

to:

```text
1,200.49
```

representing approximately:

```text
22.09% improvement
```

---

# 📌 Key Takeaways

* Built an end-to-end Walmart sales forecasting pipeline.
* Performed exploratory data analysis on historical sales.
* Created time-series, lag, rolling, expanding, store, department, and cyclical features.
* Built a machine learning forecasting model using XGBoost.
* Evaluated the model using MAE and RMSE.
* Achieved approximately **22.09% MAE improvement over the baseline**.
* Generated future weekly sales predictions for Walmart stores and departments.
* Organized reusable production-style Python code inside a `src/` directory.

---

# 🚀 Future Improvements

Possible future improvements include:

* Hyperparameter optimization with Optuna or GridSearchCV
* Time-series cross-validation
* Store-specific forecasting models
* Advanced ensemble models
* Improved MAPE/SMAPE evaluation
* Prediction intervals
* SHAP-based model explainability
* Interactive Power BI dashboard
* Automated forecasting pipeline

---

# 👤 Author

**Sara Yeana**

Aspiring Data Analyst / Business Intelligence Professional

Skills demonstrated in this project:

```text
Python
Pandas
NumPy
SQL
Data Analysis
Feature Engineering
Time Series Analysis
Machine Learning
XGBoost
Data Visualization
Business Analytics
```
