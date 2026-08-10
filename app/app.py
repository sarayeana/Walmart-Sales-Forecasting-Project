import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# WALMART SALES FORECASTING APP
# ============================================================

st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="🛒",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/xgboost_model.pkl"

FORECAST_PATH = "outputs/final_forecast.csv"

TRAIN_PATH = (
    "data/processed/train_feature_engineered.csv"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🛒 Walmart Sales Forecasting")

st.markdown(
    """
    **Machine Learning Sales Forecasting Dashboard**

    Explore Walmart sales forecasts by store and department
    using the trained **XGBoost** model.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_forecast():

    if not os.path.exists(FORECAST_PATH):
        return None

    df = pd.read_csv(FORECAST_PATH)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    return df


@st.cache_data
def load_train():

    if not os.path.exists(TRAIN_PATH):
        return None

    df = pd.read_csv(TRAIN_PATH)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    return df


model = load_model()
forecast = load_forecast()
train = load_train()


# ============================================================
# CHECK FILES
# ============================================================

if forecast is None:

    st.error(
        """
        `final_forecast.csv` was not found.

        Please place your forecast file inside:

        `outputs/final_forecast.csv`
        """
    )

    st.stop()


# ============================================================
# IDENTIFY PREDICTION COLUMN
# ============================================================

prediction_columns = [
    "Predicted_Weekly_Sales",
    "Predicted_Sales",
    "Forecast",
    "Prediction"
]

prediction_column = None

for column in prediction_columns:

    if column in forecast.columns:

        prediction_column = column

        break


if prediction_column is None:

    st.error(
        """
        No prediction column was found.

        Expected one of:

        `Predicted_Weekly_Sales`
        `Predicted_Sales`
        `Forecast`
        `Prediction`
        """
    )

    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Forecast Filters")


# Store selection

stores = sorted(
    forecast["Store"].dropna().unique()
)

selected_store = st.sidebar.selectbox(
    "Select Store",
    stores
)


# Department selection

store_data = forecast[
    forecast["Store"] == selected_store
]

departments = sorted(
    store_data["Dept"].dropna().unique()
)

selected_department = st.sidebar.selectbox(
    "Select Department",
    departments
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_forecast = forecast[
    (forecast["Store"] == selected_store)
    &
    (forecast["Dept"] == selected_department)
].copy()


# Sort by date

if "Date" in filtered_forecast.columns:

    filtered_forecast = (
        filtered_forecast
        .sort_values("Date")
    )


# ============================================================
# KPI CALCULATIONS
# ============================================================

average_forecast = (
    filtered_forecast[prediction_column]
    .mean()
)

maximum_forecast = (
    filtered_forecast[prediction_column]
    .max()
)

minimum_forecast = (
    filtered_forecast[prediction_column]
    .min()
)

total_forecast = (
    filtered_forecast[prediction_column]
    .sum()
)


# ============================================================
# KPI CARDS
# ============================================================

st.subheader(
    f"📊 Store {selected_store} | Department {selected_department}"
)

col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Average Weekly Forecast",
    f"${average_forecast:,.0f}"
)

col2.metric(
    "Highest Forecast",
    f"${maximum_forecast:,.0f}"
)

col3.metric(
    "Lowest Forecast",
    f"${minimum_forecast:,.0f}"
)

col4.metric(
    "Total Forecast",
    f"${total_forecast:,.0f}"
)


# ============================================================
# FORECAST CHART
# ============================================================

st.subheader("📈 Weekly Sales Forecast")


if "Date" in filtered_forecast.columns:

    fig = px.line(
        filtered_forecast,
        x="Date",
        y=prediction_column,
        markers=True,
        title=(
            f"Forecast — Store {selected_store}, "
            f"Department {selected_department}"
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Predicted Weekly Sales",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# HISTORICAL VS FORECAST
# ============================================================

if train is not None:

    st.subheader("📊 Historical Sales vs Forecast")

    historical = train[
        (train["Store"] == selected_store)
        &
        (train["Dept"] == selected_department)
    ].copy()

    if not historical.empty:

        historical = historical[
            ["Date", "Weekly_Sales"]
        ].copy()

        historical["Type"] = "Historical"

        historical = historical.rename(
            columns={
                "Weekly_Sales": "Sales"
            }
        )

        future = filtered_forecast[
            ["Date", prediction_column]
        ].copy()

        future["Type"] = "Forecast"

        future = future.rename(
            columns={
                prediction_column: "Sales"
            }
        )

        comparison = pd.concat(
            [
                historical,
                future
            ],
            ignore_index=True
        )

        fig2 = px.line(
            comparison,
            x="Date",
            y="Sales",
            color="Type",
            title="Historical Sales and Future Forecast"
        )

        fig2.update_layout(
            xaxis_title="Date",
            yaxis_title="Weekly Sales",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


# ============================================================
# FORECAST TABLE
# ============================================================

st.subheader("📋 Forecast Details")


display_columns = [
    "Store",
    "Dept",
    "Date",
    prediction_column
]

display_columns = [
    column
    for column in display_columns
    if column in filtered_forecast.columns
]


display_data = filtered_forecast[
    display_columns
].copy()


if prediction_column in display_data.columns:

    display_data[
        prediction_column
    ] = display_data[
        prediction_column
    ].round(2)


st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD FORECAST
# ============================================================

csv_data = filtered_forecast.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="⬇️ Download Selected Forecast",
    data=csv_data,
    file_name=(
        f"store_{selected_store}_"
        f"dept_{selected_department}_forecast.csv"
    ),
    mime="text/csv"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("🤖 Model Information")


col1, col2, col3 = st.columns(3)


col1.metric(
    "Model",
    "XGBoost"
)

col2.metric(
    "Training Rows",
    "421,570"
)

col3.metric(
    "Features",
    "56"
)


st.caption(
    "Walmart Sales Forecasting | "
    "Machine Learning Project"
)
```
