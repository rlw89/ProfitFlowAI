import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
import time

# --- Helper Functions ---

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if 0 in y_true:
        st.warning("The actual data contains 0 values, which may lead to division by zero errors in MAPE calculation.")
        return np.nan
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(y_true, y_pred):
    """Evaluates the model using MAE, RMSE, and MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, rmse, mape

def preprocess_data(data, date_column, revenue_column):
    """Preprocesses the uploaded data: handles column renaming, date parsing, missing/duplicate dates."""
    data = data[[date_column, revenue_column]].rename(columns={date_column: "Date", revenue_column: "Revenue"})
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    # Handle Missing Dates (fill with linear interpolation)
    if data["Date"].isnull().any():
        st.warning("Missing dates found. Filling with linear interpolation.")
        data = data.set_index("Date").resample("D").interpolate(method="linear").reset_index()

    # Handle Duplicate Dates (aggregate by taking the mean)
    if data["Date"].duplicated().any():
        st.warning("Duplicate dates found. Aggregating revenue by taking the mean.")
        data = data.groupby("Date", as_index=False)["Revenue"].mean()

    data = data.dropna(subset=["Date", "Revenue"]).sort_values("Date")
    return data

def plot_data(data):
    """Plots the uploaded revenue data with an interactive rangeslider."""
    fig = px.line(data, x="Date", y="Revenue", title="Revenue Data")
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig)

def run_arima_forecast(data, p, d, q, forecast_horizon):
    """Runs ARIMA forecasting."""
    try:
        model = ARIMA(data["Revenue"], order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_horizon)
        return forecast, model_fit
    except Exception as e:
        st.error(f"Error fitting ARIMA model: {e}")
        return None, None

def run_sarima_forecast(data, p, d, q, P, D, Q, s, forecast_horizon):
    """Runs SARIMA forecasting."""
    try:
        model = SARIMAX(data["Revenue"], order=(p, d, q), seasonal_order=(P, D, Q, s))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_horizon)
        return forecast, model_fit
    except Exception as e:
        st.error(f"Error fitting SARIMA model: {e}")
        return None, None

def run_prophet_forecast(data, changepoint_scale, seasonality_scale, holidays_scale, seasonality_mode, forecast_horizon):
    """Runs Prophet forecasting."""
    try:
        prophet_data = data.rename(columns={"Date": "ds", "Revenue": "y"})
        prophet = Prophet(
            changepoint_prior_scale=changepoint_scale,
            seasonality_prior_scale=seasonality_scale,
            holidays_prior_scale=holidays_scale,
            seasonality_mode=seasonality_mode
        )
        prophet.fit(prophet_data)
        future = prophet.make_future_dataframe(periods=forecast_horizon)
        forecast = prophet.predict(future)
        return forecast, prophet
    except Exception as e:
        st.error(f"Error fitting Prophet model: {e}")
        return None, None

def check_stationarity(data):
    """Performs the Augmented Dickey-Fuller test for stationarity."""
    st.write("Augmented Dickey-Fuller Test:")
    try:
        result = adfuller(data)
        st.write('ADF Statistic:', result[0])
        st.write('p-value:', result[1])
        st.write('Critical Values:')
        for key, value in result[4].items():
            st.write(f'   {key}: {value}')
        if result[1] <= 0.05:
            st.write("**Data is likely stationary.**")
        else:
            st.write("**Data is likely non-stationary.**")
    except Exception as e:
        st.error(f"Error during stationarity test: {e}")

# --- Streamlit App UI ---

# App Title and Configuration
st.set_page_config(page_title="ProfitFlowAI", page_icon="https://i.ibb.co/RbmmVwq/futureflow-logo.webp")
st.image("https://i.ibb.co/RbmmVwq/futureflow-logo.webp", width=200)
st.title("ProfitFlowAI - Predict. Boost. Succeed.")
st.write("Unlock your business's future with AI-powered revenue forecasting.")

# Instructions
st.header("How to Use")
st.markdown("""
1. **Upload a CSV file** with 'Date' and 'Revenue' columns. **(Preferred date format: YYYY-MM-DD)**
2. **Choose a model:** See descriptions below and tooltips for parameter details.
3. **View interactive forecasts** and compare model performance.
4. **Download** your forecasted results!
""")

# Sample CSV (Improved with more realistic data)
st.subheader("Sample CSV File")
st.markdown("Download a sample CSV file to see the required format:")
st.download_button(
    label="Download Sample CSV",
    data="""Date,Revenue
2023-01-01,2100
2023-01-08,2150
2023-01-15,2000
2023-01-22,2130
2023-01-29,2170
2023-02-05,2190
2023-02-12,2220
2023-02-19,2250
2023-02-26,2310""",
    file_name="sample_data.csv",
    mime="text/csv",
)

# In-app preview of sample data
st.markdown("Sample CSV Structure:")
st.caption("First 5 rows of the sample CSV")
sample_data = pd.DataFrame({
    'Date': pd.to_datetime(['2023-01-01', '2023-01-08', '2023-01-15', '2023-01-22', '2023-01-29']),
    'Revenue': [2100, 2150, 2000, 2130, 2170]
})
st.dataframe(sample_data)

# File Upload Section
uploaded_file = st.file_uploader("Upload your CSV file (Date, Revenue columns required)", type="csv")

if uploaded_file is not None:
    try:
        # Data Processing
        data = pd.read_csv(uploaded_file)
        if data.empty:
            raise ValueError("Uploaded CSV file is empty.")
        
        # Identify date and revenue columns
        date_cols = data.columns[data.apply(lambda col: pd.to_datetime(col, errors='coerce').notnull().all())]
        revenue_cols = data.select_dtypes(include=['number']).columns

        st.subheader("Select Columns")
        date_column = st.selectbox("Date Column", date_cols, help="Select the column containing dates.")
        revenue_column = st.selectbox("Revenue Column", revenue_cols, help="Select the column containing revenue data.")

        # Preprocess data
        data = preprocess_data(data, date_column, revenue_column)
        st.subheader("Uploaded Revenue Data")
        st.dataframe(data.head())
        plot_data(data)
        check_stationarity(data["Revenue"])

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a CSV file to see your forecasts.")


