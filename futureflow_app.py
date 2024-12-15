import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from io import BytesIO
import base64
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Helper functions for evaluation
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, rmse, mape

# App Title
st.set_page_config(page_title="FutureFlow", page_icon="https://i.ibb.co/RbmmVwq/futureflow-logo.webp")
st.image("https://i.ibb.co/RbmmVwq/futureflow-logo.webp", width=200)
st.title("Futureflow - Predict. Boost. Succeed.")
st.write("See your future revenue, detect slow weeks, and boost your cash flow with actionable tips!")

# "How to Use" Section
st.header("How to Use")
st.markdown(
    """
    1. **Upload your revenue data**: Use a CSV file with columns for `Date` and `Revenue`.
    2. **Choose a forecasting model**: ARIMA, SARIMA, or Prophet.
    3. **Tune parameters**: Adjust manual settings for more precise forecasts.
    4. **View forecasts and compare models**: Review predictions, errors, and trends.
    5. **Act on suggestions**: Boost your cash flow with actionable tips provided below the forecast.
    """
)
st.download_button(
    label="Download Sample CSV",
    data="""Date,Revenue\n2024-01-01,100\n2024-01-02,150\n2024-01-03,200\n2024-01-04,130\n2024-01-05,170\n""".encode("utf-8"),
    file_name="sample_data.csv",
    mime="text/csv",
)

# File Upload Section
uploaded_file = st.file_uploader("Upload your revenue CSV (Date, Revenue columns required)", type=["csv"])

if uploaded_file is not None:
    try:
        with st.spinner('Processing your file...'):
            # Read uploaded file
            data = pd.read_csv(uploaded_file)

            # Allow users to select columns
            st.subheader("Select Date and Revenue Columns")
            date_column = st.selectbox("Select the Date column", data.columns)
            revenue_column = st.selectbox("Select the Revenue column", data.columns)
            
            # Validate and clean data
            data = data[[date_column, revenue_column]].rename(columns={date_column: 'Date', revenue_column: 'Revenue'})
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data = data.dropna(subset=['Date', 'Revenue'])
            data = data.sort_values('Date')

            if data.empty:
                st.warning("No data available to display after processing. Please check your file contents.")
            else:
                # Plot raw data
                st.subheader("Original Revenue Data")
                st.line_chart(data.set_index('Date'))

                # Model Selection
                st.header("Forecasting Models")
                model_choice = st.radio("Choose a model for forecasting:", ("ARIMA", "Prophet"))
                
                # Forecasting with ARIMA
                if model_choice == "ARIMA":
                    st.subheader("ARIMA Model Parameters")
                    p = st.number_input("p (Auto-Regressive Term):", min_value=0, max_value=5, value=1)
                    d = st.number_input("d (Differencing Term):", min_value=0, max_value=2, value=1)
                    q = st.number_input("q (Moving Average Term):", min_value=0, max_value=5, value=1)
                    
                    train, test = data[:-10], data[-10:]
                    model = ARIMA(train['Revenue'], order=(p, d, q))
                    fitted_model = model.fit()
                    forecast = fitted_model.forecast(steps=len(test))

                    mae, rmse, mape = evaluate_model(test['Revenue'], forecast)
                    
                    # Plot Forecast
                    st.subheader("ARIMA Forecast")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=train['Date'], y=train['Revenue'], mode='lines', name='Train Data'))
                    fig.add_trace(go.Scatter(x=test['Date'], y=test['Revenue'], mode='lines', name='Test Data'))
                    fig.add_trace(go.Scatter(x=test['Date'], y=forecast, mode='lines', name='Forecast'))
                    st.plotly_chart(fig)

                    # Display metrics
                    st.write(f"**MAE:** {mae:.2f}, **RMSE:** {rmse:.2f}, **MAPE:** {mape:.2f}%")

                # Forecasting with Prophet
                elif model_choice == "Prophet":
                    st.subheader("Prophet Hyperparameters")
                    changepoint_prior_scale = st.slider("Changepoint Prior Scale", 0.01, 0.5, 0.05)
                    seasonality_prior_scale = st.slider("Seasonality Prior Scale", 1.0, 10.0, 5.0)

                    df_prophet = data.rename(columns={'Date': 'ds', 'Revenue': 'y'})
                    prophet = Prophet(
                        changepoint_prior_scale=changepoint_prior_scale,
                        seasonality_prior_scale=seasonality_prior_scale
                    )
                    prophet.fit(df_prophet)
                    future = prophet.make_future_dataframe(periods=30)
                    forecast = prophet.predict(future)

                    # Plot Forecast
                    st.subheader("Prophet Forecast")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Original Data'))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
                    st.plotly_chart(fig)

                    # Display components
                    st.write("### Forecast Components")
                    st.plotly_chart(px.line(forecast, x='ds', y=['trend', 'seasonal'], title="Forecast Components"))

                # Suggestions Section
                st.header("Boost Suggestions")
                st.info(
                    """
                    - ✅ Run a 3-day flash sale.
                    - ✅ Bundle your top 2 products into a discounted offer.
                    - ✅ Send a 'last chance' email to your audience.
                    """
                )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")


