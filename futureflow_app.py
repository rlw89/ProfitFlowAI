import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
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
st.title("FutureFlow - Predict. Boost. Succeed.")
st.write("See your future revenue, detect slow weeks, and boost your cash flow with actionable tips!")

# Instructions
st.header("How to Use")
st.markdown("""
1. **Upload a CSV file** with `Date` and `Revenue` columns.  
2. **Choose a model**: ARIMA or Prophet.  
3. **Adjust parameters** for precision or leave them default.  
4. **View forecasts** and compare model performance.  
5. **Download** your forecasted results!  
""")

# Sample CSV
st.download_button(
    label="Download Sample CSV",
    data="Date,Revenue\n2024-01-01,100\n2024-01-02,150\n2024-01-03,200\n2024-01-04,130\n2024-01-05,170",
    file_name="sample_data.csv",
    mime="text/csv",
)

# File Upload Section
uploaded_file = st.file_uploader("Upload your CSV file (Date, Revenue columns required)", type="csv")

if uploaded_file:
    try:
        # Data Processing
        data = pd.read_csv(uploaded_file)
        st.subheader("Select Columns")
        date_column = st.selectbox("Date Column", data.columns)
        revenue_column = st.selectbox("Revenue Column", data.columns)

        data = data[[date_column, revenue_column]].rename(columns={date_column: "Date", revenue_column: "Revenue"})
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna().sort_values("Date")

        if data.empty:
            st.error("No valid data found. Please check your CSV file.")
        else:
            # Plot Data
            st.subheader("Uploaded Revenue Data")
            st.line_chart(data.set_index("Date"))

            # Model Selection
            st.header("Forecasting Models")
            model_choice = st.radio("Select a Model:", ["ARIMA", "Prophet"])

            # ARIMA Model
            if model_choice == "ARIMA":
                st.subheader("ARIMA Model Parameters")
                p = st.number_input("p (Auto-Regressive Term)", min_value=0, max_value=5, value=1)
                d = st.number_input("d (Differencing Term)", min_value=0, max_value=2, value=1)
                q = st.number_input("q (Moving Average Term)", min_value=0, max_value=5, value=1)

                train = data[:-10]
                test = data[-10:]

                model = ARIMA(train["Revenue"], order=(p, d, q))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=len(test))

                mae, rmse, mape = evaluate_model(test["Revenue"], forecast)

                # Plot Results
                st.subheader("ARIMA Forecast")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train["Date"], y=train["Revenue"], name="Train Data"))
                fig.add_trace(go.Scatter(x=test["Date"], y=test["Revenue"], name="Test Data"))
                fig.add_trace(go.Scatter(x=test["Date"], y=forecast, name="ARIMA Forecast"))
                st.plotly_chart(fig)

                st.write(f"**MAE:** {mae:.2f}, **RMSE:** {rmse:.2f}, **MAPE:** {mape:.2f}%")

            # Prophet Model
            elif model_choice == "Prophet":
                st.subheader("Prophet Hyperparameters")
                changepoint_scale = st.slider("Changepoint Prior Scale", 0.01, 0.5, 0.05)
                seasonality_scale = st.slider("Seasonality Prior Scale", 1.0, 10.0, 5.0)

                prophet_data = data.rename(columns={"Date": "ds", "Revenue": "y"})
                prophet = Prophet(changepoint_prior_scale=changepoint_scale, seasonality_prior_scale=seasonality_scale)
                prophet.fit(prophet_data)

                future = prophet.make_future_dataframe(periods=30)
                forecast = prophet.predict(future)

                # Plot Results
                st.subheader("Prophet Forecast")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=prophet_data["ds"], y=prophet_data["y"], name="Actual Data"))
                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Prophet Forecast"))
                st.plotly_chart(fig)

                # Components
                st.subheader("Forecast Components")
                st.plotly_chart(px.line(forecast, x="ds", y=["trend", "seasonal"], title="Prophet Components"))

            # Boost Suggestions
            st.header("Boost Suggestions")
            st.info("""
            - ✅ Run a **3-day flash sale**.  
            - ✅ Bundle your **top 2 products** into a discounted offer.  
            - ✅ Send a **'last chance' email** to your audience.  
            """)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a CSV file to see your forecasts.")
