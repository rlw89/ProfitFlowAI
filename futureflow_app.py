import pandas as pd
import numpy as np
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
1. **Upload a CSV file** with 'Date' and 'Revenue' columns. **(Preferred date format: YYYY-MM-DD)**
2. **Choose a model:** See descriptions below.
3. **Adjust parameters** (optional) - tooltips are provided for each parameter.
4. **View forecasts** and compare model performance. 
5. **Download** your forecasted results!  
""")

# Sample CSV (more prominent with in-app preview)
st.subheader("Sample CSV File")  # Make it stand out more
st.markdown("Download a sample CSV file to see the required format:")
st.download_button(
    label="Download Sample CSV",
    data="""Date,Revenue
2024-01-01,100
2024-01-08,150
2024-01-15,200
2024-01-22,130
2024-01-29,170
2024-02-05,190
2024-02-12,220
2024-02-19,250
2024-02-26,210
2024-03-04,230
2024-03-11,260
2024-03-18,280""",
    file_name="sample_data.csv",
    mime="text/csv",
)

# In-app preview of sample data
st.markdown("Sample CSV Structure:")
st.caption("First 5 rows of the sample CSV")
sample_data = pd.DataFrame({
    'Date': pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-15', '2024-01-22', '2024-01-29']),
    'Revenue': [100, 150, 200, 130, 170]
})
st.dataframe(sample_data)

# File Upload Section
uploaded_file = st.file_uploader("Upload your CSV file (Date, Revenue columns required)", type="csv")

if uploaded_file:
    try:
        # Data Processing
        data = pd.read_csv(uploaded_file)

        # Identify potential date and revenue columns
        date_cols = data.columns[data.apply(lambda col: pd.to_datetime(col, errors='coerce').notnull().all())]
        revenue_cols = data.select_dtypes(include=['number']).columns

        st.subheader("Select Columns")
        
        # Ensure only valid date columns are shown in the date selector
        if not date_cols.empty:
            date_column = st.selectbox("Date Column", date_cols)
        else:
            st.error("No suitable date column found. Please ensure your CSV contains a valid date column.")
            date_column = None

        # Ensure only valid revenue columns are shown in the revenue selector
        if not revenue_cols.empty:
            revenue_column = st.selectbox("Revenue Column", revenue_cols)
        else:
            st.error("No suitable revenue column found. Please ensure your CSV contains a numeric revenue column.")
            revenue_column = None

        # Proceed only if both date and revenue columns are selected
        if date_column and revenue_column:
            data = data[[date_column, revenue_column]].rename(columns={date_column: "Date", revenue_column: "Revenue"})
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
            data = data.dropna().sort_values("Date")

            if data.empty:
                st.error("No valid data found after processing. Please check your CSV file.")
            else:
                # Plot Data
                st.subheader("Uploaded Revenue Data")
                st.line_chart(data.set_index("Date"))

                # Model Selection
                st.header("Forecasting Models")
                
                # Initialize model_metrics in session_state if it doesn't exist
                if "model_metrics" not in st.session_state:
                    st.session_state.model_metrics = []
                
                model_choice = st.radio("Select a Model:", [
                    "ARIMA (AutoRegressive Integrated Moving Average)", 
                    "SARIMA (Seasonal ARIMA)", 
                    "Prophet (by Facebook)"
                ])

                # Model Descriptions
                if model_choice == "ARIMA (AutoRegressive Integrated Moving Average)":
                    st.markdown("A statistical model that uses past data to predict future trends. Good for data with clear patterns.")
                    # Add optional "Learn More" link
                    # st.markdown("[Learn more about ARIMA](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)") 
                elif model_choice == "SARIMA (Seasonal ARIMA)":
                    st.markdown("An extension of ARIMA that handles data with seasonality (e.g., weekly, monthly, or yearly repeating patterns).")
                elif model_choice == "Prophet (by Facebook)":
                    st.markdown("A model developed by Facebook, designed for business time series data. It automatically handles seasonality, holidays, and other factors.")

                # ARIMA Model
                if model_choice.startswith("ARIMA"):
                    st.subheader(f"{model_choice} Model Parameters")
                    p = st.number_input("p (Auto-Regressive Term)", min_value=0, max_value=5, value=1, help="The number of past data points to use for predicting the next value.")
                    d = st.number_input("d (Differencing Term)", min_value=0, max_value=2, value=1, help="The number of times the data needs to be differenced to make it stationary.")
                    q = st.number_input("q (Moving Average Term)", min_value=0, max_value=5, value=1, help="The number of past forecast errors to use in the model.")

                    forecast_horizon = st.number_input("Forecast Horizon (Days)", min_value=1, max_value=365, value=30)

                    train = data[:-forecast_horizon]
                    test = data[-forecast_horizon:]
                    
                    model = ARIMA(train["Revenue"], order=(p, d, q))

                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=len(test))

                    mae, rmse, mape = evaluate_model(test["Revenue"], forecast)

                    # Store metrics for comparison
                    st.session_state.model_metrics.append({
                        "Model": model_choice,
                        "MAE": mae,
                        "RMSE": rmse,
                        "MAPE": mape
                    })

                    # Plot Results
                    st.subheader(f"{model_choice} Forecast")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=train["Date"], y=train["Revenue"], name="Train Data"))
                    fig.add_trace(go.Scatter(x=test["Date"], y=test["Revenue"], name="Test Data"))
                    fig.add_trace(go.Scatter(x=test["Date"], y=forecast, name=f"{model_choice} Forecast"))
                    st.plotly_chart(fig)

                    st.write(f"**MAE:** {mae:.2f}, **RMSE:** {rmse:.2f}, **MAPE:** {mape:.2f}%")

                # SARIMA Model
                elif model_choice.startswith("SARIMA"):
                    st.subheader(f"{model_choice} Model Parameters")
                    p = st.number_input("p (Auto-Regressive Term)", min_value=0, max_value=5, value=1, help="The number of past data points to use for predicting the next value.")
                    d = st.number_input("d (Differencing Term)", min_value=0, max_value=2, value=1, help="The number of times the data needs to be differenced to make it stationary.")
                    q = st.number_input("q (Moving Average Term)", min_value=0, max_value=5, value=1, help="The number of past forecast errors to use in the model.")
                    P = st.number_input("P (Seasonal Auto-Regressive Term)", min_value=0, max_value=5, value=0, help="Similar to 'p', but for the seasonal component.")
                    D = st.number_input("D (Seasonal Differencing Term)", min_value=0, max_value=2, value=0, help="Similar to 'd', but for the seasonal component.")
                    Q = st.number_input("Q (Seasonal Moving Average Term)", min_value=0, max_value=5, value=0, help="Similar to 'q', but for the seasonal component.")
                    s = st.number_input("S (Seasonal Periodicity)", min_value=0, max_value=24, value=0, help="The number of time steps in a single seasonal period (e.g., 12 for yearly seasonality).")
                    
                    forecast_horizon = st.number_input("Forecast Horizon (Days)", min_value=1, max_value=365, value=30)

                    train = data[:-forecast_horizon]
                    test = data[-forecast_horizon:]
                    
                    model = ARIMA(train["Revenue"], order=(p, d, q), seasonal_order=(P, D, Q, s))

                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=len(test))

                    mae, rmse, mape = evaluate_model(test["Revenue"], forecast)

                    # Store metrics for comparison
                    st.session_state.model_metrics.append({
                        "Model": model_choice,
                        "MAE": mae,
                        "RMSE": rmse,
                        "MAPE": mape
                    })

                    # Plot Results
                    st.subheader(f"{model_choice} Forecast")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=train["Date"], y=train["Revenue"], name="Train Data"))
                    fig.add_trace(go.Scatter(x=test["Date"], y=test["Revenue"], name="Test Data"))
                    fig.add_trace(go.Scatter(x=test["Date"], y=forecast, name=f"{model_choice} Forecast"))
                    st.plotly_chart(fig)

                    st.write(f"**MAE:** {mae:.2f}, **RMSE:** {rmse:.2f}, **MAPE:** {mape:.2f}%")

                # Prophet Model
                elif model_choice == "Prophet (by Facebook)":
                    st.subheader("Prophet Hyperparameters")
                    changepoint_scale = st.slider("Changepoint Prior Scale", 0.01, 0.5, 0.05, help="Controls how flexible the trend is allowed to be. Higher values allow for more abrupt changes in the trend.")
                    seasonality_scale = st.slider("Seasonality Prior Scale", 1.0, 10.0, 5.0, help="Controls the strength of the seasonality component. Higher values make the model fit more closely to the seasonal pattern.")
                    holidays_scale = st.slider("Holidays Prior Scale", 0.1, 10.0, 5.0, help="Controls the strength of the holiday effects. Higher values allow for larger impacts of holidays on the forecast.")
                    seasonality_mode = st.selectbox("Seasonality Mode", ("additive", "multiplicative"), help="Whether the seasonality should be modeled as additive (constant seasonal effect) or multiplicative (seasonal effect scales with the trend).")
                    
                    forecast_horizon = st.number_input("Forecast Horizon (Days)", min_value=1, max_value=365, value=30)

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

                    # Evaluate Prophet model using the forecast horizon
                    test_data = prophet_data.tail(forecast_horizon)
                    # Correctly extract yhat values for the forecast horizon period
                    forecast_values = forecast.set_index('ds')['yhat'].tail(forecast_horizon)

                    # Ensure test_data and forecast_values are aligned
                    if not test_data.empty and not forecast_values.empty:
                        mae, rmse, mape = evaluate_model(test_data["y"], forecast_values)
                    
                        # Store metrics for comparison
                        st.session_state.model_metrics.append({
                            "Model": "Prophet",
                            "MAE": mae,
                            "RMSE": rmse,
                            "MAPE": mape
                        })

                    # Plot Results
                    st.subheader("Prophet Forecast")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=prophet_data["ds"], y=prophet_data["y"], name="Actual Data"))
                    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Prophet Forecast"))
                    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], name="Lower Bound", line=dict(dash="dash")))
                    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], name="Upper Bound", line=dict(dash="dash")))
                    st.plotly_chart(fig)

                    # Components
                    st.subheader("Forecast Components")
                    st.plotly_chart(px.line(forecast, x="ds", y=["trend", "yearly", "weekly"], title="Prophet Components"))
                
                # Display Model Comparison
                st.subheader("Model Comparison")
                if st.session_state.model_metrics:
                    comparison_df = pd.DataFrame(st.session_state.model_metrics)
                    st.table(comparison_df.set_index("Model"))
                else:
                    st.write("No models have been evaluated yet.")

                # Boost Suggestions
                st.header("Boost Suggestions")
                st.info("""
                - ✅ Run a **3-day flash sale**. 
                - ✅ Bundle your **top 2 products** into a discounted offer. 
                - ✅ Send a **'last chance' email** to your audience. 
                """)
        else:
            st.info("Please select valid date and revenue columns to proceed.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a CSV file to see your forecasts.")
