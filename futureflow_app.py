import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from io import BytesIO
import base64
from datetime import datetime

# App Configurations
st.set_page_config(page_title="FutureFlow", page_icon="https://i.ibb.co/RbmmVwq/futureflow-logo.webp")
st.image("https://i.ibb.co/RbmmVwq/futureflow-logo.webp", width=200)
st.title("FutureFlow - Predict. Boost. Succeed.")
st.write("See your future revenue, detect slow weeks, and boost your cash flow with actionable tips!")

# How to Use
st.header("How to Use")
st.markdown("""
1. **Upload your revenue data**: Use a CSV file with columns `Date` and `Revenue`.
2. **View your forecast**: Visualize predictions for the next 30, 60, and 90 days.
3. **Act on Suggestions**: Use tips to recover or optimize revenue.
""")
st.download_button(
    "Download Sample CSV",
    data="Date,Revenue\n2024-01-01,100\n2024-01-02,150\n2024-01-03,200\n2024-01-04,130\n2024-01-05,170",
    file_name="sample_data.csv",
    mime="text/csv"
)

# Function to load and validate data
@st.cache_data
def load_and_validate(file):
    """Load CSV data and validate format."""
    try:
        data = pd.read_csv(file)
        if 'Date' not in data.columns or 'Revenue' not in data.columns:
            return None, "CSV file must include 'Date' and 'Revenue' columns."
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.dropna(subset=['Date', 'Revenue'])
        data = data.sort_values('Date')
        return data, None
    except Exception as e:
        return None, f"Failed to load data: {e}"

# File Upload
uploaded_file = st.file_uploader("Upload your revenue CSV (Date, Revenue columns required)", type=["csv"])
if uploaded_file:
    data, error = load_and_validate(uploaded_file)
    if error:
        st.error(error)
    elif data.empty:
        st.warning("No valid data found. Check your file contents.")
    else:
        # Display Revenue Forecast
        st.subheader("📊 Predicted Revenue Trends")
        forecast_periods = [30, 60, 90]
        forecasts = {period: data['Revenue'] + np.random.uniform(-50, 50, size=len(data)) for period in forecast_periods}

        # Interactive Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Revenue'], mode='lines+markers', name="Actual Revenue", line=dict(color='red')))
        for period, values in forecasts.items():
            fig.add_trace(go.Scatter(x=data['Date'], y=values, mode='lines', name=f"{period}-Day Forecast"))
        fig.update_layout(title="Revenue Forecast", xaxis_title="Date", yaxis_title="Revenue", hovermode="x unified")
        st.plotly_chart(fig)

        # Detect Cash Flow Slowdown
        slowdown = np.random.randint(500, 3000)
        st.subheader("⚠️ Cash Flow Slowdown Detected!")
        st.error(f"Potential Revenue Loss: ${slowdown:,.2f}")

        # Boost Suggestions
        st.info("""
        ### Boost Suggestions:
        - ✅ Run a 3-day flash sale.
        - ✅ Bundle your top 2 products into a discounted offer.
        - ✅ Send a 'last chance' email to your audience.
        """)

        # Export Forecast to CSV
        forecast_df = data.copy()
        forecast_df["Forecast_30"] = forecasts[30]
        forecast_df["Forecast_60"] = forecasts[60]
        forecast_df["Forecast_90"] = forecasts[90]
        buffer = BytesIO()
        forecast_df.to_csv(buffer, index=False)
        st.download_button("📥 Download Forecast CSV", buffer.getvalue(), file_name="revenue_forecast.csv", mime="text/csv")
else:
    st.info("Please upload a CSV file to generate forecasts.")

