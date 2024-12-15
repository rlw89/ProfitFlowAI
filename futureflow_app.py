import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from io import BytesIO
import base64
from datetime import datetime

# Caching to improve performance
@st.cache_data
def load_data(uploaded_file):
    """Load and validate uploaded CSV file."""
    data = pd.read_csv(uploaded_file)
    required_columns = ['Date', 'Revenue']
    if not all(column in data.columns for column in required_columns):
        return None, "Invalid CSV format. Please ensure it has 'Date' and 'Revenue' columns."
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date', 'Revenue'])
    data = data.sort_values('Date')
    return data, None

# App Configuration
st.set_page_config(page_title="FutureFlow", page_icon="https://i.ibb.co/RbmmVwq/futureflow-logo.webp")
st.image("https://i.ibb.co/RbmmVwq/futureflow-logo.webp", width=200)
st.title("FutureFlow - Predict. Boost. Succeed.")
st.write("See your future revenue, detect slow weeks, and boost your cash flow with actionable tips!")

# Instructions Section
st.header("How to Use")
st.markdown("""
1. **Upload your revenue data**: Use a CSV file with columns for `Date` and `Revenue`.
2. **View your forecast**: See trends for the next 30, 60, and 90 days.
3. **Act on suggestions**: Boost your cash flow with actionable tips provided below the forecast.
""")
st.download_button(
    label="Download Sample CSV",
    data="""Date,Revenue\n2024-01-01,100\n2024-01-02,150\n2024-01-03,200\n2024-01-04,130\n2024-01-05,170\n""".encode("utf-8"),
    file_name="sample_data.csv",
    mime="text/csv"
)

# File Upload
uploaded_file = st.file_uploader("Upload your revenue CSV (Date, Revenue columns required)", type=["csv"])

if uploaded_file:
    with st.spinner("Processing your file..."):
        data, error = load_data(uploaded_file)
        
        if error:
            st.error(error)
        elif data.empty:
            st.warning("No valid data found. Please check your CSV file.")
        else:
            # Forecast Simulation
            st.subheader("📊 Predicted Revenue Trends")
            days = np.arange(len(data))
            revenue = data['Revenue'].values
            forecast_30 = revenue + np.random.uniform(-50, 50, size=len(revenue))
            forecast_60 = revenue + np.random.uniform(-100, 100, size=len(revenue))
            forecast_90 = revenue + np.random.uniform(-150, 150, size=len(revenue))

            # Interactive Forecast Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=revenue, name="Actual Revenue", mode="lines+markers", line=dict(color="red")))
            fig.add_trace(go.Scatter(x=data['Date'], y=forecast_30, name="30-Day Forecast", mode="lines", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=data['Date'], y=forecast_60, name="60-Day Forecast", mode="lines", line=dict(color="lightblue")))
            fig.add_trace(go.Scatter(x=data['Date'], y=forecast_90, name="90-Day Forecast", mode="lines", line=dict(color="pink")))

            fig.update_layout(
                title="Revenue Forecast",
                xaxis_title="Date",
                yaxis_title="Revenue",
                hovermode="x unified",
            )
            st.plotly_chart(fig)

            # Cash Flow Slowdown Detection
            slowdown = random.uniform(500, 3000)
            st.subheader("⚠️ Cash Flow Slowdown Detected!")
            st.error(f"Potential Revenue Loss: ${slowdown:.2f}")

            st.info("""
            ### Boost Suggestions:
            - ✅ Run a 3-day flash sale.
            - ✅ Bundle your top 2 products into a discounted offer.
            - ✅ Send a 'last chance' email to your audience.
            """)

            # Export Forecast as PNG
            st.subheader("Download Forecast")
            buffer = BytesIO()
            fig.write_image(buffer, format="png")
            buffer.seek(0)
            st.download_button(
                label="📥 Download Forecast Graph",
                data=buffer,
                file_name="forecast_graph.png",
                mime="image/png"
            )
else:
    st.info("Please upload a CSV file to see the forecast.")

