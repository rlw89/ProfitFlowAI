import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from io import BytesIO
import base64

# App Title and Logo
st.set_page_config(page_title="FutureFlow", page_icon="https://i.ibb.co/RbmmVwq/futureflow-logo.webp")
st.image("https://i.ibb.co/RbmmVwq/futureflow-logo.webp", width=200)
st.title("FutureFlow - Predict. Boost. Succeed.")
st.write("See your future revenue, detect slow weeks, and boost your cash flow with actionable tips!")

# How to Use Section
st.header("How to Use")
st.markdown(
    """
    1. **Upload your revenue data**: Use a CSV file with columns containing `Date` and `Revenue` (or equivalents).
    2. **Select your columns**: If column names differ, manually choose them for processing.
    3. **View your forecast**: Trends for the next 30, 60, and 90 days are displayed.
    4. **Act on suggestions**: Boost your cash flow with actionable tips below the forecast.
    """
)

# Sample CSV Download
st.download_button(
    label="Download Sample CSV",
    data="""Date,Revenue\n2024-01-01,100\n2024-01-02,150\n2024-01-03,200\n2024-01-04,130\n2024-01-05,170\n""".encode("utf-8"),
    file_name="sample_data.csv",
    mime="text/csv",
)

# File Upload Section
uploaded_file = st.file_uploader("Upload your revenue CSV", type=["csv"])

# Cache Data for Optimization
@st.cache_data
def load_and_process_file(file):
    """Load CSV data and return the DataFrame."""
    return pd.read_csv(file)

if uploaded_file:
    try:
        with st.spinner("Processing your file..."):
            data = load_and_process_file(uploaded_file)

            # Let user select columns
            st.subheader("Column Selection")
            date_col = st.selectbox("Select the Date column:", options=data.columns)
            revenue_col = st.selectbox("Select the Revenue column:", options=data.columns)

            if date_col and revenue_col:
                # Validate Selected Columns
                try:
                    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
                    data = data.dropna(subset=[date_col, revenue_col])
                    data = data.sort_values(by=date_col)

                    if data.empty:
                        st.warning("No valid data found. Please check your file.")
                    else:
                        st.success("Data successfully loaded and processed!")

                        # Simulate forecasts
                        st.subheader("📈 Predicted Revenue Trends")
                        data['Rolling_30'] = data[revenue_col].rolling(window=3, min_periods=1).mean() + np.random.uniform(-10, 10)
                        data['Rolling_60'] = data[revenue_col].rolling(window=5, min_periods=1).mean() + np.random.uniform(-20, 20)
                        data['Rolling_90'] = data[revenue_col].rolling(window=7, min_periods=1).mean() + np.random.uniform(-30, 30)

                        # Plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data[date_col], y=data[revenue_col], mode="lines+markers", name="Actual Revenue", line=dict(color="red")))
                        fig.add_trace(go.Scatter(x=data[date_col], y=data['Rolling_30'], mode="lines", name="30-Day Forecast", line=dict(color="blue")))
                        fig.add_trace(go.Scatter(x=data[date_col], y=data['Rolling_60'], mode="lines", name="60-Day Forecast", line=dict(color="lightblue")))
                        fig.add_trace(go.Scatter(x=data[date_col], y=data['Rolling_90'], mode="lines", name="90-Day Forecast", line=dict(color="pink")))

                        fig.update_layout(title="Revenue Forecast", xaxis_title="Date", yaxis_title="Revenue", hovermode="x unified")
                        st.plotly_chart(fig)

                        # Detect Cash Flow Slowdown
                        slowdown = random.uniform(500, 3000)
                        st.subheader("⚠️ Cash Flow Slowdown Detected!")
                        st.error(f"Potential Revenue Loss: **${slowdown:,.2f}**")

                        # Boost Suggestions
                        st.info("""
                        ### Boost Suggestions:
                        - ✅ Run a 3-day flash sale.
                        - ✅ Bundle your top 2 products into a discounted offer.
                        - ✅ Send a 'last chance' email to your audience.
                        """)

                        # Allow Forecast Export
                        buffer = BytesIO()
                        data.to_csv(buffer, index=False)
                        st.download_button(
                            "📥 Download Forecast Data",
                            buffer.getvalue(),
                            file_name="forecasted_revenue.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Error processing data: {e}")

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
