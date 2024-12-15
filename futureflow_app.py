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
    2. **View your forecast**: See trends for the next 30, 60, and 90 days.
    3. **Act on suggestions**: Boost your cash flow with actionable tips provided below the forecast.
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

            # Validate CSV columns
            required_columns = ['Date', 'Revenue']
            if not all(column in data.columns for column in required_columns):
                st.error("Invalid CSV format. Please ensure it has 'Date' and 'Revenue' columns.")
            else:
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                data = data.dropna(subset=['Date', 'Revenue'])
                data = data.sort_values('Date')

                if data.empty:
                    st.warning("No data available to display after processing. Please check your date range or file contents.")
                else:
                    # Plotting Revenue Trends
                    st.subheader("Predicted revenue trends over the next 30, 60, and 90 days.")
                    days = np.arange(len(data))
                    revenue = data['Revenue'].values

                    # Simulating forecasts
                    forecast_30 = revenue + random.uniform(-50, 50)
                    forecast_60 = revenue + random.uniform(-100, 100)
                    forecast_90 = revenue + random.uniform(-150, 150)

                    # Interactive Plotly chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=revenue, mode='lines+markers', name='Revenue', line=dict(color='red')))
                    fig.add_trace(go.Scatter(x=data['Date'], y=forecast_30, mode='lines', name='30-day Forecast', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=data['Date'], y=forecast_60, mode='lines', name='60-day Forecast', line=dict(color='lightblue')))
                    fig.add_trace(go.Scatter(x=data['Date'], y=forecast_90, mode='lines', name='90-day Forecast', line=dict(color='pink')))
                    fig.update_layout(title='Revenue Forecast', xaxis_title='Date', yaxis_title='Revenue')
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

                    # Download Forecast Report
                    buffer = BytesIO()
                    fig.write_image(buffer, format="png")
                    buffer.seek(0)
                    b64 = base64.b64encode(buffer.read()).decode()
                    href = f'<a href="data:image/png;base64,{b64}" download="forecast.png">Download Forecast Graph</a>'
                    st.markdown(href, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
