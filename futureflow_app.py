import streamlit as st
import pandas as pd
import numpy as np
import datetime

# Step 1: Dynamic Date Handling (No Hard-Coded Dates)
today = datetime.date.today()
df = pd.DataFrame({'date': pd.date_range(start=today, periods=60)})

# Step 2: Allow User to Upload CSV Revenue Data
uploaded_file = st.file_uploader("📁 Upload your revenue CSV (Date, Revenue columns required)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ **Revenue Data Successfully Uploaded**", df)
else:
    st.write("📂 **Upload a CSV to see your cash flow forecast.**")
    # Example Data (only if no CSV uploaded)
    df['revenue'] = np.random.randint(100, 500, size=60)  # Example data

# Step 3: Calculate Forecasts (30, 60, 90 Day Forecasts)
if 'revenue' in df.columns:
    df['30_day_forecast'] = df['revenue'].rolling(window=30).mean()
    df['60_day_forecast'] = df['revenue'].rolling(window=60).mean()
    df['90_day_forecast'] = df['revenue'].rolling(window=90).mean()
    st.line_chart(df[['revenue', '30_day_forecast', '60_day_forecast', '90_day_forecast']])

# Step 4: Detect Cash Flow Dips (and Provide Boost Suggestions)
if any(df['revenue'] < 200):
    st.subheader('⚠️ **Cash Flow Slowdown Detected!**')
    potential_loss = df.loc[df['revenue'] < 200, 'revenue'].sum()
    st.write(f"🚨 **Potential Revenue Loss:** ${potential_loss}")
    st.write("💡 **Boost Suggestions:**")
    st.write("- Run a 3-day flash sale")
    st.write("- Bundle your top 2 products into a discounted offer")
    st.write("- Send a 'last chance' email to your audience")