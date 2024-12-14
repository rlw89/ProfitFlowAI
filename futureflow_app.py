import streamlit as st
import pandas as pd
import numpy as np
import datetime

# Logo and Header
st.image("https://ibb.co/nmvv5JF", width=200)
st.title("Futureflow - Predict. Boost. Succeed.")
st.markdown("See your future revenue, detect slow weeks, and boost your cash flow with actionable tips!")

# Step 1: Dynamic Date Handling
today = datetime.date.today()
df = pd.DataFrame({'date': pd.date_range(start=today, periods=60)})

# Step 2: Allow User to Upload CSV Revenue Data
uploaded_file = st.file_uploader("📁 Upload your revenue CSV (Date, Revenue columns required)", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if 'Date' not in df.columns or 'Revenue' not in df.columns:
            st.error("Your CSV must contain 'Date' and 'Revenue' columns.")
            st.stop()
        df['Date'] = pd.to_datetime(df['Date'])
        st.write("✅ **Revenue Data Successfully Uploaded**", df)
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.write("📂 **Upload a CSV to see your cash flow forecast.**")
    # Example Data (only if no CSV uploaded)
    df['revenue'] = np.random.randint(100, 500, size=60)  # Example data
    df['Date'] = df['date']

# Step 3: Calculate Forecasts (30, 60, 90 Day Forecasts)
if 'revenue' in df.columns:
    df['30_day_forecast'] = df['revenue'].rolling(window=30).mean()
    df['60_day_forecast'] = df['revenue'].rolling(window=60).mean()
    df['90_day_forecast'] = df['revenue'].rolling(window=90).mean()
    
    # Plot Forecast
    st.line_chart(df.set_index('Date')[['revenue', '30_day_forecast', '60_day_forecast', '90_day_forecast']])
    st.caption("Predicted revenue trends over the next 30, 60, and 90 days.")

# Step 4: Detect Cash Flow Dips and Provide Boost Suggestions
if any(df['revenue'] < 200):
    st.subheader('⚠️ **Cash Flow Slowdown Detected!**')
    potential_loss = df.loc[df['revenue'] < 200, 'revenue'].sum()
    st.write(f"🚨 **Potential Revenue Loss:** ${potential_loss}")
    st.markdown("💡 **Boost Suggestions:**")
    st.write("✅ Run a 3-day flash sale.")
    st.write("✅ Bundle your top 2 products into a discounted offer.")
    st.write("✅ Send a 'last chance' email to your audience.")
else:
    st.success("🎉 Your projected revenue for the next 30 days is on track!")

# Step 5: Download Sample CSV
st.download_button(
    label="📂 Download Sample CSV",
    data="Date,Revenue\n2024-01-01,200\n2024-01-02,250\n",
    file_name="sample_revenue.csv",
    mime="text/csv",
)
