import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------
# STEP 1: Generate Cash Flow Data
# ---------------------
np.random.seed(42)
daily_revenue = np.random.randint(100, 500, size=60)

df = pd.DataFrame({
    'date': pd.date_range(start='2024-10-01', periods=60),
    'revenue': daily_revenue
})

df['30_day_forecast'] = df['revenue'].rolling(window=30).mean()
df['60_day_forecast'] = df['revenue'].rolling(window=60).mean()

slow_weeks = df[df['revenue'] < 200]

def suggest_revenue_boost(revenue):
    if revenue < 200:
        suggestions = [
            "Run a 3-day flash sale on your top product.",
            "Offer a 20% discount on your lowest-selling product.",
            "Send an email blast to your audience with a time-sensitive offer.",
            "Launch a 'Buy 1 Get 1 Free' campaign for 48 hours.",
            "Offer a limited-time bundle for your top 2 products."
        ]
        return random.choice(suggestions)
    else:
        return "Revenue looks stable. No boost needed."

df['boost_suggestion'] = df['revenue'].apply(suggest_revenue_boost)

# ---------------------
# STEP 2: Build the Streamlit Web App
# ---------------------
st.title('💸 Futureflow - Your Cash Flow Predictor')

st.header('📅 Cash Flow Summary')
current_30_day_forecast = round(df['30_day_forecast'].iloc[-1], 2)
current_60_day_forecast = round(df['60_day_forecast'].iloc[-1], 2)

st.write(f"**30-Day Forecast:** ${current_30_day_forecast}")
st.write(f"**60-Day Forecast:** ${current_60_day_forecast}")

st.header('📈 Revenue Flow & Forecasts')
st.line_chart(df[['revenue', '30_day_forecast', '60_day_forecast']])

st.subheader('📉 Revenue Dips')
slow_weeks_display = df[df['revenue'] < 200][['date', 'revenue', 'boost_suggestion']]
st.dataframe(slow_weeks_display)

st.header('🚀 Revenue Boost Alerts')
st.write("If your cash flow is looking slow, here are some quick actions you can take to boost it:")

unique_suggestions = df['boost_suggestion'].unique()
for suggestion in unique_suggestions:
    if suggestion != "Revenue looks stable. No boost needed.":
        st.write(f"🟢 {suggestion}")

st.subheader('⚠️ Cash Flow Slowdown Detected!')
potential_loss = df.loc[df['revenue'] < 200, 'revenue'].sum()
st.write(f"💥 **Potential Revenue Loss:** ${potential_loss}")

if st.button('💡 Boost My Revenue Now'):
    st.write("Here are some specific steps you can take to increase your revenue this week:")
    for i in range(3):
        suggestion = random.choice([
            "Run a 3-day flash sale on your top product.",
            "Offer a 20% discount on your lowest-selling product.",
            "Send an email blast to your audience with a time-sensitive offer.",
            "Launch a 'Buy 1 Get 1 Free' campaign for 48 hours.",
            "Offer a limited-time bundle for your top 2 products."
        ])
        st.write(f"👉 {suggestion}")
