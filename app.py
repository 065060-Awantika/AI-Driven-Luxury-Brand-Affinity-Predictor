import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="Aura Luxury Analytics", page_icon="💎", layout="wide")

# --- CUSTOM CSS FOR LUXURY FEEL ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { background-color: #ffd700; color: black; border-radius: 20px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
model = tf.keras.models.load_model('aura_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

# --- SIDEBAR INPUTS ---
st.sidebar.header("📊 Customer Demographics")
spend = st.sidebar.number_input("Total Purchase Amount (USD)", 0, 100000, 500)
prev_purchases = st.sidebar.slider("Historical Purchase Count", 0, 100, 10)
rating = st.sidebar.slider("Customer Satisfaction Rating", 1.0, 5.0, 4.5)

# --- MAIN DASHBOARD ---
st.title("💎 Aura: Luxury Brand Affinity Predictor")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Customer Overview")
    st.metric(label="Selected Spend", value=f"${spend}")
    st.metric(label="Loyalty Level", value=f"{prev_purchases} Orders")
    
    if st.button("🚀 Analyze Affinity"):
        features = np.array([[spend, rating, prev_purchases]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0][0]
        
        st.session_state['score'] = prediction

with col2:
    if 'score' in st.session_state:
        score = st.session_state['score']
        st.subheader("AI Prediction Result")
        
        # Visual Meter
        st.progress(float(score))
        
        if score > 0.5:
            st.success(f"### High Value Advocate (Confidence: {score*100:.1f}%)")
            st.balloons()
            st.info("💡 Strategy: Invite to Exclusive VIP Preview Events.")
        else:
            st.warning(f"### Potential Churn Risk (Confidence: {(1-score)*100:.1f}%)")
            st.info("💡 Strategy: Deploy Re-engagement Discount Campaign.")
            
        # Add a small chart for visual interest
        chart_data = pd.DataFrame({'Metrics': ['Score', 'Threshold'], 'Value': [score, 0.5]})
        st.bar_chart(chart_data.set_index('Metrics'))
    # Predict
    prediction = model.predict(scaled_features)[0][0]
    
    st.divider()
    if prediction > 0.5:
        st.success(f"### High Value Advocate! (Score: {prediction:.2f})")
        st.balloons()
    else:
        st.warning(f"### Standard Shopper (Score: {prediction:.2f})")
