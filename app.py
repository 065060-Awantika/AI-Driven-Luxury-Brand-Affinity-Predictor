import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- PREMIUM PAGE CONFIG ---
st.set_page_config(page_title="Aura Luxury Intelligence", page_icon="💎", layout="wide")

# --- CUSTOM CSS FOR PROFESSOR-LEVEL POLISH ---
st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    .metric-card { background: #1e2130; padding: 20px; border-radius: 10px; border-left: 5px solid #ffd700; }
    div[data-testid="stMetricValue"] { color: #ffd700; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('aura_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# --- SIDEBAR: CONTROL CENTER ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/6171/6171565.png", width=100)
st.sidebar.title("Customer Intelligence")
st.sidebar.markdown("Adjust parameters to simulate customer profiles.")

spend = st.sidebar.number_input("Transaction Value ($)", 0, 100000, 450)
prev_purchases = st.sidebar.slider("Historical Loyalty (Orders)", 0, 50, 12)
rating = st.sidebar.slider("Sentiment Score (1-5 Stars)", 1.0, 5.0, 4.2)

# --- MAIN DASHBOARD ---
st.title("💎 Aura: Luxury Brand Affinity Predictor")
st.write("Predicting Customer Lifetime Value using Deep Learning Neural Networks.")
st.markdown("---")

# Layout: 3 Columns for Metrics
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Model Architecture", "Sequential ANN")
with m2:
    st.metric("Optimization", "Adam / Dropout")
with m3:
    st.metric("Input Features", "3 Dimensional")

st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📍 Real-time Prediction")
    if st.button("RUN NEURAL ANALYSIS"):
        # Prediction Logic
        input_data = np.array([[spend, rating, prev_purchases]])
        scaled_features = scaler.transform(input_data)
        score = model.predict(scaled_features)[0][0]
        
        # Display Results
        st.write(f"### Probability of High-Value Advocacy: `{score*100:.2f}%`")
        st.progress(float(score))
        
        if score > 0.7:
            st.success("🏆 **CATEGORY: PLATINUM ADVOCATE**")
            st.balloons()
            st.info("**Strategy:** High-touch engagement. Send invitation to Private Yacht Event.")
        elif score > 0.4:
            st.warning("🥈 **CATEGORY: PREMIUM PROSPECT**")
            st.info("**Strategy:** Personalized cross-selling for limited edition accessories.")
        else:
            st.error("📉 **CATEGORY: STANDARD SHOPPER**")
            st.info("**Strategy:** Automated drip-campaign with introductory discount codes.")

with col_right:
    st.subheader("📊 Profile Visualization")
    # Radar Chart for Professor-level "Wow"
    categories = ['Spend', 'Sentiment', 'Loyalty']
    values = [spend/1000, rating, prev_purchases/10] # Normalized for chart
    
    fig = go.Figure(data=go.Scatterpolar(
      r=values,
      theta=categories,
      fill='toself',
      line_color='#ffd700'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Developed for FSM Deep Learning & Modeling Project | AI-Driven Consumer Insights")
