import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Aura Intelligence Platform", page_icon="💎", layout="wide")

# --- CUSTOM CSS ---
st.markdown("<style>.stAlert { border-radius: 10px; border: 1px solid #ffd700; }</style>", unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('aura_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# --- SIDEBAR & NAVIGATION ---
st.sidebar.title("💎 Aura Analytics")
app_mode = st.sidebar.selectbox("Select Capability", ["Individual Predictor", "Batch Intelligence"])

# --- MODE 1: INDIVIDUAL PREDICTOR (Your Current View) ---
if app_mode == "Individual Predictor":
    st.title("📍 Individual Consumer Intelligence")
    spend = st.sidebar.number_input("Transaction Value ($)", 0, 100000, 450)
    prev_purchases = st.sidebar.slider("Historical Loyalty", 0, 50, 12)
    rating = st.sidebar.slider("Sentiment Score", 1.0, 5.0, 4.2)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("RUN NEURAL ANALYSIS"):
            data = scaler.transform([[spend, rating, prev_purchases]])
            score = model.predict(data)[0][0]
            st.metric("Affinity Score", f"{score*100:.2f}%")
            
            # --- THE "WOW" FEATURE: AI INSIGHTS ---
            st.subheader("💡 Automated Strategy")
            if score > 0.7:
                st.info(f"**High-Value Prospect identified.** Recommended Action: Priority Customer Support and early access to the Winter Collection.")
            else:
                st.info(f"**Standard Profile.** Recommended Action: Targeted discount campaign on category-specific items.")

    with c2:
        # (Keep your Radar Chart code here)
        st.subheader("📊 Profile Visualization")
        # ... radar chart logic ...

# --- MODE 2: BATCH INTELLIGENCE (The Professor's Favorite) ---
else:
    st.title("📂 Batch Marketing Intelligence")
    st.write("Upload a customer list to generate a mass segmentation report.")
    
    uploaded_file = st.file_uploader("Upload CSV (Must have: Spend, Rating, Purchases)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Assuming the CSV has 3 columns matching our features
        X_batch = scaler.transform(df.iloc[:, :3])
        predictions = model.predict(X_batch)
        
        df['Affinity_Score'] = predictions
        df['Segment'] = df['Affinity_Score'].apply(lambda x: "High-Value" if x > 0.5 else "Standard")
        
        st.write("### Analysis Results")
        st.dataframe(df.style.highlight_max(axis=0, subset=['Affinity_Score'], color='#ffd700'))
        
        # Download Button for the Professor
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📩 Download Professional Report", data=csv, file_name="Aura_Analysis_Report.csv", mime="text/csv")
