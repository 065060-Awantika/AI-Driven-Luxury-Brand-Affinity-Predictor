import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

# Load the saved model and scaler
model = tf.keras.models.load_model('aura_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="Aura Luxury Analytics", page_icon="💎")

st.title(" Aura: Luxury Brand Affinity Predictor")
st.markdown("---")

# User Inputs based on our training features
col1, col2 = st.columns(2)
with col1:
    spend = st.number_input("Purchase Amount (USD)", min_value=0, max_value=100000, value=500)
    prev_purchases = st.slider("Previous Purchases", 0, 50, 5)
with col2:
    rating = st.slider("Your Review Rating", 1.0, 5.0, 4.0)

if st.button("Analyze Customer Profile"):
    # Preprocess the input
    features = np.array([[spend, rating, prev_purchases]])
    scaled_features = scaler.transform(features)
    
    # Predict
    prediction = model.predict(scaled_features)[0][0]
    
    st.divider()
    if prediction > 0.5:
        st.success(f"### High Value Advocate! (Score: {prediction:.2f})")
        st.balloons()
    else:
        st.warning(f"### Standard Shopper (Score: {prediction:.2f})")
