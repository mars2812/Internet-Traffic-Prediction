import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

# Load the trained model without compiling
scaler = joblib.load('scaler.pkl')
df = pd.read_csv('internet_traffic_data.csv')  # Load the dataset

# Recalculate rolling_mean to match training data
df['rolling_mean'] = df['data_usage_mb'].rolling(window=7, min_periods=1).mean()

# Scale the dataset using the same features as in training
df_scaled = scaler.transform(df[['data_usage_mb', 'rolling_mean']])

seq_length = 10  # Define sequence length
last_sequence = df_scaled[-seq_length:].reshape(1, seq_length, 2)  # Match input shape

# Load and compile model
model = tf.keras.models.load_model('internet_traffic_model.h5', compile=False)
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam')  # Explicit loss

# Custom CSS styling
st.markdown(
    """
    <style>
        .main {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            padding: 10px;
            font-size: 16px;
            border-radius: 5px;
        }
        .stSuccess {
            color: green;
            font-size: 18px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("📡 Internet Traffic Prediction")
st.write("🔍 Predict the internet usage for the next day based on past data.")

if st.button("📊 Predict Next Day Usage"):
    try:
        last_sequence = df_scaled[-seq_length:].reshape(1, seq_length, 2)
        next_day_prediction = model.predict(last_sequence)
        next_day_inv = scaler.inverse_transform(np.hstack((next_day_prediction, np.zeros_like(next_day_prediction))))[:, 0]
        prediction = round(next_day_inv[0], 2)
        
        st.markdown(f"<p class='stSuccess'>Predicted Internet Usage: {prediction} MB</p>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        © 2025 Internet Traffic Predictor | Developed by Matrika Dhamala 
    </div>
""", unsafe_allow_html=True)