import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="⚡ Electricity Demand Forecasting", layout="centered")

st.title("⚡ Electricity Demand Forecasting")
st.write("Upload a CSV file with electricity demand data to view LSTM-based predictions.")

uploaded_file = st.file_uploader("📁 Upload your dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("📊 Raw Data Preview")
        st.write(data.head())

        # Automatically pick numeric columns except date/time
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            st.error("❌ No numeric columns found for prediction.")
            st.stop()

        st.success(f"✅ Using columns {numeric_cols} for prediction")

        # Load model and get required input shape
        model = load_model("model.h5")
        sequence_length = 30  # You used 30 in training

        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[numeric_cols])

        # Create sequences
        X = []
        for i in range(sequence_length, len(scaled_data)):
            seq = scaled_data[i-sequence_length:i]
            if seq.shape[0] == sequence_length:
                X.append(seq)
        X = np.array(X)

        # Ensure correct shape
        if X.ndim == 2:
            X = np.expand_dims(X, axis=2)

        # Make prediction
        predictions_scaled = model.predict(X)

        # Inverse transform if possible (for single output)
        if predictions_scaled.shape[1] == 1:
            predictions = scaler.inverse_transform(
                np.hstack((predictions_scaled, np.zeros((len(predictions_scaled), len(numeric_cols)-1))))
            )[:, 0]
        else:
            predictions = predictions_scaled  # multi-output

        # Plot
        st.subheader("📈 Forecasted Demand")
        plt.figure(figsize=(10, 4))
        plt.plot(predictions, label="Predicted")
        plt.xlabel("Time Step")
        plt.ylabel("Predicted Value")
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
