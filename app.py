import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io

# Set Streamlit page config
st.set_page_config(page_title="Electricity Demand Forecasting", layout="centered")

# Title and file uploader
st.title("⚡ Electricity Demand Forecasting")
st.markdown("Upload a CSV file with electricity demand data to view LSTM-based predictions.")
uploaded_file = st.file_uploader("📁 Upload your dataset (.csv)", type=["csv"])

# Prediction parameters
sequence_length = 30

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Preview raw data
        st.subheader("📊 Raw Data Preview")
        st.dataframe(df.head())

        # Detect numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        feature_cols = numeric_df.columns.tolist()

        if len(feature_cols) < 1:
            st.error("❌ No numeric columns found for prediction.")
            st.stop()

        st.success(f"✅ Using columns {feature_cols} for prediction")

        # Scale the selected features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # Build sequences
        X = []
        for i in range(sequence_length, len(scaled_data)):
            seq = scaled_data[i - sequence_length:i]
            if seq.shape[0] == sequence_length:
                X.append(seq)

        X = np.array(X)

        # Ensure correct shape for LSTM
        if X.ndim == 2:
            X = np.expand_dims(X, axis=2)

        # Load model
        model = tf.keras.models.load_model("model.h5")

        # Check input shape match
        expected_shape = model.input_shape[-1]
        actual_shape = X.shape[-1]

        if expected_shape != actual_shape:
            st.error(f"❌ Model expects {expected_shape} features but dataset has {actual_shape}. Please upload matching data or retrain model.")
            st.stop()

        # Predict
        predictions = model.predict(X)

        # Inverse scale if possible
        prediction_col = 0
        if scaler.scale_.shape[0] > 1:
            predictions_rescaled = predictions * scaler.data_range_[prediction_col] + scaler.data_min_[prediction_col]
        else:
            predictions_rescaled = scaler.inverse_transform(predictions)

        # Display chart
        st.subheader("📈 Forecasted Electricity Demand")
        fig, ax = plt.subplots()
        ax.plot(predictions_rescaled, label="Predicted Demand", color="green")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Electricity Demand (kWh)")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
