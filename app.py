import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Electricity Demand Forecasting", layout="centered")
st.title("⚡ Electricity Demand Forecasting")
st.write("Upload a CSV file with electricity demand data to view LSTM-based predictions.")

# File upload
uploaded_file = st.file_uploader("📁 Upload your dataset (.csv)", type="csv")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Raw Data Preview")
        st.dataframe(df.head())

        # Try to find demand column automatically
        demand_col = None
        for col in df.columns:
            if "demand" in col.lower():
                demand_col = col
                break

        if demand_col is None:
            demand_col = df.columns[0]  # fallback to first column

        st.info(f"Using column **'{demand_col}'** for prediction.")

        # Normalize
        scaler = MinMaxScaler()
        demand_scaled = scaler.fit_transform(df[demand_col].values.reshape(-1, 1))

        # Create sequences
        sequence_length = 30
        X = []
        for i in range(sequence_length, len(demand_scaled)):
            X.append(demand_scaled[i - sequence_length:i])
        X = np.array(X).reshape(-1, sequence_length, 1)

        # Predict
        predictions = model.predict(X)
        predictions_inverse = scaler.inverse_transform(predictions)

        # Plot results
        st.subheader("📈 Forecasted Electricity Demand")
        fig, ax = plt.subplots()
        ax.plot(predictions_inverse, label="Predicted", color="orange")
        ax.plot(df[demand_col].values[sequence_length:], label="Actual", alpha=0.6)
        ax.set_xlabel("Time")
        ax.set_ylabel("Demand")
        ax.legend()
        st.pyplot(fig)

        st.success("✅ Forecasting complete!")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("👆 Upload a CSV file to start.")
