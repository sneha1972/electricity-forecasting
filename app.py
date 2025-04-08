import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Electricity Demand Forecasting", layout="wide")

st.title("⚡ Electricity Demand Forecasting")
st.write("Upload a CSV file with electricity demand data to view LSTM-based predictions.")

uploaded_file = st.file_uploader("📁 Upload your dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("📊 Raw Data Preview")
        st.write(data.head())

        # Select only numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("❌ Dataset must have at least two numeric columns for prediction.")
        else:
            selected_cols = numeric_cols[:2]
            st.success(f"✅ Using columns {selected_cols} for prediction")

            # Preprocessing
            selected_data = data[selected_cols].dropna()
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(selected_data)

            sequence_length = 30
            X = []
            for i in range(sequence_length, len(scaled_data)):
                seq = scaled_data[i - sequence_length:i]
                if seq.shape[0] == sequence_length:
                    X.append(seq)
            X = np.array(X)

            # Reshape if needed
            if X.ndim == 2:
                X = np.expand_dims(X, axis=2)

            model = load_model("model.h5")
            predictions = model.predict(X)

            # Inverse scale only the target (first column)
            dummy = np.zeros((predictions.shape[0], scaled_data.shape[1]))
            dummy[:, 0] = predictions[:, 0]
            inv_scaled = scaler.inverse_transform(dummy)[:, 0]

            # Plot
            st.subheader("📈 Forecasted Electricity Demand")
            plt.figure(figsize=(12, 6))
            plt.plot(inv_scaled, label="Predicted Demand (kWh)")
            plt.xlabel("Time Step")
            plt.ylabel("Electricity Demand (kWh)")
            plt.legend()
            st.pyplot(plt)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
