import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Electricity Demand Forecasting", layout="centered")
st.title("⚡ Electricity Demand Forecasting")
st.write("Upload a CSV file with historical electricity demand data to view LSTM-based predictions.")

uploaded_file = st.file_uploader("📁 Upload your electricity_demand_x_company_2022_2024.csv", type="csv")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Raw Data Preview")
        st.dataframe(df.head())

        if "Electricity_Demand_kWh" not in df.columns:
            st.error("❌ CSV must contain a column named 'Electricity_Demand_kWh'")
        else:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df[["Electricity_Demand_kWh", "Temperature_C"]])

            seq_len = 30
            X = []
            for i in range(seq_len, len(scaled)):
                X.append(scaled[i - seq_len:i])

            X = np.array(X)
            predictions = model.predict(X)

            padded_predictions = np.concatenate([predictions, np.zeros((predictions.shape[0], 1))], axis=1)
            inverse_predictions = scaler.inverse_transform(padded_predictions)[:, 0]

            st.subheader("📈 Forecasted Electricity Demand")
            fig, ax = plt.subplots()
            ax.plot(df["Electricity_Demand_kWh"].values[seq_len:], label="Actual", alpha=0.6)
            ax.plot(inverse_predictions, label="Predicted", color="orange")
            ax.set_xlabel("Time")
            ax.set_ylabel("Electricity Demand (kWh)")
            ax.legend()
            st.pyplot(fig)

            st.success("✅ Forecasting complete!")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👆 Please upload your CSV file to begin.")
