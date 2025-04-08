import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 🎨 Page Configuration
st.set_page_config(
    page_title="👑 Electricity Demand Forecasting",
    page_icon="👑",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 🌈 Custom Styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #ffe6e6, #ffffcc);
        padding: 2rem;
        border-radius: 20px;
    }
    h1 {
        color: #ff5733;
        text-align: center;
        font-family: 'Segoe UI';
    }
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 10px;
    }
    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

# 🧠 Load model
model = load_model("model.h5")

st.title("⚡ Electricity Demand Forecasting")
st.markdown("Upload a CSV file with electricity demand data to view LSTM-based predictions.")

# 📁 File upload
uploaded_file = st.file_uploader("📁 Upload your dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        st.subheader("📊 Raw Data Preview")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("❌ Please upload a dataset with at least 2 numeric columns.")
        else:
            col1 = st.selectbox("🧩 Choose the first feature", numeric_cols, index=0)
            col2 = st.selectbox("🧩 Choose the second feature", numeric_cols, index=1)

            if col1 == col2:
                st.warning("⚠️ Please select two different columns.")
            else:
                st.markdown(f"✅ Using columns ['{col1}', '{col2}'] for prediction")

                features = df[[col1, col2]].values

                # 🔢 Scaling
                scaler = MinMaxScaler()
                scaled_features = scaler.fit_transform(features)

                # 🧩 Create sequences
                sequence_length = 30
                X = []
                for i in range(sequence_length, len(scaled_features)):
                    seq = scaled_features[i - sequence_length:i]
                    if seq.shape == (sequence_length, 2):
                        X.append(seq)

                X = np.array(X)

                if X.shape[0] == 0:
                    st.error("❌ Not enough data for prediction. Please upload a longer dataset.")
                else:
                    # 🔮 Predict
                    predictions = model.predict(X)
                    predicted_values = scaler.inverse_transform(
                        np.hstack([predictions, np.zeros((predictions.shape[0], 1))])
                    )[:, 0]

                    # 🖼️ Plotting
                    st.subheader("📈 Electricity Demand Prediction")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(predicted_values, color='blue', linewidth=2)
                    ax.set_title("Predicted Electricity Demand", fontsize=16)
                    ax.set_xlabel("Time Steps")
                    ax.set_ylabel("Electricity Demand")
                    ax.grid(True)
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# 👣 Professional Footer
st.markdown("""
    <hr style="margin-top: 3rem; margin-bottom: 1rem;">
    <div style='text-align: center; color: #555; font-size: 14px;'>
        Developed by <strong>CSBS Final Year Students – GPREC</strong><br>
        <em>Electricity Demand Forecasting using LSTM</em><br>
        © 2025 All rights reserved.
    </div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
