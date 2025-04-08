import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="⚡ Electricity Demand Forecasting")
st.title("⚡ Electricity Demand Forecasting")
st.markdown("Upload a CSV file with electricity demand data to view LSTM-based predictions.")

uploaded_file = st.file_uploader("📁 Upload your dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.subheader("📊 Raw Data Preview")
        st.dataframe(df.head())

        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        feature_cols = numeric_df.columns.tolist()

        # Match model input shape
        model = tf.keras.models.load_model("model.h5")
        expected_shape = model.input_shape[-1]

        if len(feature_cols) < expected_shape:
            st.error(f"❌ Model expects {expected_shape} features but dataset has {len(feature_cols)}. Please upload matching data or retrain model.")
        else:
            # Trim or select exact number of features
            feature_cols = feature_cols[:expected_shape]
            numeric_df = numeric_df[feature_cols]
            st.success(f"✅ Using columns {feature_cols} for prediction")

            # Preprocessing
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(numeric_df)

            sequence_length = 30
            X = []
            for i in range(sequence_length, len(scaled_data)):
                seq = scaled_data[i - sequence_length:i]
                if seq.shape[0] == sequence_length:
                    X.append(seq)

            X = np.array(X)

            if X.ndim == 2:
                X = np.expand_dims(X, axis=2)

            # Prediction
            predictions = model.predict(X)
            predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], scaled_data.shape[1] - 1)))))
            predicted_demand = predictions[:, 0]

            st.subheader("📈 Predicted Electricity Demand")
            result_df = pd.DataFrame({"Predicted_Demand": predicted_demand})
            st.line_chart(result_df)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("Please upload a CSV file to continue.")
