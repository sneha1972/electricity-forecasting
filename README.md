# ⚡ Electricity Demand Forecasting

A Streamlit-based web app that forecasts future electricity demand using an LSTM model trained on historical demand and temperature data.

## 🔍 Features

- Upload your own `.csv` file
- Predict electricity demand using LSTM
- Visualize actual vs predicted demand
- Simple and interactive UI

## 🗂 Project Files

- `app.py` – Streamlit app frontend
- `model.h5` – Trained LSTM model
- `requirements.txt` – Python dependencies
- `electricity_demand_x_company_2022_2024.csv` – (Optional) Sample dataset

## 📊 Input Format

Upload a `.csv` file with at least one column named:

- `Electricity_Demand_kWh`

Optionally, include:

- `Temperature_C`
- `Date` (for visual clarity)

## 📦 Requirements

