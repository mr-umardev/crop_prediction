
# 🌾 Crop Production Predictor

A **Streamlit web app** that predicts estimated **crop production (in tons)** based on agricultural inputs like region, crop type, season, soil and weather conditions.

## 🚀 Live Demo

> https://cropprediction-1000.streamlit.app/ or run locally (see instructions below)

## 📌 Features

- 📊 Machine learning-based prediction using **Random Forest Regressor**
- 🌎 Inputs: **State, District, Crop, Season, Year, Temperature, Humidity, Soil Moisture, Area**
- 🌱 Built using **Streamlit** for interactive frontend
- 🔄 Real-time input handling with form-based submission
- ✅ Optimized for both local and cloud deployment

## 🧠 Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas & NumPy
- PIL (Pillow) for image rendering

## 📂 Dataset

The app uses a crop production dataset including:
- `State_Name`, `District_Name`, `Season`, `Crop`, `Crop_Year`, `Temperature`, `Humidity`, `Soil_Moisture`, `Area`, `Production`

> 📁 Make sure to place `Crop Prediction dataset.csv` and `agri.jpg` in the root directory.

