import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# ------------------------ Load and display image ------------------------
image = Image.open("agri.jpg")  # Make sure agri.jpg is in the same directory
st.image(image, use_container_width=True, caption="Smart Agriculture - Crop Production Predictor")

# ------------------------ Streamlit Title and Info ------------------------
st.title("🌾 Crop Production Predictor")
st.markdown("Enter the following details to predict estimated crop production for your region.")

# ------------------------ Load dataset ------------------------
df = pd.read_csv("Crop Prediction dataset.csv")
df = df.dropna(subset=["Production"])  # Drop rows with missing Production

# Encode categorical features
categorical_cols = ["State_Name", "District_Name", "Season", "Crop"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare data
X = df.drop(columns=["Production"])
y = df["Production"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------ User input ------------------------
state_name = st.selectbox("State", label_encoders["State_Name"].classes_)
district_name = st.selectbox("District", label_encoders["District_Name"].classes_)
season = st.selectbox("Season", label_encoders["Season"].classes_)
crop = st.selectbox("Crop", label_encoders["Crop"].classes_)

crop_year = st.number_input("Crop Year", min_value=2000, max_value=2050, value=2023)
temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=70.0)
soil_moisture = st.number_input("Soil Moisture (%)", value=35.0)
area = st.number_input("Area (hectares)", value=1.0)

# ------------------------ Prediction ------------------------
if st.button("Predict Production"):
    try:
        input_data = pd.DataFrame([{
            "State_Name": label_encoders["State_Name"].transform([state_name])[0],
            "District_Name": label_encoders["District_Name"].transform([district_name])[0],
            "Season": label_encoders["Season"].transform([season])[0],
            "Crop": label_encoders["Crop"].transform([crop])[0],
            "Crop_Year": crop_year,
            "Temperature": temperature,
            "Humidity": humidity,
            "Soil_Moisture": soil_moisture,
            "Area": area
        }])

        # Match column order
        input_data = input_data[X.columns]
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.success(f"🌾 Estimated Crop Production: **{prediction:.2f} tons**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
