import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Crop Production Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #EDF7ED;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 20px;
    }
    .metrics-card {
        background-color: #F1F8E9;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #388E3C;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------ Sidebar for navigation ------------------------
with st.sidebar:
    st.markdown("# 🌾 Navigation")
    page = st.radio("Select a page", ["🏠 Home", "📊 Data Exploration", "🔮 Prediction", "📈 Results Dashboard"])
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This app uses machine learning to predict crop production based on various environmental and agricultural factors.")
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Explore the dataset in the Data Exploration page
    2. Enter your farm details in the Prediction page
    3. View results and recommendations in the Dashboard
    """)

# ------------------------ Load dataset ------------------------
@st.cache_data
def load_data():
    try:
        # Try to locate the CSV file in the current directory or a data subdirectory
        possible_paths = ["Crop Prediction dataset.csv", "data/Crop Prediction dataset.csv"]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                return df
                
        # If file not found, use a sample dataset for demo
        st.warning("Dataset not found. Using sample data for demonstration.")
        
        # Create sample data
        states = ["Maharashtra", "Punjab", "Karnataka", "Tamil Nadu", "Uttar Pradesh"]
        districts = ["District1", "District2", "District3", "District4", "District5"]
        seasons = ["Kharif", "Rabi", "Whole Year", "Summer"]
        crops = ["Rice", "Wheat", "Maize", "Potato", "Cotton"]
        
        sample_data = {
            "State_Name": np.random.choice(states, 1000),
            "District_Name": np.random.choice(districts, 1000),
            "Season": np.random.choice(seasons, 1000),
            "Crop": np.random.choice(crops, 1000),
            "Crop_Year": np.random.randint(2010, 2023, 1000),
            "Temperature": np.random.uniform(15, 40, 1000),
            "Humidity": np.random.uniform(50, 90, 1000),
            "Soil_Moisture": np.random.uniform(20, 80, 1000),
            "Area": np.random.uniform(1, 100, 1000),
            "Production": np.random.uniform(5, 500, 1000)
        }
        
        return pd.DataFrame(sample_data)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

# ------------------------ Prepare Machine Learning Model ------------------------
@st.cache_data
def prepare_model(df):
    if df is None:
        return None, None, None, None
    
    # Drop any rows with missing Production values
    df = df.dropna(subset=["Production"])
    
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
    
    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, label_encoders, X.columns, rmse, r2

model_data = prepare_model(df)
if model_data is not None and len(model_data) == 6:
    model, scaler, label_encoders, feature_names, rmse, r2 = model_data

# ------------------------ Home Page ------------------------
if page == "🏠 Home":
    st.markdown('<p class="main-header">🌾 Smart Agriculture: Crop Production Predictor</p>', unsafe_allow_html=True)
    
    # Try to load the image
    try:
        image_paths = ["agri.jpg", "images/agri.jpg", "data/agri.jpg"]
        image_loaded = False
        
        for path in image_paths:
            if os.path.exists(path):
                image = Image.open(path)
                st.image(image, use_column_width=True)
                image_loaded = True
                break
                
        if not image_loaded:
            # If no image is found, display a colorful divider
            st.markdown("""
            <div style="background: linear-gradient(to right, #76b852, #8DC26F);
                        height: 10px;
                        border-radius: 5px;
                        margin: 20px 0px;">
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.warning(f"Could not load image: {e}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="sub-header">Welcome to the Crop Production Prediction System</p>', unsafe_allow_html=True)
        st.markdown("""
        This intelligent system helps farmers and agricultural experts predict crop yields based on:
        
        * Environmental factors (temperature, humidity, soil moisture)
        * Location data (state, district)
        * Seasonal variations
        * Crop type and cultivated area
        
        Navigate through the sidebar to explore data, make predictions, and view insights.
        """)
    
    with col2:
        st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
        st.metric(label="Data Points", value=f"{len(df):,}" if df is not None else "N/A")
        st.metric(label="States", value=len(df["State_Name"].unique()) if df is not None else "N/A")
        st.metric(label="Crops", value=len(df["Crop"].unique()) if df is not None else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    st.markdown('<p class="sub-header">Quick Statistics</p>', unsafe_allow_html=True)
    
    if df is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
            st.metric(label="Average Production (tons)", 
                      value=f"{df['Production'].mean():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
            st.metric(label="Average Humidity (%)", 
                      value=f"{df['Humidity'].mean():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
            st.metric(label="Average Temperature (°C)", 
                      value=f"{df['Temperature'].mean():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Model performance metrics if available
        if 'rmse' in locals() and 'r2' in locals():
            st.markdown("### Model Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
                st.metric(label="RMSE", value=f"{rmse:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
                st.metric(label="R² Score", value=f"{r2:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)

# ------------------------ Data Exploration Page ------------------------
elif page == "📊 Data Exploration":
    st.markdown('<p class="main-header">📊 Data Exploration</p>', unsafe_allow_html=True)
    
    if df is not None:
        # Display raw data with pagination
        st.markdown('<p class="sub-header">Dataset Preview</p>', unsafe_allow_html=True)
        
        with st.expander("View Raw Data"):
            st.dataframe(df.head(100), use_container_width=True)
            
            if st.button("Download Full Dataset"):
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Click to Download",
                    data=csv,
                    file_name="crop_prediction_data.csv",
                    mime="text/csv"
                )
        
        # Summary statistics
        st.markdown('<p class="sub-header">Summary Statistics</p>', unsafe_allow_html=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if st.checkbox("Show Summary Statistics"):
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Visualizations
        st.markdown('<p class="sub-header">Data Visualizations</p>', unsafe_allow_html=True)
        
        viz_type = st.selectbox(
            "Choose Visualization", 
            ["Production by State", "Production by Crop", "Production by Season", 
             "Temperature vs Production", "Humidity vs Production", "Correlation Matrix"]
        )
        
        if viz_type == "Production by State":
            fig = px.bar(
                df.groupby("State_Name")["Production"].mean().reset_index().sort_values("Production", ascending=False),
                x="State_Name",
                y="Production",
                title="Average Production by State",
                color="Production",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Production by Crop":
            fig = px.bar(
                df.groupby("Crop")["Production"].mean().reset_index().sort_values("Production", ascending=False),
                x="Crop",
                y="Production",
                title="Average Production by Crop",
                color="Production",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Production by Season":
            fig = px.pie(
                df.groupby("Season")["Production"].sum().reset_index(),
                values="Production",
                names="Season",
                title="Total Production by Season",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Temperature vs Production":
            fig = px.scatter(
                df.sample(min(1000, len(df))),
                x="Temperature",
                y="Production",
                color="Crop",
                size="Area",
                hover_data=["State_Name", "Season"],
                title="Temperature vs Production",
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Humidity vs Production":
            fig = px.scatter(
                df.sample(min(1000, len(df))),
                x="Humidity",
                y="Production",
                color="Crop",
                size="Area",
                hover_data=["State_Name", "Season"],
                title="Humidity vs Production",
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Correlation Matrix":
            corr = df[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="Viridis",
                title="Feature Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
            
# ------------------------ Prediction Page ------------------------
elif page == "🔮 Prediction":
    st.markdown('<p class="main-header">🔮 Crop Production Prediction</p>', unsafe_allow_html=True)
    
    if df is not None and 'label_encoders' in locals():
        st.markdown("Enter the following details to predict estimated crop production for your region.")
        
        # Create a nice form layout
        col1, col2 = st.columns(2)
        
        with col1:
            state_name = st.selectbox("State", label_encoders["State_Name"].classes_)
            district_name = st.selectbox("District", label_encoders["District_Name"].classes_)
            season = st.selectbox("Season", label_encoders["Season"].classes_)
            crop = st.selectbox("Crop", label_encoders["Crop"].classes_)
        
        with col2:
            crop_year = st.number_input("Crop Year", min_value=2000, max_value=2050, value=2023)
            temperature = st.slider("Temperature (°C)", min_value=10.0, max_value=45.0, value=25.0, step=0.1)
            humidity = st.slider("Humidity (%)", min_value=30.0, max_value=100.0, value=70.0, step=0.1)
            soil_moisture = st.slider("Soil Moisture (%)", min_value=10.0, max_value=90.0, value=35.0, step=0.1)
            area = st.number_input("Area (hectares)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1)
        
        predict_button = st.button("Predict Production", use_container_width=True)
        
        if predict_button:
            try:
                # Create input data frame
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
                
                # Match training data column order
                input_data = input_data[feature_names]
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                
                # Store prediction for dashboard
                if 'predictions' not in st.session_state:
                    st.session_state.predictions = []
                
                # Add current prediction to history
                st.session_state.predictions.append({
                    "State": state_name,
                    "District": district_name,
                    "Season": season,
                    "Crop": crop,
                    "Year": crop_year,
                    "Temperature": temperature,
                    "Humidity": humidity,
                    "Soil_Moisture": soil_moisture,
                    "Area": area,
                    "Predicted_Production": prediction
                })
                
                # Display prediction with a nice UI
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"### 🌾 Estimated Crop Production: {prediction:.2f} tons")
                
                # Calculate yield per hectare
                yield_per_hectare = prediction / area
                st.markdown(f"#### 📊 Yield per Hectare: {yield_per_hectare:.2f} tons/ha")
                
                # Provide some context or recommendations
                if yield_per_hectare > 5:
                    st.markdown("✅ **Great productivity!** Your predicted yield is excellent.")
                elif yield_per_hectare > 2:
                    st.markdown("✅ **Good productivity.** Your predicted yield is above average.")
                else:
                    st.markdown("⚠️ **Consider optimizing conditions.** Your predicted yield could be improved.")
                    
                    # Simple recommendations based on input values
                    if temperature < 15 or temperature > 35:
                        st.markdown("🌡️ Temperature may not be optimal for this crop.")
                    if humidity < 50:
                        st.markdown("💧 Consider increasing humidity or irrigation.")
                    if soil_moisture < 25:
                        st.markdown("🌱 Soil moisture levels could be increased.")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show feature importance for this prediction
                feature_importance = model.feature_importances_
                feat_imp_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importance
                }).sort_values(by='Importance', ascending=False)
                
                st.markdown("### Feature Importance")
                fig = px.bar(
                    feat_imp_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Factors Affecting Crop Production"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in prediction: {e}")
    else:
        st.error("Model or data not available. Please check if the dataset was loaded correctly.")

# ------------------------ Results Dashboard ------------------------
elif page == "📈 Results Dashboard":
    st.markdown('<p class="main-header">📈 Results Dashboard</p>', unsafe_allow_html=True)
    
    if 'predictions' in st.session_state and len(st.session_state.predictions) > 0:
        st.markdown('<p class="sub-header">Your Prediction History</p>', unsafe_allow_html=True)
        
        # Convert predictions to DataFrame
        pred_df = pd.DataFrame(st.session_state.predictions)
        
        # Display prediction history table
        with st.expander("View All Predictions"):
            st.dataframe(pred_df, use_container_width=True)
        
        # Visualization options
        viz_option = st.selectbox(
            "Visualize Predictions By:", 
            ["Crop Type", "Season", "State", "Temperature vs Production", "Area vs Production"]
        )
        
        if viz_option == "Crop Type":
            fig = px.bar(
                pred_df.groupby("Crop")["Predicted_Production"].mean().reset_index(),
                x="Crop",
                y="Predicted_Production",
                title="Average Predicted Production by Crop Type",
                color="Predicted_Production",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Season":
            fig = px.pie(
                pred_df,
                values="Predicted_Production",
                names="Season",
                title="Predicted Production by Season",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "State":
            fig = px.bar(
                pred_df.groupby("State")["Predicted_Production"].mean().reset_index(),
                x="State",
                y="Predicted_Production",
                title="Average Predicted Production by State",
                color="Predicted_Production",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Temperature vs Production":
            fig = px.scatter(
                pred_df,
                x="Temperature",
                y="Predicted_Production",
                color="Crop",
                size="Area",
                hover_data=["State", "Season"],
                title="Temperature vs Predicted Production",
                trendline="ols",
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Area vs Production":
            fig = px.scatter(
                pred_df,
                x="Area",
                y="Predicted_Production",
                color="Crop",
                size="Predicted_Production",
                hover_data=["State", "Season"],
                title="Area vs Predicted Production",
                trendline="ols",
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Productivity analysis
        st.markdown('<p class="sub-header">Productivity Analysis</p>', unsafe_allow_html=True)
        
        # Calculate yield per hectare
        pred_df["Yield_per_Hectare"] = pred_df["Predicted_Production"] / pred_df["Area"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                pred_df,
                x="Crop",
                y="Yield_per_Hectare",
                title="Yield Distribution by Crop",
                color="Crop",
                points="all"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.box(
                pred_df,
                x="Season",
                y="Yield_per_Hectare",
                title="Yield Distribution by Season",
                color="Season",
                points="all"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Production recommendations
        st.markdown('<p class="sub-header">Recommendations</p>', unsafe_allow_html=True)
        
        # Find the crop with the highest average yield
        best_crop = pred_df.groupby("Crop")["Yield_per_Hectare"].mean().idxmax()
        best_yield = pred_df.groupby("Crop")["Yield_per_Hectare"].mean().max()
        
        # Find the best season
        best_season = pred_df.groupby("Season")["Yield_per_Hectare"].mean().idxmax()
        
        st.markdown(f"""
        Based on your prediction history, here are some recommendations:
        
        * **Best Performing Crop**: {best_crop} with an average yield of {best_yield:.2f} tons/ha
        * **Best Growing Season**: {best_season}
        * **Optimal Conditions**: Consider maintaining soil moisture between 30-40% and humidity around 60-80% for better yields.
        """)
        
        # Download options
        if st.button("Download Prediction Report"):
            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="crop_prediction_report.csv",
                mime="text/csv"
            )
    else:
        st.info("No predictions made yet. Go to the Prediction page to make predictions.")
        
        # Display sample dashboard with demo data
        st.markdown('<p class="sub-header">Sample Dashboard</p>', unsafe_allow_html=True)
        
        # Create sample prediction data
        sample_predictions = pd.DataFrame({
            "Crop": ["Rice", "Wheat", "Maize", "Cotton", "Rice", "Wheat"],
            "Season": ["Kharif", "Rabi", "Kharif", "Whole Year", "Kharif", "Rabi"],
            "Area": [5.2, 3.8, 2.5, 4.0, 6.0, 2.7],
            "Predicted_Production": [23.4, 13.2, 8.5, 12.0, 27.0, 9.8],
            "Yield_per_Hectare": [4.5, 3.47, 3.4, 3.0, 4.5, 3.63]
        })
        
        fig = px.bar(
            sample_predictions.groupby("Crop")["Predicted_Production"].mean().reset_index(),
            x="Crop",
            y="Predicted_Production",
            title="Sample: Average Predicted Production by Crop Type",
            color="Predicted_Production",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Crop Production Predictor | Developed with ❤️ for Smart Agriculture</p>
</div>
""", unsafe_allow_html=True)
