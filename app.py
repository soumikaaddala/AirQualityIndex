import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # To load trained model

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("station_day.csv")  # Replace with actual file path
    
    return df

df = load_data()


df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year

# Print to verify
print(df[['Date', 'Year']].head())  # Ensure the transformation worked

# Now you can safely use 'Year'
yearly_aqi = df.groupby('Year')['AQI'].mean()

print(yearly_aqi)
# Load trained model
model = joblib.load("aqi_model.pkl")  # Replace with your trained model file

# Define function for AQI prediction
def predict_aqi(year, pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene, xylene, month):
    input_data = pd.DataFrame([[pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene, xylene, year, month]], 
                              columns=['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'Year', 'Month'])
    predicted_aqi = model.predict(input_data)[0]
    return predicted_aqi

# Health risk mapping
def health_risk(aqi):
    if aqi <= 50:
        return "Good - No health risk"
    elif aqi <= 100:
        return "Moderate - Minor breathing issues for sensitive groups"
    elif aqi <= 150:
        return "Unhealthy for sensitive groups - Asthma patients affected"
    elif aqi <= 200:
        return "Unhealthy - Risk of breathing problems"
    elif aqi <= 300:
        return "Very Unhealthy - Lung infections and severe asthma attacks"
    else:
        return "Hazardous - Severe respiratory diseases like bronchitis"

# Streamlit UI
st.title("🌍 Air Quality Index (AQI) Prediction")

# User selects station
station_ids = df['StationId'].unique()
selected_station = st.selectbox("Select Monitoring Station", station_ids)

# User inputs future year & month
year = st.number_input("Enter Future Year", min_value=2024, max_value=2050, value=2030)
month = st.number_input("Enter Month (1-12)", min_value=1, max_value=12, value=6)

# Get average pollution values for selected station
station_data = df[df['StationId'] == selected_station].select_dtypes(include=['number']).mean()

# Predict AQI when user clicks
if st.button("Predict AQI"):
    future_aqi = predict_aqi(year, *station_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']], month)
    risk = health_risk(future_aqi)
    st.success(f"Predicted AQI for {selected_station} in {year}: {future_aqi}")
    st.warning(f"Health Risk: {risk}")

# Visualize AQI Trends
st.subheader("📊 AQI Trend Over the Years")
yearly_aqi = df.groupby('Year')['AQI'].mean()
fig, ax = plt.subplots()
sns.lineplot(x=yearly_aqi.index, y=yearly_aqi.values, marker="o", color='red', ax=ax)
ax.set_xlabel("Year")
ax.set_ylabel("Average AQI")
ax.set_title("AQI Trend Over the Years in India")
st.pyplot(fig)
