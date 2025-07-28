import streamlit as st
import pandas as pd
import joblib

# Load model and features
model, feature_names = joblib.load('rainpredict_logistic_model.pkl')

st.set_page_config(
    page_title="Australia Rain Prediction 🌦️",
    page_icon="🌧️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Title and subtitle
st.title("🌧️ Australia Rain Prediction")
st.markdown(
    """
    Predict whether it will rain tomorrow based on today's weather data.
    Use the inputs below to enter current weather observations.
    """
)

# Group inputs in collapsible sections for cleaner UI
with st.expander("Location & Season Details", expanded=True):
    location = st.selectbox("Select Location", [
        "Adelaide", "Albany", "Albury", "AliceSprings", "BadgerysCreek", "Ballarat", "Bendigo",
        "Brisbane", "Cairns", "Canberra", "Cobar", "CoffsHarbour", "Dartmoor", "Darwin",
        "GoldCoast", "Hobart", "Katherine", "Launceston", "Melbourne", "MelbourneAirport",
        "Mildura", "Moree", "MountGambier", "MountGinini", "Newcastle", "Nhil", "NorahHead",
        "NorfolkIsland", "Nuriootpa", "PearceRAAF", "Penrith", "Perth", "PerthAirport", "Portland",
        "Richmond", "Sale", "SalmonGums", "Sydney", "SydneyAirport", "Townsville", "Tuggeranong",
        "Uluru", "WaggaWagga", "Walpole", "Watsonia", "Williamtown", "Witchcliffe", "Wollongong",
        "Woomera"
    ])

    season = st.selectbox("Select Season", ["Autumn", "Spring", "Summer", "Winter"])

with st.expander("Temperature & Rainfall", expanded=True):
    min_temp = st.number_input("Min Temperature (°C)", -10.0, 50.0, 15.0, help="Lowest temperature recorded today")
    max_temp = st.number_input("Max Temperature (°C)", -10.0, 50.0, 25.0, help="Highest temperature recorded today")
    rainfall = st.number_input("Rainfall (mm)", 0.0, 200.0, 2.0, help="Amount of rain fallen today")

with st.expander("Sun & Evaporation", expanded=False):
    evaporation = st.number_input("Evaporation (mm)", 0.0, 100.0, 5.0, help="Evaporation measurement today")
    sunshine = st.number_input("Sunshine (hours)", 0.0, 15.0, 8.0, help="Sunshine duration in hours")

with st.expander("Humidity & Pressure", expanded=False):
    humidity_9am = st.slider("Humidity at 9am (%)", 0, 100, 60)
    humidity_3pm = st.slider("Humidity at 3pm (%)", 0, 100, 50)
    pressure_9am = st.number_input("Pressure at 9am (hPa)", 980.0, 1050.0, 1010.0)
    pressure_3pm = st.number_input("Pressure at 3pm (hPa)", 980.0, 1050.0, 1008.0)

with st.expander("Cloud Cover", expanded=False):
    cloud_9am = st.slider(
        "Cloud Cover at 9am (0–8 oktas)", 0, 8, 4,
        help="Cloud cover measured in oktas: 0 = clear sky, 8 = overcast"
    )
    cloud_3pm = st.slider(
        "Cloud Cover at 3pm (0–8 oktas)", 0, 8, 5,
        help="Cloud cover measured in oktas: 0 = clear sky, 8 = overcast"
    )

with st.expander("Temperature at Times", expanded=False):
    temp_9am = st.number_input("Temperature at 9am (°C)", -10.0, 50.0, 18.0)
    temp_3pm = st.number_input("Temperature at 3pm (°C)", -10.0, 50.0, 23.0)

with st.expander("Wind Information", expanded=False):
    wind_gust_dir = st.selectbox("Wind Gust Direction", [
        'E','ENE','ESE','N','NE','NNE','NNW','NW','S','SE','SSE','SSW','SW','W','WNW','WSW'
    ])
    wind_dir_9am = st.selectbox("Wind Direction at 9am", [
        'E','ENE','ESE','N','NE','NNE','NNW','NW','S','SE','SSE','SSW','SW','W','WNW','WSW'
    ])
    wind_dir_3pm = st.selectbox("Wind Direction at 3pm", [
        'E','ENE','ESE','N','NE','NNE','NNW','NW','S','SE','SSE','SSW','SW','W','WNW','WSW'
    ])
    wind_gust_speed = st.slider("Wind Gust Speed (km/h)", 0, 150, 35)
    wind_speed_9am = st.slider("Wind Speed at 9am (km/h)", 0, 100, 20)
    wind_speed_3pm = st.slider("Wind Speed at 3pm (km/h)", 0, 100, 25)

# Build input DataFrame
input_data = {
    'MinTemp': min_temp,
    'MaxTemp': max_temp,
    'Rainfall': rainfall,
    'Evaporation': evaporation,
    'Sunshine': sunshine,
    'WindGustSpeed': wind_gust_speed,
    'WindSpeed9am': wind_speed_9am,
    'WindSpeed3pm': wind_speed_3pm,
    'Humidity9am': humidity_9am,
    'Humidity3pm': humidity_3pm,
    'Pressure9am': pressure_9am,
    'Pressure3pm': pressure_3pm,
    'Cloud9am': cloud_9am,
    'Cloud3pm': cloud_3pm,
    'Temp9am': temp_9am,
    'Temp3pm': temp_3pm
}

df_input = pd.DataFrame([input_data])

# Add categorical one-hot columns
for col in feature_names:
    if col.startswith("Location_"):
        df_input[col] = 1 if col == f"Location_{location}" else 0
    elif col.startswith("Season_"):
        df_input[col] = 1 if col == f"Season_{season}" else 0
    elif col.startswith("WindGustDir_"):
        df_input[col] = 1 if col == f"WindGustDir_{wind_gust_dir}" else 0
    elif col.startswith("WindDir9am_"):
        df_input[col] = 1 if col == f"WindDir9am_{wind_dir_9am}" else 0
    elif col.startswith("WindDir3pm_"):
        df_input[col] = 1 if col == f"WindDir3pm_{wind_dir_3pm}" else 0
    else:
        if col not in df_input.columns:
            df_input[col] = 0

df_input = df_input[feature_names]

if st.button("Predict Rain Tomorrow"):
    prediction = model.predict(df_input)[0]
    prediction_label = "🌧️ Yes" if prediction == 1 else "☀️ No"
    st.success(f"Will it rain tomorrow? {prediction_label}")

# Footer with a light background color
st.markdown(
    """
    <style>
    footer {
        visibility: visible;
        text-align: center;
        padding: 1rem;
        font-size: 0.9rem;
        color: #555;
        background-color: #f0f8ff;
        margin-top: 2rem;
        border-top: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True
)
