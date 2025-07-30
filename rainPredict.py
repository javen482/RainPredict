import streamlit as st
import pandas as pd
import pydeck as pdk
import joblib

model, feature_names = joblib.load('rainpredict_logistic_model.pkl')

city_coords = {
    "Adelaide": [-34.9285, 138.6007],
    "Albany": [-35.0266, 117.8836],
    "Albury": [-36.0737, 146.9135],
    "AliceSprings": [-23.6980, 133.8807],
    "BadgerysCreek": [-33.9118, 150.7417],
    "Ballarat": [-37.5622, 143.8503],
    "Bendigo": [-36.7570, 144.2794],
    "Brisbane": [-27.4698, 153.0251],
    "Cairns": [-16.9186, 145.7781],
    "Canberra": [-35.2809, 149.1300],
    "Cobar": [-31.4859, 145.8388],
    "CoffsHarbour": [-30.2963, 153.1135],
    "Dartmoor": [-37.9946, 141.1175],
    "Darwin": [-12.4634, 130.8456],
    "GoldCoast": [-28.0167, 153.4000],
    "Hobart": [-42.8821, 147.3272],
    "Katherine": [-14.4659, 132.2635],
    "Launceston": [-41.4332, 147.1441],
    "Melbourne": [-37.8136, 144.9631],
    "MelbourneAirport": [-37.6733, 144.8433],
    "Mildura": [-34.2088, 142.1311],
    "Moree": [-29.4615, 149.8403],
    "MountGambier": [-37.8316, 140.7790],
    "MountGinini": [-35.5739, 148.9657],
    "Newcastle": [-32.9283, 151.7817],
    "Nhil": [-36.3526, 141.6017],
    "NorahHead": [-33.2829, 151.5686],
    "NorfolkIsland": [-29.0408, 167.9483],
    "Nuriootpa": [-34.5265, 139.0089],
    "PearceRAAF": [-35.3333, 117.6333],
    "Penrith": [-33.7511, 150.6949],
    "Perth": [-31.9505, 115.8605],
    "PerthAirport": [-31.9406, 115.9675],
    "Portland": [-38.3446, 141.6056],
    "Richmond": [-33.5914, 150.7652],
    "Sale": [-38.1045, 147.0666],
    "SalmonGums": [-32.0461, 121.8116],
    "Sydney": [-33.8688, 151.2093],
    "SydneyAirport": [-33.9333, 151.1750],
    "Townsville": [-19.2589, 146.8169],
    "Tuggeranong": [-35.4277, 149.0794],
    "Uluru": [-25.3450, 131.0361],
    "WaggaWagga": [-35.1084, 147.3598],
    "Walpole": [-34.9497, 116.7339],
    "Watsonia": [-37.7073, 145.1060],
    "Williamtown": [-32.7923, 151.8333],
    "Witchcliffe": [-34.1889, 115.0992],
    "Wollongong": [-34.4278, 150.8931],
    "Woomera": [-31.1500, 136.8167]
}

st.set_page_config(
    page_title="Australia Rain Prediction 🌦️",
    page_icon="🌧️",
    layout="wide",  
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    body, .stApp {
    background-image:
        linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)),
        url("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1470&q=80");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
    background-color: #0e1117; 
    color: #eee;
    }
    .stButton>button {
        background-color: #0a84ff;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        padding: 0.6em 1.2em;
        font-size: 1rem;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #006ddb;
    }
    .css-1aumxhk {
        background-color: #121417 !important;
    }
    footer {
        visibility: visible; 
        text-align: center; 
        color: #999; 
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌧️ Australia Rain Prediction")
st.markdown(
    """
    Predict whether it will rain tomorrow based on today's weather data.  
    Use the inputs below to enter current weather observations.
    """
)

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Location & Map")
    location = st.selectbox("Select Location", list(city_coords.keys()))
    season = st.selectbox("Select Season", ["Autumn", "Spring", "Summer", "Winter"])
    lat, lon = city_coords[location]

    st.markdown("### 📍 Selected City Location")
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=6,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame([{"lat": lat, "lon": lon}]),
                get_position='[lon, lat]',
                get_color='[255, 50, 50, 200]',
                get_radius=60000,
                pickable=True
            )
        ],
        map_style="dark",
    ))

with col2:
    st.header("Weather Inputs")

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

st.markdown(
    """
    <footer>
    Made with 💧 for Australian weather insights
    </footer>
    """,
    unsafe_allow_html=True,
)

