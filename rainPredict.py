import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk

# --- Page config ---
st.set_page_config(page_title="Australia Rain Prediction", layout="centered", initial_sidebar_state="auto")

# --- Background & Theme ---
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1504384308090-c894fdcc538d?auto=format&fit=crop&w=2100&q=80");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}
.block-container {
    background-color: rgba(0, 0, 0, 0.75);
    padding: 2rem;
    border-radius: 10px;
}
h1, h2, h3, h4, h5, h6, p, label, .stSelectbox label, .stTextInput label {
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

st.title("🌧️ Australia Rain Prediction App")

# --- Load model ---
model = joblib.load("rain_model.pkl")

# --- Location dropdown ---
location_coords = {
    "Albury": (-36.0737, 146.9135),
    "AliceSprings": (-23.6980, 133.8807),
    "Brisbane": (-27.4698, 153.0251),
    "Cairns": (-16.9203, 145.7710),
    "Canberra": (-35.2809, 149.1300),
    "Cobar": (-31.4997, 145.8384),
    "CoffsHarbour": (-30.2963, 153.1135),
    "Darwin": (-12.4634, 130.8456),
    "Hobart": (-42.8821, 147.3272),
    "Melbourne": (-37.8136, 144.9631),
    "Mildura": (-34.2086, 142.1317),
    "Moree": (-29.4634, 149.8415),
    "MountGambier": (-37.8290, 140.7790),
    "NorahHead": (-33.2810, 151.5760),
    "NorfolkIsland": (-29.0408, 167.9547),
    "Penrith": (-33.7510, 150.6927),
    "Perth": (-31.9505, 115.8605),
    "Portland": (-38.3481, 141.6055),
    "Richmond": (-37.8183, 145.0017),
    "Sale": (-38.1035, 147.0667),
    "Sydney": (-33.8688, 151.2093),
    "Townsville": (-19.2589, 146.8169),
    "Tuggeranong": (-35.4099, 149.0672),
    "WaggaWagga": (-35.1180, 147.3598),
    "Watsonia": (-37.7160, 145.0827),
    "Williamtown": (-32.7980, 151.8340),
    "Wollongong": (-34.4278, 150.8931),
}

location = st.selectbox("Select Location", list(location_coords.keys()))
lat, lon = location_coords.get(location, (-33.8688, 151.2093))

# --- Show map ---
st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=6),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"lat": lat, "lon": lon}]),
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=40000,
        )
    ],
    map_style="mapbox://styles/mapbox/dark-v10"
))

# --- User Inputs ---
st.subheader("🌦️ Weather Conditions")

rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=1.0)
humidity9am = st.slider("Humidity at 9am (%)", 0, 100, 60)
humidity3pm = st.slider("Humidity at 3pm (%)", 0, 100, 50)
pressure9am = st.number_input("Pressure at 9am (hPa)", min_value=900.0, max_value=1050.0, value=1010.0)
pressure3pm = st.number_input("Pressure at 3pm (hPa)", min_value=900.0, max_value=1050.0, value=1008.0)
temp3pm = st.number_input("Temperature at 3pm (°C)", min_value=0.0, max_value=50.0, value=23.0)
windSpeed = st.slider("Wind Speed (km/h)", 0, 100, 15)

# --- Encode location as dummy variable ---
location_df = pd.get_dummies(pd.Series(location), prefix='Location')
all_possible = [f"Location_{loc}" for loc in location_coords.keys()]
location_encoded = pd.DataFrame(columns=all_possible)
for col in all_possible:
    location_encoded.at[0, col] = 1 if col == f"Location_{location}" else 0

# --- Create feature vector ---
features = pd.DataFrame({
    "Rainfall": [rainfall],
    "Humidity9am": [humidity9am],
    "Humidity3pm": [humidity3pm],
    "Pressure9am": [pressure9am],
    "Pressure3pm": [pressure3pm],
    "Temp3pm": [temp3pm],
    "WindSpeed": [windSpeed],
})
X_input = pd.concat([features, location_encoded], axis=1).fillna(0)

# --- Predict ---
if st.button("Predict Will It Rain Tomorrow?"):
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]
    if prediction == 1:
        st.success(f"🌧️ Yes, it is likely to rain tomorrow. (Confidence: {probability:.2%})")
    else:
        st.info(f"☀️ No, it is unlikely to rain tomorrow. (Confidence: {1 - probability:.2%})")
