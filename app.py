import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib

# 1. Page Setup
st.set_page_config(page_title="Live Landslide Risk AI", layout="wide")
st.title("Live Landslide Risk Predictor")
st.markdown("**Regions:** India & Japan | **Powered by:** NASA GPM Live Rainfall & Random Forest AI")

# Load the Model
@st.cache_resource
def load_model():
    return joblib.load('landslide_model.pkl')

try:
    rf = load_model()
except Exception as e:
    st.error(f"Model not found. Please ensure landslide_model.pkl is uploaded. Error: {e}")
    st.stop()

# Load the Data 
# For a live app, we would connect to Earth Engine here. 
#for now, we load a pre-generated live CSV)
@st.cache_data
def load_data():
    df = pd.read_csv('live_grid_for_python_.csv')
    
    # Extract lat/lon
    def extract_coords(geo_str):
        coords = geo_str.replace('POINT (', '').replace(')', '').split(' ')
        return pd.Series([float(coords[1]), float(coords[0])])
    
    df[['latitude', 'longitude']] = df['.geo'].apply(extract_coords)
    return df

df_live = load_data()

# Predict
features = ['DEM', 'slope_y', 'rainfall', 'NDVI']
if 'slope' in df_live.columns:
    df_live = df_live.rename(columns={'slope': 'slope_y'})

X_live = df_live[features].dropna()
df_live = df_live.loc[X_live.index]
df_live['Prediction'] = rf.predict(X_live)

# Create Map
st.subheader("Live Risk Map")
st.markdown("🔴 Red = High Risk | 🟢 Green = Safe. Hover over the dots for details.")

m = folium.Map(location=[25.0, 95.0], zoom_start=4, tiles='Esri.WorldImagery')

for _, row in df_live.iterrows():
    color = 'red' if row['Prediction'] == 1 else 'green'
    
    tooltip_text = (
        f"<b>{'HAZARD' if row['Prediction'] == 1 else 'SAFE'}</b><br>"
        f"Slope: {row['slope_y']:.2f}°<br>"
        f"Rainfall: {row['rainfall']:.2f} mm"
    )
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        tooltip=folium.Tooltip(tooltip_text)
    ).add_to(m)

# Display Map in Streamlit
st_folium(m, width=1000, height=600)

st.sidebar.header("About the AI")
st.sidebar.write("This model uses a Random Forest Classifier trained on NASA Landslide data.")
st.sidebar.write("Features used: Elevation, Slope, Rainfall, NDVI.")