import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib
import json
from datetime import datetime

# Page Setup
st.set_page_config(page_title="Live Landslide Risk AI", layout="wide", page_icon="🌍")
st.title("Live Landslide Risk Predictor")
st.markdown("Regions: India & Japan | Powered by: NASA GPM Live Rainfall & Random Forest AI")

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
@st.cache_data
def load_data():
    df = pd.read_csv('Data/live_grid_for_python_.csv')
    
    def extract_coords(geo_data):
        if isinstance(geo_data, str):
            # Case 1: JSON format
            if geo_data.startswith('{'):
                try:
                    data = json.loads(geo_data)
                    return pd.Series([data['coordinates'][1], data['coordinates'][0]])
                except:
                    pass
            # Case 2: WKT format "POINT (lon lat)"
            if 'POINT' in geo_data:
                try:
                    coords = geo_data.replace('POINT (', '').replace(')', '').split(' ')
                    return pd.Series([float(coords[1]), float(coords[0])])
                except:
                    pass
        return pd.Series([None, None])

    df[['latitude', 'longitude']] = df['.geo'].apply(extract_coords)
    df = df.dropna(subset=['latitude', 'longitude'])
    return df

# Process Data
with st.spinner("Loading satellite data and running AI predictions..."):
    df_live = load_data()

    # change Column Names
    if 'slope' in df_live.columns:
        df_live = df_live.rename(columns={'slope': 'slope_y'})

    # Predict
    features = ['DEM', 'slope_y', 'rainfall', 'NDVI']
    X_live = df_live[features].dropna()
    df_live = df_live.loc[X_live.index]
    df_live['Prediction'] = rf.predict(X_live)

# Dashboard Metrics
total_points = len(df_live)
hazards = len(df_live[df_live['Prediction'] == 1])
safe = len(df_live[df_live['Prediction'] == 0])

col1, col2, col3 = st.columns(3)
col1.metric("Total Monitored Points", total_points)
col2.metric("High Risk (Hazards)", hazards, delta_color="inverse")
col3.metric("Safe Zones", safe)

st.divider()

# MAP SECTION WITH TOGGLE 
st.subheader("Interactive Risk Map")

# Create the toggle
map_type = st.radio(
    "Choose Map Style:",
    ("Interactive Dots (Hover for details)", " Heatmap (Fast overview of hazard density)"),
    horizontal=True
)

# Create the  map
m = folium.Map(location=[25.0, 95.0], zoom_start=4, tiles='Esri.WorldImagery')

if map_type == "Interactive Dots (Hover for details)":
    st.markdown("*Hover over the dots to see the exact Slope and Rainfall.*")
    
    # Optimized loop for dots
    for row in df_live.to_dict('records'):
        color = 'red' if row['Prediction'] == 1 else 'green'
        tooltip_text = (
            f"<b>{'HAZARD' if row['Prediction'] == 1 else 'SAFE'}</b><br>"
            f"Slope: {row['slope_y']:.2f}°<br>"
            f"Rainfall: {row['rainfall']:.2f} mm"
        )
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=2.5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.5,
            weight=0.5,
            tooltip=folium.Tooltip(tooltip_text)
        ).add_to(m)

elif map_type == " Heatmap (Fast overview of hazard density)":
    st.markdown("*This heatmap only shows the areas predicted as **High Risk (Hazards)**.*")
    
    # Filter only hazards for the heatmap
    hazard_points = df_live[df_live['Prediction'] == 1]
    
    # Create list of [lat, lon] for the heatmap
    heat_data = hazard_points[['latitude', 'longitude']].values.tolist()
    
    # Add heatmap layer (control the size of the glow)
    HeatMap(heat_data, radius=12, blur=15, min_opacity=0.5).add_to(m)

#Display Map (wont rerun on click)
st_folium(m, width=1100, height=600, returned_objects=[])

# Sidebar
st.sidebar.header("About the AI")
st.sidebar.info("This model uses a Random Forest Classifier trained on NASA Landslide data, achieving 91% accuracy.")
st.sidebar.write("Features used: Elevation, Slope, Rainfall, NDVI.")
st.sidebar.write("Data Snapshot Date: 24 November 2026")
st.sidebar.warning("Note: This dashboard uses a cached data snapshot. To update the map wiht todays rainfall,the earth engine pipeline must be run manually.")
