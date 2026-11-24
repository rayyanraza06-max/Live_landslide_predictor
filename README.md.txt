#Global Live Landslide Risk Predictor(India and Japan)
    

	It is an AI powered disaster management tool that predicts real time landslide risk across India and Japan. It relies on satellite data and weather metrics completely.
 

#project Overview

	Landslides are devastating disasters...traditional models are often regional and historical. This project uses MACHINE LEARNING combined with LIVE SATELLITE DATA to predict land slide risk fir the day. (Map attached is of the specific day the code was run. Specifically on 24TH NOVEMBER 2026)

#Working

--->Training Data
    - Hazards (Label 1): Real historical landslides from the NASA Landslide Catalog.
   - Safe (Label 0): Pseudo-absence points, filtered to remove steep slopes (>10°) to prevent label noise.
--->Features Extracted via Google Earth Engine:**
   - **Terrain:** Elevation & Slope (Copernicus DEM 30m)
   - **Vegetation:** NDVI (MODIS)
   - **Historical Climate:** 5-year Average Rainfall (CHIRPS)
--->The AI Model:
   - A RANDOM FOREST CLASSIFIER (Scikit-Learn) trained with `class_weight='balanced'` to handle the imbalanced dataset.
   - **Accuracy:** ~88% (with a 97% recall rate for hazards, ensuring few disasters are missed).
---->The Live System:
   - The system queries NASA's **GPM (Global Precipitation Measurement)** satellite for the actual rainfall over the last 7 days.
   - It merges this live weather with the static terrain data and runs it through the Random Forest model.
   - It outputs a real-time interactive map (Folium) showing Safe (Green) and Hazard (Red) zones.

NOTE: I TRIED TO ALSO USE SOIL DATA WIHT THE STATIC TERRAIN DATA, BUT IN THE END I HAD A LESSER ACCURACY AND RECALL, THUS I HAD TO DROP THE SOIL FEATURE..(OVERFITTING EXAMPLE)

## Repository Contents
- `model_training.ipynb`: The Jupyter Notebook containing data extraction, feature engineering, and model training.
- `landslide_model.pkl`: The trained Random Forest model.
- `live_risk_map.html`: The interactive live map generated for today.
- `app.py`: The Streamlit web application code.

## Tech Stack
- **Python** (Pandas, Scikit-Learn, Folium)
- **Google Earth Engine** (GEE Python API)
- **Satellite Data:** Copernicus, MODIS, CHIRPS, GPM (NASA)


