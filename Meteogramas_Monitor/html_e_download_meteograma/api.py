from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pandas as pd
import numpy as np
import httpx
import time
import json

import os

# Define the current directory where api.py and meteogram.html are located
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="/")
CORS(app)  # Enable CORS for all routes

@app.route("/")
def index():
    return app.send_static_file("meteogram.html")

GFS_VARS = ",".join([
    "temperature_2m",
    "relativehumidity_2m",
    "precipitation",
    "cloudcover",
    "pressure_msl",
    "windspeed_10m",
    "windgusts_10m",
    "winddirection_10m",
    "is_day",
    "cloud_cover_1000hPa",
    "cloud_cover_950hPa",
    "cloud_cover_925hPa",
    "cloud_cover_900hPa",
    "cloud_cover_850hPa",
    "cloud_cover_800hPa",
    "cloud_cover_700hPa",
    "cloud_cover_600hPa",
    "cloud_cover_500hPa",
    "cloud_cover_400hPa",
    "cloud_cover_300hPa",
    "cloud_cover_250hPa",
    "cloud_cover_200hPa",
    "cloud_cover_150hPa",
    "cloud_cover_100hPa"
])

MARINE_VARS = ",".join([
    "wave_height",
    "wave_direction",
    "wind_wave_height",
    "wind_wave_direction",
    "swell_wave_height",
    "swell_wave_direction",
])

def fetch_gfs(lat: float, lon: float, days: int = 7, tz: str = "America/Sao_Paulo") -> dict:
    url = (
        "https://api.open-meteo.com/v1/gfs"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={GFS_VARS}"
        f"&timezone={tz}"
        f"&forecast_days={days}"
    )
    
    attempts = 3
    for i in range(attempts):
        try:
            with httpx.Client(timeout=60) as cx:
                r = cx.get(url)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            if i < attempts - 1:
                print(f"Error downloading GFS (attempt {i+1}/{attempts}): {e}. Retrying in 2s...")
                time.sleep(2)
            else:
                raise e

def fetch_marine(lat: float, lon: float, days: int = 7, tz: str = "America/Sao_Paulo") -> dict:
    url = (
        "https://marine-api.open-meteo.com/v1/marine"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={MARINE_VARS}"
        f"&timezone={tz}"
        f"&forecast_days={days}"
    )
    
    attempts = 3
    for i in range(attempts):
        try:
            with httpx.Client(timeout=60) as cx:
                r = cx.get(url)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            if i < attempts - 1:
                print(f"Error downloading Marine data (attempt {i+1}/{attempts}): {e}. Retrying in 2s...")
                time.sleep(2)
            else:
                raise e

def to_dataframe(d: dict) -> pd.DataFrame:
    h = d["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    return df

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "relativehumidity_2m" in df.columns and "temperature_2m" in df.columns:
        T = df["temperature_2m"]
        RH = df["relativehumidity_2m"] / 100.0
        a = 17.27
        b = 237.7
        gamma = (a * T) / (b + T) + np.log(RH)
        df["dewpoint_2m"] = (b * gamma) / (a - gamma)
    
    if "windspeed_10m" in df.columns and "winddirection_10m" in df.columns:
        rad = np.deg2rad(df["winddirection_10m"])
        df["wind_u10"] = -df["windspeed_10m"] * np.sin(rad)
        df["wind_v10"] = -df["windspeed_10m"] * np.cos(rad)
    
    return df

@app.route("/meteogram", methods=["GET"])
def get_meteogram():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    days = request.args.get("days", 7, type=int)
    data_format = request.args.get("format", "csv")

    if lat is None or lon is None:
        return {"error": "Missing lat or lon parameters"}, 400

    try:
        data = fetch_gfs(lat, lon, days)
        df = to_dataframe(data)

        # Build lat lon dataframe
        n = len(data)
        df_latlon = df[["lat", "lon"]] # Keep lat lon columns for merging later
        df_latlon["lat"] = np.full(n, lat)
        df_latlon["lon"] = np.full(n, lon)
        
        # Fetch Marine data (add waves, etc.)
        try:
            d_marine = fetch_marine(lat, lon, days)
            df_marine = to_dataframe(d_marine)

            # Merge marine data into main dataframe
            # Use join to align on index (time)
            df = df.join(df_marine, rsuffix="_marine")

            # Merge lat lon data into main dataframe
            # Use join to align on index (time)
            df = df.join(df_latlon, rsuffix="_marine")

        except Exception as e:
            return {"error": str(e)}, 500

        df = enrich_dataframe(df)
        print(f"Erro :")

        caminho = os.path.join(os.path.dirname(__file__), "meteogram-data.csv")
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        df.to_csv(caminho, index=False)

        try:
            df.to_csv(caminho, index=False)
            print("Arquivo salvo com sucesso.")
        except Exception as e:
            print(f"Erro ao salvar o arquivo: {e}")

    except Exception as e:
        return {"error": str(e)}, 500

    if data_format == "json":
        response = make_response(df.to_json())
        response.headers["Content-Type"] = "application/json"
        return response, 200
    else:
        return df.to_csv(index=True), 200

get_meteogram ()
## if __name__ == "__main__":
##    app.run()