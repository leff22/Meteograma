from pathlib import Path
import json
import httpx
import pandas as pd
import numpy as np

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

import time

def fetch_with_retry(url: str, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            with httpx.Client(timeout=60) as cx:
                r = cx.get(url)
                r.raise_for_status()
                return r.json()
        except httpx.RequestError as e:
            if i == retries - 1:
                raise
            print(f"Erro de conexão ({e}). Tentando novamente ({i+1}/{retries})...")
            time.sleep(2)
    return {}

def fetch_gfs(lat: float, lon: float, days: int = 7, tz: str = "America/Sao_Paulo") -> dict:
    url = (
        "https://api.open-meteo.com/v1/gfs"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={GFS_VARS}"
        f"&timezone={tz}"
        f"&forecast_days={days}"
    )
    return fetch_with_retry(url)

def fetch_marine(lat: float, lon: float, days: int = 7, tz: str = "America/Sao_Paulo") -> dict:
    url = (
        "https://marine-api.open-meteo.com/v1/marine"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={MARINE_VARS}"
        f"&timezone={tz}"
        f"&forecast_days={days}"
    )
    return fetch_with_retry(url)

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

def save_json(d: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, ensure_ascii=False))

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)

def main():
    # Coordenadas Carapicuíba, SP
    lat, lon = -23.5235, -46.8407
    print(f"Baixando dados para Carapicuíba: {lat}, {lon}")
    
    gfs = fetch_gfs(lat, lon)
    marine = fetch_marine(lat, lon)
    
    df_gfs = to_dataframe(gfs)
    df_marine = to_dataframe(marine)
    
    df = df_gfs.join(df_marine, how="outer")
    df = enrich_dataframe(df)
    
    base = Path(__file__).parent
    save_json(gfs, base / "data" / "gfs_carapicuiba.json")
    save_json(marine, base / "data" / "marine_carapicuiba.json")
    save_csv(df, base / "data" / "gfs_carapicuiba_hourly.csv")
    print("Concluído!")

if __name__ == "__main__":
    main()

