from pathlib import Path
import json
import httpx
import pandas as pd
import numpy as np
import time

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
                print(f"Erro ao baixar GFS (tentativa {i+1}/{attempts}): {e}. Retentando em 2s...")
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
                print(f"Erro ao baixar Marine (tentativa {i+1}/{attempts}): {e}. Retentando em 2s...")
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

def save_json(d: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, ensure_ascii=False))

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)

def main():
    # Coordenadas Ubatuba
    lat = -23.43
    lon = -45.07
    data_dir = Path(__file__).parent / "data"
    raw_path = data_dir / "gfs_ubatuba_raw.json"
    csv_path = data_dir / "gfs_ubatuba_hourly.csv"

    print(f"Baixando dados para Ubatuba ({lat}, {lon})...")
    try:
        d = fetch_gfs(lat, lon, days=7)
        df = to_dataframe(d)

        # Fetch Marine data (add waves, etc.)
        try:
            d_marine = fetch_marine(lat, lon, days=7)
            df_marine = to_dataframe(d_marine)
            # Merge marine data into main dataframe
            # Use join to align on index (time)
            df = df.join(df_marine, rsuffix="_marine")
            print("Dados marinhos mesclados com sucesso.")
        except Exception as e:
            print(f"Warning: Could not fetch marine data: {e}")

        df = enrich_dataframe(df)
        
        save_json(d, raw_path)
        save_csv(df, csv_path)
        print({"rows": len(df), "start": str(df.index.min()), "end": str(df.index.max())})
        print({"json": str(raw_path), "csv": str(csv_path)})
        print({"saved_json": raw_path.exists(), "saved_csv": csv_path.exists()})
        print("Download concluído com sucesso!")
        
    except Exception as e:
        print(f"Erro fatal no download: {e}")

if __name__ == "__main__":
    main()
