import numpy as np
import pandas as pd
import requests


GFS_VARS = ",".join(
    [
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
        "cloud_cover_100hPa",
    ]
)


MARINE_VARS = ",".join(
    [
        "wave_height",
        "wave_direction",
        "wind_wave_height",
        "wind_wave_direction",
        "swell_wave_height",
        "swell_wave_direction",
    ]
)


def fetch_gfs(
    lat: float, lon: float, days: int = 7, tz: str = "America/Sao_Paulo"
) -> dict:
    url = (
        "https://api.open-meteo.com/v1/gfs"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={GFS_VARS}"
        f"&timezone={tz}"
        f"&forecast_days={days}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_marine(
    lat: float, lon: float, days: int = 7, tz: str = "America/Sao_Paulo"
) -> dict:
    url = (
        "https://marine-api.open-meteo.com/v1/marine"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={MARINE_VARS}"
        f"&timezone={tz}"
        f"&forecast_days={days}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


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

def get_meteogram():
    lat = -20.550520
    lon = -40.633308
    days = 7
    data_format = "csv"
    caminho = "meteograma.csv"  # Define o caminho do arquivo CSV
    
    try:
        data = fetch_gfs(lat, lon, days)
        df = to_dataframe(data)        
                      
        # Fetch Marine data (add waves, etc.)
        try:
            d_marine = fetch_marine(lat, lon, days)
            df_marine = to_dataframe(d_marine)
            
            # Merge marine data into main dataframe
            df = df.join(df_marine, rsuffix="_marine")
            
        except Exception as e:
            print(f"Erro ao buscar dados marinhos: {e}")
             
        df = enrich_dataframe(df)

        # Build lat lon dataframe
        df["lat"] = f"{lat:.2f}"
        df["lon"] = f"{lon:.2f}"  
        
        try:
            df.to_csv(caminho, index=True)  # Salvar com índice (tempo)
            print(f"Arquivo salvo com sucesso em: {caminho}")
        except Exception as e:
            print(f"Erro ao salvar o arquivo: {e}")

    except Exception as e:
        print(f"Erro geral: {e}")
        return

    if data_format == "json":
        return df.to_json()
    else:
        return df.to_csv(index=True)

get_meteogram ()