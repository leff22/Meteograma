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

def fetch_gfs(lat: float, lon: float, days: int = 7, tz: str = "America/Sao_Paulo") -> dict:
    url = (
        "https://api.open-meteo.com/v1/gfs"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={GFS_VARS}"
        f"&timezone={tz}"
        f"&forecast_days={days}"
    )
    with httpx.Client(timeout=30) as cx:
        r = cx.get(url)
        r.raise_for_status()
        return r.json()

def to_dataframe(d: dict) -> pd.DataFrame:
    h = d["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
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
    # Simulação: ondas e maré
    n = len(df)
    t_hours = (df.index.view('int64') // 3_600_000_000_000) % 24
    tide_amp = 1.0
    tide_base = 0.5
    tide = tide_base + tide_amp * np.sin(2 * np.pi * (t_hours) / 12.42)
    wave = 0.6 + 0.06 * (df["windspeed_10m"].fillna(0)) + 0.3 * np.sin(2 * np.pi * (t_hours) / 8.0)
    wave = np.clip(wave, 0.2, None)
    wdir = (df["winddirection_10m"].fillna(0) + 20) % 360
    df["wave_height"] = wave.round(2)
    df["wave_direction"] = wdir.round(0)
    df["tide_height"] = tide.round(2)
    return df

def save_json(d: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, ensure_ascii=False))

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)

def main():
    lat = -22.00
    lon = -47.89
    data_dir = Path(__file__).parent / "data"
    raw_path = data_dir / "gfs_sao_carlos_raw.json"
    csv_path = data_dir / "gfs_sao_carlos_hourly.csv"
    #raw_path = data_dir / "gfs_sao_sebastiao_raw.json"
    #csv_path = data_dir / "gfs_sao_sebastiao_hourly.csv"

    d = fetch_gfs(lat, lon, days=7)
    df = to_dataframe(d)
    save_json(d, raw_path)
    save_csv(df, csv_path)
    print({"rows": len(df), "start": str(df.index.min()), "end": str(df.index.max())})
    print({"json": str(raw_path), "csv": str(csv_path)})
    print({"saved_json": raw_path.exists(), "saved_csv": csv_path.exists()})

if __name__ == "__main__":
    main()