import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import matplotlib.path as mpath
import matplotlib.markers as mmarkers
from pathlib import Path
import locale
from matplotlib.colors import LinearSegmentedColormap
import json
import httpx

try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except Exception:
    try:
        locale.setlocale(locale.LC_TIME, 'Portuguese_Brazil.1252')
    except Exception:
        pass

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
    print(f"Baixando dados de: {url}")
    with httpx.Client(timeout=30) as cx:
        r = cx.get(url)
        r.raise_for_status()
        return r.json()


def fetch_marine(lat: float, lon: float, days: int = 7, tz: str = "America/Sao_Paulo") -> dict:
    url = (
        "https://marine-api.open-meteo.com/v1/marine"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={MARINE_VARS}"
        f"&timezone={tz}"
        f"&forecast_days={days}"
    )
    print(f"Baixando dados marinhos de: {url}")
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
    return df


def save_json(d: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, ensure_ascii=False))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def load_tide_table() -> pd.DataFrame:
    tide_path = Path(__file__).parent / "data" / "tabua_mares_2026.csv"
    if not tide_path.exists():
        return pd.DataFrame(columns=["altura"])
    df = pd.read_csv(tide_path, sep=";", header=None, names=["data", "hora", "altura"], engine="python")
    df = df[df["altura"].notna()]
    df["altura"] = df["altura"].astype(str).str.replace(",", ".", regex=False)
    df["altura"] = df["altura"].astype(float)
    df["data_hora"] = pd.to_datetime(df["data"] + " " + df["hora"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["data_hora"])
    df = df.set_index("data_hora")[["altura"]]
    return df


def aggregate_12h(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        "temperature_2m": "mean",
        "dewpoint_2m": "mean",
        "precipitation": "sum",
        "cloudcover": "mean",
        "windspeed_10m": "max",
        "windgusts_10m": "max",
        "pressure_msl": "min",
        "is_day": "first",
        "winddirection_10m": "mean",
    }
    if "wave_height" in df.columns:
        agg_dict["wave_height"] = "mean"
    if "wave_direction" in df.columns:
        agg_dict["wave_direction"] = "first"
    for col in df.columns:
        if "cloud_cover_" in col:
            agg_dict[col] = "mean"
    return df.resample("12h").agg(agg_dict)


def get_wind_color(speed: float):
    if speed < 15:
        return "#9ca3af", 0.3
    if speed < 40:
        return "#22d3ee", 0.6
    if speed < 60:
        return "#06b6d4", 0.8
    if speed < 80:
        return "#f59e0b", 0.9
    return "#ef4444", 1.0


def draw_font_icon(ax, x_center, y_center, condition, is_day):
    FONT_NAME = "Segoe UI Emoji"
    FONT_SIZE = 12

    icon_char = ""
    icon_color = ""
    y_pos = y_center - 0.15

    if condition == "rain":
        icon_char = "\U0001F327"
        icon_color = "#3B82F6"
    elif condition == "cloudy":
        icon_char = "\u2601"
        icon_color = "#9CA3AF"
    elif condition == "partly":
        icon_char = "\u26C5"
        icon_color = "#9CA3AF"
        if not is_day:
            icon_char = "\u2601"
    else:
        if is_day:
            icon_char = "\u2600"
            icon_color = "#F59E0B"
        else:
            icon_char = "\U0001F319"
            icon_color = "#FCD34D"

    try:
        ax.text(
            x_center,
            y_pos,
            icon_char,
            fontname=FONT_NAME,
            fontsize=FONT_SIZE,
            color=icon_color,
            ha="center",
            va="center",
            zorder=10,
        )
    except Exception:
        ax.text(x_center, y_pos, "?", fontsize=20, ha="center", va="center")

    if condition == "rain":
        return "Chuva", "#2563EB"
    if condition == "cloudy":
        return "Nublado", "#6B7280"
    if condition == "partly":
        return "Parc.", "#6B7280"
    return "Limpo", ("#F59E0B" if is_day else "#4B5563")


def draw_meteogram(df_agg, df_hourly, output_path, city_name, lat, lon, is_litoral):
    plt.style.use("fast")
    fig = plt.figure(figsize=(11.69, 8.27), dpi=150, facecolor="#f7f7f7")

    C_BG = "#f7f7f7"
    C_GRID = "#e5e7eb"
    C_TEXT = "#111827"
    C_MUTED = "#6b7280"
    C_TEMP = "#ef4444"
    C_DEW = "#3b82f6"
    C_RAIN = "#1e3a8a"
    C_WAVE = "#0891b2"

    gs = gridspec.GridSpec(
        4,
        1,
        height_ratios=[1.4, 2, 2, 1.5],
        hspace=0.1,
        left=0.072,
        right=0.95,
        top=0.9,
        bottom=0.05,
    )

    ax_table = fig.add_subplot(gs[0], facecolor=C_BG)
    ax_temp = fig.add_subplot(gs[1], sharex=ax_table, facecolor=C_BG)
    ax_precip = fig.add_subplot(gs[2], sharex=ax_table, facecolor=C_BG)
    ax_wave = fig.add_subplot(gs[3], sharex=ax_table, facecolor=C_BG)

    times = df_agg.index

    ax_table.set_xlim(times[0], times[-1] + pd.Timedelta(hours=12))
    ax_table.set_ylim(0, 4.5)
    ax_table.axis("off")

    labels = [(0.6, "VENTO"), (1.5, "PRESSÃO"), (2.5, "CONDIÇÃO"), (3.5, "DATA")]
    for y, txt in labels:
        ax_table.text(
            times[0],
            y,
            txt + "  ",
            ha="right",
            va="center",
            fontsize=9,
            color=C_MUTED,
            fontweight="bold",
            fontfamily="sans-serif",
        )
    ax_table.text(
        times[0],
        0.22,
        "RAJADA  ",
        ha="right",
        va="center",
        fontsize=7,
        color=C_MUTED,
        fontweight="bold",
        fontfamily="sans-serif",
    )

    for i, t in enumerate(times):
        if i > 0:
            ax_table.axvline(t, color=C_GRID, linestyle="--", linewidth=1)
        x_center = t + pd.Timedelta(hours=6)
        hour_txt = t.strftime("%H")
        ax_table.text(
            x_center,
            3.4,
            hour_txt + "h",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=C_MUTED,
        )
        if i == 0 or t.day != times[i - 1].day:
            day_txt = t.strftime("%a %d").upper()
            has_next_block = (i + 1 < len(times)) and (times[i + 1].day == t.day)
            day_x = t + pd.Timedelta(hours=12) if has_next_block else t + pd.Timedelta(hours=6)
            ax_table.text(
                day_x,
                3.9,
                day_txt,
                ha="center",
                va="center",
                fontsize=9,
                color=C_TEXT,
                fontweight="bold",
            )
            if i > 0:
                ax_table.axvline(t, color="#d1d5db", linestyle="-", linewidth=1.5)

        is_day = df_agg["is_day"].iloc[i]
        precip = df_agg["precipitation"].iloc[i]
        cloud = df_agg["cloudcover"].iloc[i]

        cond_type = "clear"
        if precip >= 1.0:
            cond_type = "rain"
        elif cloud >= 80:
            cond_type = "cloudy"
        elif cloud >= 40:
            cond_type = "partly"

        label_txt, label_color = draw_font_icon(ax_table, x_center, 2.95, cond_type, is_day)
        ax_table.text(
            x_center,
            2.25,
            label_txt,
            ha="center",
            va="center",
            fontsize=7,
            color=label_color,
            fontweight="bold",
        )

        press = df_agg["pressure_msl"].iloc[i]
        ax_table.text(
            x_center,
            1.55,
            f"{press:.0f}",
            ha="center",
            va="center",
            fontsize=9,
            color="#8b5cf6",
            fontweight="bold",
        )
        ax_table.text(
            x_center,
            1.15,
            "hPa",
            ha="center",
            va="center",
            fontsize=7,
            color="#8b5cf6",
        )

        w_spd = df_agg["windspeed_10m"].iloc[i]
        w_dir = df_agg["winddirection_10m"].iloc[i]
        w_gust = df_agg["windgusts_10m"].iloc[i]

        arrow_path = mpath.Path([(0, 1.0), (-0.6, -0.8), (0, -0.4), (0.6, -0.8), (0, 1.0)])
        t_trans = matplotlib.transforms.Affine2D().rotate_deg(270 - w_dir)
        m = mmarkers.MarkerStyle(marker=arrow_path, transform=t_trans)

        ax_table.scatter(
            x_center - pd.Timedelta(hours=3.5),
            0.6,
            marker=m,
            s=120,
            color=C_MUTED,
        )
        ax_table.text(
            x_center + pd.Timedelta(hours=1.0),
            0.6,
            f"{w_spd:.0f}",
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=C_TEXT,
        )
        ax_table.text(
            x_center + pd.Timedelta(hours=1.2),
            0.6,
            "km/h",
            ha="left",
            va="center",
            fontsize=7,
            color=C_MUTED,
        )

        c_gust, alpha_gust = get_wind_color(w_gust)
        rect_w = 0.4
        rect_x = mdates.date2num(x_center) - (rect_w / 2)
        ax_table.add_patch(Rectangle((rect_x, 0.1), rect_w, 0.25, color=c_gust, alpha=alpha_gust))
        ax_table.text(
            x_center,
            0.22,
            f"{w_gust:.0f}",
            ha="center",
            va="center",
            fontsize=9,
            color=c_gust,
            fontweight="bold",
        )

    ax_temp.grid(True, linestyle="-", color=C_GRID, alpha=0.8)
    ax_temp.spines["top"].set_visible(False)
    ax_temp.spines["right"].set_visible(False)
    ax_temp.spines["left"].set_visible(False)
    ax_temp.spines["bottom"].set_color(C_GRID)

    ax_temp.fill_between(df_hourly.index, df_hourly["temperature_2m"], alpha=0.1, color=C_TEMP)
    ax_temp.plot(df_hourly.index, df_hourly["temperature_2m"], color=C_TEMP, linewidth=2.5, label="Temp")
    ax_temp.plot(
        df_hourly.index,
        df_hourly["dewpoint_2m"],
        color=C_DEW,
        linewidth=1.5,
        linestyle="--",
        label="Orvalho",
    )

    for _, group in df_hourly.groupby(df_hourly.index.date):
        t_max = group["temperature_2m"].max()
        idx_max = group["temperature_2m"].idxmax()
        ax_temp.text(
            idx_max,
            t_max + 0.5,
            f"{t_max:.1f}°",
            ha="center",
            va="bottom",
            fontsize=9,
            color=C_TEMP,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
        )
        t_min = group["temperature_2m"].min()
        idx_min = group["temperature_2m"].idxmin()
        ax_temp.text(
            idx_min,
            t_min - 0.8,
            f"{t_min:.1f}°",
            ha="center",
            va="top",
            fontsize=9,
            color=C_TEMP,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
        )

    ax_temp.set_ylabel("Temperatura (°C)", color=C_TEMP, fontweight="bold")
    ax_temp.tick_params(axis="y", colors=C_TEMP)

    ax_dew = ax_temp.twinx()
    t_min = min(df_hourly["temperature_2m"].min(), df_hourly["dewpoint_2m"].min())
    t_max = max(df_hourly["temperature_2m"].max(), df_hourly["dewpoint_2m"].max())
    padding = (t_max - t_min) * 0.1
    y_limits = (t_min - padding, t_max + padding)
    ax_temp.set_ylim(y_limits)
    ax_dew.set_ylim(y_limits)

    ax_dew.set_ylabel("Ponto de Orvalho (°C)", color=C_DEW, fontweight="bold")
    ax_dew.tick_params(axis="y", colors=C_DEW)
    ax_dew.spines["top"].set_visible(False)
    ax_dew.spines["left"].set_visible(False)
    ax_dew.spines["bottom"].set_visible(False)
    ax_dew.spines["right"].set_color(C_DEW)

    for i, t in enumerate(times):
        if df_agg["is_day"].iloc[i] == 0:
            ax_temp.axvspan(t, t + pd.Timedelta(hours=12), color="#111827", alpha=0.05)

    ax_precip.grid(False)
    ax_precip.spines["top"].set_visible(False)
    ax_precip.spines["right"].set_visible(False)
    ax_precip.spines["left"].set_visible(False)
    ax_precip.spines["bottom"].set_visible(False)

    levels = [1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100]
    alts = [44.33 * (1 - (p / 1013.25) ** 0.1903) for p in levels]

    cloud_data = []
    for p in levels:
        col = f"cloud_cover_{p}hPa"
        if col in df_hourly.columns:
            cloud_data.append(df_hourly[col].values)
        else:
            cloud_data.append(np.zeros(len(df_hourly)))

    Z = np.array(cloud_data)
    cmap_colors = ["#f7f7f7", "#d1d5db", "#9ca3af", "#4b5563", "#1f2937"]
    cmap = LinearSegmentedColormap.from_list("cloud_cmap", cmap_colors)

    time_nums = mdates.date2num(df_hourly.index)
    T_grid, A_grid = np.meshgrid(time_nums, alts)

    ax_precip.pcolormesh(T_grid, A_grid, Z, shading="gouraud", cmap=cmap, vmin=0, vmax=100, alpha=0.9)

    ax_rain = ax_precip.twinx()

    ax_precip.set_ylim(0, 16)
    ax_precip.set_ylabel("Altitude Nuvens (km)", color=C_MUTED)
    ax_precip.yaxis.set_label_position("right")
    ax_precip.yaxis.tick_right()
    ax_precip.spines["left"].set_visible(False)
    ax_precip.spines["right"].set_visible(True)

    ax_rain.set_ylabel("Chuva (mm)", color=C_RAIN, fontweight="bold")
    ax_rain.yaxis.set_label_position("left")
    ax_rain.yaxis.tick_left()
    ax_rain.spines["right"].set_visible(False)
    ax_rain.spines["left"].set_visible(True)
    ax_rain.spines["left"].set_color(C_RAIN)
    ax_rain.tick_params(axis="y", colors=C_RAIN)

    bar_width = 0.45
    ax_rain.bar(mdates.date2num(times) + 0.25, df_agg["precipitation"], width=bar_width, color=C_RAIN, alpha=0.3)

    for t, val in zip(times, df_agg["precipitation"]):
        if val > 0.1:
            ax_rain.text(
                t + pd.Timedelta(hours=6),
                val + 0.2,
                f"{val:.1f}mm",
                ha="center",
                va="bottom",
                fontsize=9,
                color=C_RAIN,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
            )

    ax_rain.set_ylim(0, max(15, df_agg["precipitation"].max() * 1.5))

    if is_litoral and "wave_height" in df_hourly.columns:
        ax_wave.grid(True, linestyle="-", color=C_GRID, alpha=0.8)
        ax_wave.spines["top"].set_visible(False)
        ax_wave.spines["right"].set_visible(False)
        ax_wave.spines["left"].set_visible(False)
        ax_wave.spines["bottom"].set_visible(False)

        ax_wave.fill_between(df_hourly.index, df_hourly["wave_height"], alpha=0.2, color=C_WAVE)
        ax_wave.plot(df_hourly.index, df_hourly["wave_height"], color=C_WAVE, linewidth=2)

        ax_wave.set_ylabel("Ondas (m)", color=C_WAVE, fontweight="bold")
        ax_wave.tick_params(axis="y", colors=C_WAVE)

        wave_sub = df_hourly.resample("3h").first()
        for t, row in wave_sub.iterrows():
            if pd.notnull(row["wave_height"]) and pd.notnull(row["wave_direction"]):
                w_dir = row["wave_direction"]
                marker_angle = -(w_dir + 180)
                wave_arrow_path = mpath.Path([(0, 1.0), (-0.5, -0.7), (0, -0.2), (0.5, -0.7), (0, 1.0)])
                t_trans = matplotlib.transforms.Affine2D().rotate_deg(marker_angle)
                m = mmarkers.MarkerStyle(marker=wave_arrow_path, transform=t_trans)
                ax_wave.scatter(t, row["wave_height"], marker=m, s=100, color="#0e7490", zorder=10)
    elif is_litoral:
        msg = "Dados de Onda não disponíveis"
        ax_wave.text(0.5, 0.5, msg, ha="center", va="center", transform=ax_wave.transAxes)
        ax_wave.set_ylabel("Ondas (m)", color=C_WAVE, fontweight="bold")
        ax_wave.tick_params(axis="y", colors=C_WAVE)
        ax_wave.spines["top"].set_visible(False)
        ax_wave.spines["right"].set_visible(False)
        ax_wave.spines["left"].set_visible(False)
        ax_wave.spines["bottom"].set_visible(False)
    else:
        ax_wave.grid(False)
        ax_wave.spines["top"].set_visible(False)
        ax_wave.spines["right"].set_visible(False)
        ax_wave.spines["left"].set_visible(False)
        ax_wave.spines["bottom"].set_visible(False)
        ax_wave.set_ylabel("Modelo hidrológico", color=C_MUTED, fontweight="bold")
        ax_wave.tick_params(axis="y", colors=C_MUTED)
        ax_wave.text(
            0.5,
            0.5,
            "Modelo hidrológico (a integrar)",
            ha="center",
            va="center",
            transform=ax_wave.transAxes,
            color=C_MUTED,
        )

    ax_precip.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax_table.get_xticklabels(), visible=False)
    plt.setp(ax_temp.get_xticklabels(), visible=False)
    plt.setp(ax_precip.get_xticklabels(), visible=False)
    plt.setp(ax_wave.get_xticklabels(), visible=False)

    source_label = "GFS/GFS-WAVE" if is_litoral else "GFS"
    fig.text(
        0.5,
        0.02,
        f"Gerado por Geopixel • Fonte: {source_label} • {city_name}",
        ha="center",
        fontsize=10,
        color=C_MUTED,
    )

    logo_path = Path(__file__).parent / "geo_logo1.png"
    if logo_path.exists():
        try:
            img = plt.imread(str(logo_path))
            ax_logo = fig.add_axes([0.85, 0.895, 0.14, 0.1], anchor="NE", zorder=12)
            ax_logo.imshow(img)
            ax_logo.axis("off")
        except Exception:
            pass

    now_str = pd.Timestamp.now().strftime("%d/%m %H:%M")
    lat_str = f"{abs(lat):.2f}°S" if lat < 0 else f"{lat:.2f}°N"
    lon_str = f"{abs(lon):.2f}°W" if lon < 0 else f"{lon:.2f}°E"
    info_txt = f"Lat: {lat_str}   Lon: {lon_str}\nRodada: {now_str} Local"
    fig.text(0.02, 0.96, info_txt, ha="left", va="top", fontsize=11, color=C_MUTED, linespacing=1.4)
    fig.text(0.5, 0.95, f"Meteograma — {city_name}", ha="center", va="center", fontsize=18, fontweight="bold", color=C_TEXT)

    plt.savefig(output_path, facecolor=C_BG)
    print(f"Sucesso! Arquivo salvo em: {output_path}")
    plt.close(fig)


def main():
    cities = [
        {
            "name": "Florianópolis, SC",
            "file_name": "florianopolis",
            "lat": -27.60,
            "lon": -48.55,
            "litoral": True,
        },
    ]

    root = Path(__file__).parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for city in cities:
        print("\n==================================================")
        print(f"Processando: {city['name']}")
        print("==================================================")

        lat = city["lat"]
        lon = city["lon"]
        file_name = city["file_name"]
        is_litoral = city["litoral"]

        raw_path = data_dir / f"gfs_{file_name}_raw.json"
        csv_path = data_dir / f"gfs_{file_name}_hourly.csv"
        output_path = root / f"meteograma_{file_name}.png"

        print("--- Etapa 1: Download de Dados GFS e Marine ---")
        df = None
        try:
            d = fetch_gfs(lat, lon, days=7)
            df = to_dataframe(d)
            if is_litoral:
                try:
                    d_marine = fetch_marine(lat, lon, days=7)
                    df_marine = to_dataframe(d_marine)
                    df = df.join(df_marine, rsuffix="_marine")
                    print("Dados marinhos mesclados com sucesso.")
                except Exception as em:
                    print(f"Erro no download marinho: {em}")
            else:
                print("Cidade do interior: pulando dados marinhos.")

            save_json(d, raw_path)
            save_csv(df, csv_path)
            print("Dados baixados e salvos com sucesso.")
        except Exception as e:
            print(f"Erro no download: {e}")
            if csv_path.exists():
                print("Tentando usar dados locais em cache...")
                df = pd.read_csv(csv_path)
                df["time"] = pd.to_datetime(df["time"])
                df = df.set_index("time")
            else:
                print("Não foi possível obter dados. Pulando esta cidade.")
                continue

        print("\n--- Etapa 2: Geração do Meteograma ---")
        try:
            if df is not None and not df.empty:
                df_12h = aggregate_12h(df)
                draw_meteogram(df_12h, df, output_path, city["name"], lat, lon, is_litoral)
            else:
                print("DataFrame vazio ou inválido.")
        except Exception as e:
            print(f"Erro na geração do gráfico: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
