import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend para gerar arquivos sem janela
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

# Configuração de Locale
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except Exception:
    try:
        locale.setlocale(locale.LC_TIME, 'Portuguese_Brazil.1252')
    except Exception:
        pass

# --- CONSTANTES DE DOWNLOAD (Do download_dados.py) ---
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

# --- FUNÇÕES DE DOWNLOAD ---
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
    tide_path = Path(__file__).parent / 'data' / 'tabua_mares_2026.csv'
    if not tide_path.exists():
        return pd.DataFrame(columns=['altura'])
    df = pd.read_csv(tide_path, sep=';', header=None, names=['data', 'hora', 'altura'], engine='python')
    df = df[df['altura'].notna()]
    df['altura'] = df['altura'].astype(str).str.replace(',', '.', regex=False)
    df['altura'] = df['altura'].astype(float)
    df['data_hora'] = pd.to_datetime(df['data'] + ' ' + df['hora'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['data_hora'])
    df = df.set_index('data_hora')[['altura']]
    return df

# --- FUNÇÕES DE PROCESSAMENTO E GRÁFICO (Do generate_meteogram.py) ---
def aggregate_6h(df):
    # Agrega dados horários para blocos de 6h
    agg_dict = {
        'temperature_2m': 'mean',
        'dewpoint_2m': 'mean',
        'precipitation': 'sum',
        'cloudcover': 'mean',
        'windspeed_10m': 'max',
        'windgusts_10m': 'max',
        'pressure_msl': 'min',
        'is_day': 'first',
        'winddirection_10m': 'mean'
    }
    
    # Adicionar agregação de ondas se as colunas existirem
    if 'wave_height' in df.columns:
        agg_dict['wave_height'] = 'mean'
    if 'wave_direction' in df.columns:
        agg_dict['wave_direction'] = 'first'

    # Inclui nuvens por nível se houver
    for col in df.columns:
        if 'cloud_cover_' in col:
            agg_dict[col] = 'mean'
            
    return df.resample('6h').agg(agg_dict)

def get_wind_color(speed):
    # Cores baseadas na velocidade (km/h) - Gradiente Suave
    # 0: Cinza (#9ca3af)
    # 10: Ciano claro (#67e8f9)
    # 20: Ciano (#22d3ee)
    # 40: Amarelo (#facc15)
    # 60: Laranja (#fb923c)
    # 80: Vermelho (#ef4444)
    # 100: Vermelho (#ef4444)
    
    # Normalizar velocidade entre 0 e 100 km/h (satura em 100)
    s = max(0, min(speed, 100))
    norm = s / 100.0
    
    # Definir gradiente
    colors = [
        (0.0, '#9ca3af'), 
        (0.1, '#67e8f9'), 
        (0.2, '#22d3ee'),
        (0.3, '#facc15'),
        (0.5, '#fb923c'),
        (0.8, '#ef4444'),
        (1.0, '#ef4444')
    ]
    cmap = LinearSegmentedColormap.from_list("wind_gust", colors)
    
    # Obter cor RGBA interpolada
    rgba = cmap(norm)
    color_hex = matplotlib.colors.to_hex(rgba)
    
    # Alpha progressivo para dar destaque visual
    alpha = 0.3 + (0.6 * norm)
    
    return color_hex, alpha

def draw_font_icon(ax, x_center, y_center, condition, is_day):
    # Usar fonte do sistema com suporte a emojis/símbolos (Windows)
    FONT_NAME = 'Segoe UI Emoji' 
    FONT_SIZE = 12 # Tamanho grande para o ícone
    
    icon_char = ''
    icon_color = ''
    
    # Ajuste fino vertical (deslocado para baixo conforme solicitado)
    y_pos = y_center - 0.15

    if condition == 'rain':
        # Nuvem com chuva (🌧️)
        icon_char = '\U0001F327' 
        icon_color = '#3B82F6' # Azul
    
    elif condition == 'cloudy':
        # Nuvem (☁)
        icon_char = '\u2601' 
        icon_color = '#9CA3AF' # Cinza
        
    elif condition == 'partly':
        # Sol com nuvem (⛅)
        icon_char = '\u26C5'
        icon_color = '#9CA3AF' # Cinza
        if not is_day:
            # Não existe "Lua atrás da nuvem" universal. Vamos usar nuvem simples.
            icon_char = '\u2601'
    
    else: # Clear
        if is_day:
            # Sol (☀)
            icon_char = '\u2600'
            icon_color = '#F59E0B' # Laranja
        else:
            # Lua (🌙)
            icon_char = '\U0001F319' # Crescent Moon
            icon_color = '#FCD34D' # Amarelo Lua

    try:
        ax.text(x_center, y_pos, icon_char, fontname=FONT_NAME, fontsize=FONT_SIZE, 
                color=icon_color, ha='center', va='center', zorder=10)
    except Exception:
        ax.text(x_center, y_pos, "?", fontsize=20, ha='center', va='center')

    # Retornar rótulos de texto
    if condition == 'rain':
        return "Chuva", '#2563EB'
    elif condition == 'cloudy':
        return "Nublado", '#6B7280'
    elif condition == 'partly':
        return "Parc.", '#6B7280'
    else:
        return "Limpo", ('#F59E0B' if is_day else '#FCD34D')

def draw_meteogram(df_agg, df_hourly, output_path, city_name, lat, lon, is_litoral):
    # Configuração A4 Paisagem Estilizada
    plt.style.use('fast') # Base clean
    fig = plt.figure(figsize=(11.69, 8.27), dpi=150, facecolor='#f7f7f7') # Cor de fundo HTML
    
    # Cores do HTML
    C_BG = '#f7f7f7'
    C_GRID = '#e5e7eb'
    C_TEXT = '#111827'
    C_MUTED = '#6b7280'
    C_TEMP = '#ef4444'
    C_DEW = '#3b82f6'
    C_RAIN = '#1e3a8a' # Azul mais escuro (blue-900)
    C_WAVE = '#0891b2' # Ciano escuro
    
    # Layout: Topo (Tabela), Meio (Temp), Baixo (Chuva)
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.4, 3, 3], hspace=0.1, 
                           left=0.072, right=0.95, top=0.90, bottom=0.05)
    
    ax_table = fig.add_subplot(gs[0], facecolor=C_BG)
    ax_temp = fig.add_subplot(gs[1], sharex=ax_table, facecolor=C_BG)
    ax_precip = fig.add_subplot(gs[2], sharex=ax_table, facecolor=C_BG)
    
    times = df_agg.index
    
    # --- 1. TABELA DE DADOS ---
    # Ajustado para 6h (intervalo menor, então +6h para margem)
    ax_table.set_xlim(times[0], times[-1] + pd.Timedelta(hours=6))
    ax_table.set_ylim(0, 4.5)
    ax_table.axis('off')
    
    # Rótulos Laterais
    labels = [
        (0.6, "VENTO"), (1.5, "PRESSÃO"), 
        (2.5, "CONDIÇÃO"), (3.5, "DATA")
    ]
    for y, txt in labels:
        ax_table.text(times[0], y, txt + "  ", ha='right', va='center', fontsize=9, color=C_MUTED, fontweight='bold', fontfamily='sans-serif')
        
    ax_table.text(times[0], 0.22, "RAJADA  ", ha='right', va='center', fontsize=7, color=C_MUTED, fontweight='bold', fontfamily='sans-serif')

    for i, t in enumerate(times):
        if i > 0:
            ax_table.axvline(t, color=C_GRID, linestyle='--', linewidth=1)
            
        # Centro do bloco de 6h = t + 3h
        x_center = t + pd.Timedelta(hours=3)
        
        # LINHA 3: Hora e Dia
        hour_txt = t.strftime('%H')
        ax_table.text(x_center, 3.4, hour_txt + "h", ha='center', va='center', fontsize=8, fontweight='bold', color=C_MUTED)
        
        # Lógica de Rótulo do Dia (Centralizado nos blocos do dia)
        if i == 0 or t.day != times[i-1].day:
            day_txt = t.strftime('%a %d').upper()
            
            # Encontrar todos os blocos deste dia
            current_day = t.day
            day_blocks = [time for time in times if time.day == current_day]
            
            if day_blocks:
                start_t = day_blocks[0]
                end_t = day_blocks[-1] + pd.Timedelta(hours=6)
                day_x = start_t + (end_t - start_t) / 2
                
                ax_table.text(day_x, 3.9, day_txt, ha='center', va='center', fontsize=9, color=C_TEXT, fontweight='bold')
            
            if i > 0:
                ax_table.axvline(t, color='#d1d5db', linestyle='-', linewidth=1.5)

        # LINHA 2: Ícone do Tempo
        is_day = df_agg['is_day'].iloc[i]
        
        # Ajuste: Se for 18h, considerar como noite (para exibir lua se estiver limpo)
        if t.hour == 18:
            is_day = 0

        precip = df_agg['precipitation'].iloc[i]
        cloud = df_agg['cloudcover'].iloc[i]
        
        cond_type = 'clear'
        if precip >= 1.0: cond_type = 'rain'
        elif cloud >= 80: cond_type = 'cloudy'
        elif cloud >= 40: cond_type = 'partly'
        
        label_txt, label_color = draw_font_icon(ax_table, x_center, 2.95, cond_type, is_day)
        ax_table.text(x_center, 2.25, label_txt, ha='center', va='center', fontsize=7, color=label_color, fontweight='bold')

        # LINHA 1: Pressão
        press = df_agg['pressure_msl'].iloc[i]
        ax_table.text(x_center, 1.55, f"{press:.0f}", ha='center', va='center', fontsize=9, color='#8b5cf6', fontweight='bold')
        ax_table.text(x_center, 1.15, "hPa", ha='center', va='center', fontsize=7, color='#8b5cf6')
        
        # LINHA 0: Vento
        w_spd = df_agg['windspeed_10m'].iloc[i]
        w_dir = df_agg['winddirection_10m'].iloc[i]
        w_gust = df_agg['windgusts_10m'].iloc[i]
        
        arrow_path = mpath.Path([
            (0, 1.0), (-0.6, -0.8), (0, -0.4), (0.6, -0.8), (0, 1.0)
        ])
        t_trans = matplotlib.transforms.Affine2D().rotate_deg(270 - w_dir)
        m = mmarkers.MarkerStyle(marker=arrow_path, transform=t_trans)
        
        # Ajuste de posição para caber em 6h (mais estreito)
        ax_table.scatter(x_center - pd.Timedelta(hours=1.5), 0.6, marker=m, s=100, color=C_MUTED)
        ax_table.text(x_center + pd.Timedelta(hours=0.5), 0.6, f"{w_spd:.0f}", ha='left', va='center', fontsize=10, fontweight='bold', color=C_TEXT)
        # ax_table.text(x_center + pd.Timedelta(hours=1.2), 0.6, "km/h", ha='left', va='center', fontsize=7, color=C_MUTED) # Removido para economizar espaço ou ajustado
        
        # Rajada
        c_gust, alpha_gust = get_wind_color(w_gust)
        rect_w = 0.20 # Mais estreito para 6h (era 0.40 para 12h)
        rect_x = mdates.date2num(x_center) - (rect_w / 2)
        ax_table.add_patch(Rectangle((rect_x, 0.1), rect_w, 0.25, color=c_gust, alpha=alpha_gust))
        ax_table.text(x_center, 0.22, f"{w_gust:.0f}", ha='center', va='center', fontsize=8, color=c_gust, fontweight='bold')

    # --- 2. GRÁFICO TEMPERATURA ---
    ax_temp.grid(True, linestyle='-', color=C_GRID, alpha=0.8)
    ax_temp.spines['top'].set_visible(False)
    ax_temp.spines['right'].set_visible(False)
    ax_temp.spines['left'].set_visible(False)
    ax_temp.spines['bottom'].set_color(C_GRID)
    
    ax_temp.fill_between(df_hourly.index, df_hourly['temperature_2m'], alpha=0.1, color=C_TEMP)
    ax_temp.plot(df_hourly.index, df_hourly['temperature_2m'], color=C_TEMP, linewidth=2.5, label='Temp')
    ax_temp.plot(df_hourly.index, df_hourly['dewpoint_2m'], color=C_DEW, linewidth=1.5, linestyle='--', label='Orvalho')
    
    # Extremos diários
    for _, group in df_hourly.groupby(df_hourly.index.date):
        t_max = group['temperature_2m'].max()
        idx_max = group['temperature_2m'].idxmax()
        ax_temp.text(idx_max, t_max + 0.5, f"{t_max:.1f}°", ha='center', va='bottom', fontsize=9, color=C_TEMP, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        
        t_min = group['temperature_2m'].min()
        idx_min = group['temperature_2m'].idxmin()
        ax_temp.text(idx_min, t_min - 0.8, f"{t_min:.1f}°", ha='center', va='top', fontsize=9, color=C_TEMP, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        
    ax_temp.set_ylabel('Temperatura (°C)', color=C_TEMP, fontweight='bold')
    ax_temp.tick_params(axis='y', colors=C_TEMP)

    # Eixo Orvalho
    ax_dew = ax_temp.twinx()
    t_min = min(df_hourly['temperature_2m'].min(), df_hourly['dewpoint_2m'].min())
    t_max = max(df_hourly['temperature_2m'].max(), df_hourly['dewpoint_2m'].max())
    padding = (t_max - t_min) * 0.1
    y_limits = (t_min - padding, t_max + padding)
    ax_temp.set_ylim(y_limits)
    ax_dew.set_ylim(y_limits)
    
    ax_dew.set_ylabel('Ponto de Orvalho (°C)', color=C_DEW, fontweight='bold')
    ax_dew.tick_params(axis='y', colors=C_DEW)
    ax_dew.spines['top'].set_visible(False)
    ax_dew.spines['left'].set_visible(False)
    ax_dew.spines['bottom'].set_visible(False)
    ax_dew.spines['right'].set_color(C_DEW)
    
    for i, t in enumerate(times):
        if df_agg['is_day'].iloc[i] == 0:
            # 6h block night highlight
            ax_temp.axvspan(t, t + pd.Timedelta(hours=6), color='#111827', alpha=0.05)

    # --- 3. GRÁFICO CHUVA E NUVENS ---
    ax_precip.grid(False)
    ax_precip.spines['top'].set_visible(False)
    ax_precip.spines['right'].set_visible(False)
    ax_precip.spines['left'].set_visible(False)
    ax_precip.spines['bottom'].set_visible(False)
    
    # Heatmap
    levels = [1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100]
    alts = [44.33 * (1 - (p/1013.25)**0.1903) for p in levels]
    
    cloud_data = []
    for p in levels:
        col = f'cloud_cover_{p}hPa'
        if col in df_hourly.columns:
            cloud_data.append(df_hourly[col].values)
        else:
            cloud_data.append(np.zeros(len(df_hourly)))
    
    Z = np.array(cloud_data)
    cmap_colors = ['#f7f7f7', '#d1d5db', '#9ca3af', '#4b5563', '#1f2937']
    cmap = LinearSegmentedColormap.from_list("cloud_cmap", cmap_colors)
    
    time_nums = mdates.date2num(df_hourly.index)
    T_grid, A_grid = np.meshgrid(time_nums, alts)
    
    ax_precip.pcolormesh(T_grid, A_grid, Z, shading='gouraud', cmap=cmap, vmin=0, vmax=100, alpha=0.9)
    
    # Eixos (Ordem Corrigida)
    # Criar eixo secundário (Chuva) PRIMEIRO
    ax_rain = ax_precip.twinx()
    
    # Configurar Eixo Nuvens (Principal) -> DIREITA
    ax_precip.set_ylim(0, 16)
    ax_precip.yaxis.set_label_position("right")
    ax_precip.yaxis.tick_right()
    ax_precip.set_ylabel('Altitude Nuvens (km)', color=C_MUTED)
    ax_precip.tick_params(axis='y', colors=C_MUTED)
    
    # Configurar Eixo Chuva (Secundário) -> ESQUERDA
    ax_rain.yaxis.set_label_position("left")
    ax_rain.yaxis.tick_left()
    ax_rain.set_ylabel('Precipitação (mm)', color=C_RAIN, fontweight='bold')
    ax_rain.tick_params(axis='y', colors=C_RAIN)
    ax_rain.spines['top'].set_visible(False)
    ax_rain.spines['right'].set_visible(False)
    ax_rain.spines['bottom'].set_visible(False)
    ax_rain.spines['left'].set_color(C_RAIN)
    
    # Barras de chuva (Agregado 6h)
    # Alinhar barras ao centro do intervalo
    # width = 0.2 dias (aprox 5h)
    ax_rain.bar(df_agg.index + pd.Timedelta(hours=3), df_agg['precipitation'], width=0.2, color=C_RAIN, alpha=0.7, zorder=10)
    
    # Rótulos de chuva
    for t, v in df_agg['precipitation'].items():
        if v > 0.1:
            ax_rain.text(t + pd.Timedelta(hours=3), v + 0.2, f"{v:.1f}", ha='center', va='bottom', fontsize=8, color=C_RAIN, fontweight='bold')
            
    ax_rain.set_ylim(0, max(10, df_agg['precipitation'].max() * 1.2))

    # Eixo X comum
    ax_precip.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax_table.get_xticklabels(), visible=False)
    plt.setp(ax_temp.get_xticklabels(), visible=False)
    # plt.setp(ax_precip.get_xticklabels(), visible=False) # Agora é o último, deve mostrar? Não, vamos ver.
    # Normalmente o último mostra datas/horas. Mas a tabela lá em cima já tem.
    # O original removia do ax_wave. Se removermos ax_wave, ax_precip é o ultimo.
    # Mas a tabela (ax_table) já mostra horas.
    plt.setp(ax_precip.get_xticklabels(), visible=False) 
    
    source_label = "GFS/GFS-WAVE/Marinha" if is_litoral else "GFS"
    fig.text(0.5, 0.02, f"Gerado por Geopixel • Fonte: {source_label} • {city_name}", ha='center', fontsize=10, color=C_MUTED)
    
    logo_path = Path(__file__).parent / "geo_logo1.png"
    if logo_path.exists():
        try:
            img = plt.imread(str(logo_path))
            ax_logo = fig.add_axes([0.85, 0.895, 0.14, 0.10], anchor='NE', zorder=12)
            ax_logo.imshow(img)
            ax_logo.axis('off')
        except Exception:
            pass

    now_str = pd.Timestamp.now().strftime('%d/%m %H:%M')
    lat_str = f"{abs(lat):.2f}°S" if lat < 0 else f"{lat:.2f}°N"
    lon_str = f"{abs(lon):.2f}°W" if lon < 0 else f"{lon:.2f}°E"
    info_txt = f"Lat: {lat_str}   Lon: {lon_str}\nRodada: {now_str} Local"
    fig.text(0.02, 0.96, info_txt, ha='left', va='top', fontsize=11, color=C_MUTED, linespacing=1.4)
    fig.text(0.5, 0.95, f"Meteograma — {city_name}", ha='center', va='center', fontsize=18, fontweight='bold', color=C_TEXT)

    plt.savefig(output_path, facecolor=C_BG)
    print(f"Sucesso! Arquivo salvo em: {output_path}")
    plt.close(fig)

def main():
    # Configuração das Cidades
    cities = [
        {"name": "Pouso Alegre, MG", "file_name": "pouso_alegre", "lat": -22.23, "lon": -45.94, "litoral": False},
    ]

    root = Path(__file__).parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for city in cities:
        print(f"\n==================================================")
        print(f"Processando: {city['name']}")
        print(f"==================================================")
        
        lat = city["lat"]
        lon = city["lon"]
        file_name = city["file_name"]
        is_litoral = city["litoral"]
        
        raw_path = data_dir / f"gfs_{file_name}_raw.json"
        csv_path = data_dir / f"gfs_{file_name}_hourly.csv"
        output_path = root / f"meteograma_boletim_{file_name}.png" # Ajustado nome output

        # 1. Download de Dados
        print("--- Etapa 1: Download de Dados GFS e Marine ---")
        df = None
        try:
            # GFS - 3 DIAS APENAS
            d = fetch_gfs(lat, lon, days=3)
            df = to_dataframe(d)
            
            # Marine (apenas se for litoral)
            if is_litoral:
                try:
                    d_marine = fetch_marine(lat, lon, days=3)
                    df_marine = to_dataframe(d_marine)
                    df = df.join(df_marine, rsuffix="_marine")
                    print("Dados marinhos mesclados com sucesso.")
                except Exception as em:
                    print(f"Erro no download marinho: {em}")
            else:
                print("Cidade do interior: pulando dados marinhos.")

            # Salvar backup
            save_json(d, raw_path)
            save_csv(df, csv_path)
            print("Dados baixados e salvos com sucesso.")
            
        except Exception as e:
            print(f"Erro no download: {e}")
            if csv_path.exists():
                print("Tentando usar dados locais em cache...")
                df = pd.read_csv(csv_path)
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
            else:
                print("Não foi possível obter dados. Pulando esta cidade.")
                continue

        # 2. Processamento e Geração do Gráfico
        print("\n--- Etapa 2: Geração do Meteograma ---")
        try:
            if df is not None and not df.empty:
                # Agregação 6h
                df_agg = aggregate_6h(df)
                draw_meteogram(df_agg, df, output_path, city["name"], lat, lon, is_litoral)
            else:
                print("DataFrame vazio ou inválido.")
        except Exception as e:
            print(f"Erro na geração do gráfico: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
