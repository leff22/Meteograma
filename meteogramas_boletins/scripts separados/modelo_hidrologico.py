import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def simple_rain_runoff_model(df_hourly: pd.DataFrame, city_name: str) -> pd.DataFrame:
    default_params = {
        "k_surf": 0.85,
        "k_sub": 0.99,
        "alpha": 0.08,
        "beta": 0.02,
        "base_level": 1.2,
        "thresh_alert": 2.0,
        "thresh_flood": 3.5,
    }

    city_params_db = {
        "São Carlos": {"base_level": 1.5, "alpha": 0.1, "thresh_flood": 4.0},
        "Carapicuíba": {"base_level": 2.0, "k_surf": 0.8, "alpha": 0.15},
    }

    params = default_params.copy()
    for k, v in city_params_db.items():
        if k in city_name:
            params.update(v)
            break

    S_surf = 0.0
    S_sub = 0.0
    levels = []

    precip = df_hourly["precipitation"].fillna(0).values

    for p in precip:
        S_surf += p
        S_sub += p * 0.2

        S_surf *= params["k_surf"]
        S_sub *= params["k_sub"]

        lvl = params["base_level"] + (S_surf * params["alpha"]) + (S_sub * params["beta"])
        levels.append(lvl)

    levels = np.array(levels)

    risk = np.zeros_like(levels)
    base = params["base_level"]
    alert = params["thresh_alert"]
    flood = params["thresh_flood"]

    for i, l in enumerate(levels):
        if l <= base:
            r = 0.0
        elif l < alert:
            r = 5 * (l - base) / (alert - base)
        elif l < flood:
            r = 5 + 4 * (l - alert) / (flood - alert)
        else:
            r = 9 + 1 * (l - flood) / (flood * 0.2)
        risk[i] = np.clip(r, 0, 10)

    return pd.DataFrame(
        {"river_level": levels, "flood_risk": risk},
        index=df_hourly.index,
    )


def run_from_csv(input_csv: Path, city_name: str, output_csv: Path) -> None:
    df = pd.read_csv(input_csv, parse_dates=["time"])
    if "time" in df.columns:
        df = df.set_index("time")

    result = simple_rain_runoff_model(df, city_name)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--city-name", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_from_csv(Path(args.input), args.city_name, Path(args.output))


if __name__ == "__main__":
    main()

