import os
import glob
import sys
import subprocess
import time
import datetime
from io import StringIO
import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.set_page_config(
        page_title="Setup Needed",
        page_icon="🌍",
        layout="wide"
    )
    st.error("Plotly is required for this dashboard.")
    st.code("python -m pip install plotly", language="bash")
    st.stop()

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data"
OUTPUT_DIR = "output"
COORD_PATH = os.path.join(DATA_DIR, "coordinates.csv")

PREDICTOR_SCRIPT_CANDIDATES = [
    "disaster_prediction.py",
    "full_disaster_predictor.py",
    "predictor.py",
]

EONET_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"

FIRE_KEYWORDS = [
    "fire",
    "frp",
    "burn",
    "hotspot",
    "thermal",
    "bright",
    "flame",
    "confidence",
]

HEAT_KEYWORDS = [
    "heat",
    "tmax",
    "temp",
    "temperature",
    "lst",
    "surface_temperature",
    "heatwave",
    "heat_index",
]

CO2_KEYWORDS = [
    "co2",
    "carbon_dioxide",
    "xco2",
    "carbon",
    "greenhouse",
    "ghg",
]

SMOKE_KEYWORDS = [
    "smoke",
    "aerosol",
    "aod",
    "pm25",
    "pm10",
    "particulate",
    "optical_depth",
    "air_quality",
]

HAZE_KEYWORDS = [
    "haze",
    "aod",
    "aerosol",
    "smoke",
    "pm25",
    "pm10",
    "particulate",
    "visibility",
]

FLOOD_KEYWORDS = [
    "flood",
    "inundation",
    "water",
    "precip",
    "rain",
    "rainfall",
    "discharge",
    "river",
]

EMERGENCY_KEYWORDS = [
    "emergency",
    "disaster",
    "event",
    "flood",
    "landslide",
    "heatwave",
    "drought",
    "cyclone",
    "storm",
]

DATE_CANDIDATES = {
    "date",
    "time",
    "datetime",
    "timestamp",
    "day",
    "acq_date",
    "acquisition_date",
    "obs_date",
    "observation_date",
}

REGION_CANDIDATES = {
    "region",
    "place",
    "location",
    "area",
    "country",
    "state",
    "province",
    "district",
    "county",
}

DERIVED_EXCLUDE = {
    "lat",
    "lon",
    "label",
    "disaster_label",
    "__file__",
    "fire_activity",
    "fire_intensity",
    "heat_intensity",
    "co2_intensity",
    "smoke_intensity",
    "haze_intensity",
    "flood_intensity",
    "emergency_signal",
    "emergency_label",
}

HAZARD_COLORS = {
    "Fire": "#ff3b30",
    "Heat": "#ff9500",
    "Flood": "#3b82f6",
    "Storm": "#8b5cf6",
    "Smoke/Haze": "#9ca3af",
    "Volcano": "#b45309",
    "Ice": "#67e8f9",
    "Other": "#22c55e",
}


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Earth Emergency Command",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSS = """
<style>
html, body, [data-testid="stAppViewContainer"], .stApp {
    background:
        radial-gradient(circle at 15% 18%, rgba(56, 189, 248, 0.20), transparent 32%),
        radial-gradient(circle at 82% 12%, rgba(34, 197, 94, 0.16), transparent 30%),
        radial-gradient(circle at 70% 82%, rgba(249, 115, 22, 0.14), transparent 30%),
        linear-gradient(135deg, #020617, #071129 42%, #071c30 72%, #021019);
    color: #e8f7ff;
}

.hero {
    text-align: center;
    padding: 36px 24px 30px 24px;
    border-radius: 30px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 0 70px rgba(0, 191, 255, 0.14);
    margin-bottom: 18px;
    backdrop-filter: blur(10px);
}

.earth-badge {
    font-size: 76px;
    display: inline-block;
    animation: float 4.2s ease-in-out infinite;
    filter: drop-shadow(0 0 22px rgba(56, 189, 248, 0.65));
}

.hero h1 {
    margin-top: 8px;
    font-size: 42px;
    font-weight: 900;
    background: linear-gradient(90deg, #7dd3fc, #4ade80, #facc15, #fb7185);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #cbe8ff;
    font-size: 18px;
}

.card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 12px;
}

.small-note {
    color: #9fb8d8;
    font-size: 13px;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-14px); }
}

[data-testid="stSidebar"] {
    background: rgba(2, 8, 23, 0.92);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding-left: 16px;
    padding-right: 16px;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# SAFE PLOTLY SCATTER GEO
# ============================================================

_SCATTER_GEO = px.scatter_geo


def scatter_geo_safe(data_frame, **kwargs):
    if data_frame is None or data_frame.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No mappable data",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d9f2ff"),
        )
        return fig

    df = data_frame.copy()

    color_col = kwargs.get("color")
    if color_col and color_col in df.columns:
        if df[color_col].isna().all():
            kwargs.pop("color", None)

    size_col = kwargs.get("size")

    if size_col is not None and size_col in df.columns:
        s = pd.to_numeric(df[size_col], errors="coerce").fillna(0.0)

        if not s.empty:
            min_val = s.min()

            if pd.notna(min_val) and min_val < 0:
                s = s - min_val

            s = s.clip(lower=0.0)

            max_val = s.max()

            if pd.notna(max_val) and max_val > 0:
                df["__safe_size__"] = 6.0 + 14.0 * (s / max_val)
            else:
                df["__safe_size__"] = 6.0

            kwargs["size"] = "__safe_size__"

    return _SCATTER_GEO(df, **kwargs)


def make_thermal_fire_map(data_frame, projection="orthographic", height=600):
    """Render FIRMS detections as layered thermal hotspots, not a color ramp."""
    df = data_frame.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        return scatter_geo_safe(df, projection=projection, height=height)

    total_detections = len(df)
    if total_detections > 12000:
        # Keep every strongest hotspot, then add a stable spatial sample so a
        # global NRT feed stays responsive without changing the source table.
        if "frp" in df.columns:
            ranked = df.assign(__sample_rank=pd.to_numeric(df["frp"], errors="coerce").fillna(0))
            strongest = ranked.nlargest(3000, "__sample_rank").drop(columns="__sample_rank")
        else:
            strongest = df.iloc[0:0]
        remainder = df.drop(index=strongest.index)
        sample_count = 12000 - len(strongest)
        random_sample = remainder.sample(n=min(sample_count, len(remainder)), random_state=17)
        df = pd.concat([strongest, random_sample]).drop_duplicates().copy()

    frp = pd.to_numeric(df.get("frp", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0).clip(lower=0)
    log_frp = np.log1p(frp)
    peak = float(log_frp.max()) if not log_frp.empty else 0.0
    strength = log_frp / peak if peak > 0 else pd.Series(0.35, index=df.index)
    df["__frp"] = frp
    df["__strength"] = strength

    custom_columns = [c for c in ["__frp", "confidence", "date", "acq_date", "acq_time", "satellite", "daynight"] if c in df.columns]
    custom_data = df[custom_columns].to_numpy() if custom_columns else None
    hover_lines = ["<b>Thermal hotspot</b>"]
    if "__frp" in custom_columns:
        hover_lines.append("FRP: %{customdata[0]:.1f} MW")
    if "confidence" in custom_columns:
        idx = custom_columns.index("confidence")
        hover_lines.append(f"Confidence: %{{customdata[{idx}]}}")
    if "acq_date" in custom_columns:
        idx = custom_columns.index("acq_date")
        hover_lines.append(f"Observed: %{{customdata[{idx}]}}")
    elif "date" in custom_columns:
        idx = custom_columns.index("date")
        hover_lines.append(f"Observed: %{{customdata[{idx}]}}")
    hover_lines.append("Lat/Lon: %{lat:.2f}, %{lon:.2f}<extra></extra>")
    hovertemplate = "<br>".join(hover_lines)

    fig = go.Figure()
    # Small, restrained halos prevent dense VIIRS pixels from painting whole
    # regions as if they were continuous fires. FRP still controls hotspot size.
    traces = [
        ("rgba(255, 35, 0, 0.045)", 8 + 9 * strength),
        ("rgba(255, 72, 0, 0.14)", 4.5 + 5 * strength),
        ("rgba(255, 174, 32, 0.82)", 2.5 + 3.5 * strength),
        ("rgba(255, 248, 203, 0.96)", 1.2 + 1.0 * strength),
    ]
    for color, sizes in traces:
        fig.add_trace(go.Scattergeo(
            lat=df["lat"],
            lon=df["lon"],
            mode="markers",
            marker=dict(size=sizes, color=color, line=dict(width=0)),
            customdata=custom_data,
            hovertemplate=hovertemplate,
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"Thermal anomaly detections · {len(df):,} of {total_detections:,} plotted"
                if len(df) < total_detections
                else f"Thermal anomaly detections · {total_detections:,}"
            ),
            x=0.02,
            xanchor="left",
            font=dict(size=15, color="#f4e7cf"),
        ),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e9f2ff"),
        margin=dict(l=8, r=8, t=12, b=8),
        geo=dict(
            projection=dict(type=projection),
            showland=True,
            landcolor="#151a22",
            showocean=True,
            oceancolor="#03070d",
            showcountries=True,
            countrycolor="#293341",
            showcoastlines=True,
            coastlinecolor="#657484",
            coastlinewidth=0.7,
            showframe=False,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


# ============================================================
# GENERAL HELPERS
# ============================================================

def make_dark_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d9f2ff"),
        margin=dict(l=25, r=25, t=45, b=35),
    )
    return fig


def make_geo_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=35, b=20),
        geo=dict(
            showland=True,
            landcolor="#0e3b26",
            showocean=True,
            oceancolor="#06182b",
            showcountries=False,
            showcoastlines=True,
            coastlinecolor="#67e8f9",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def zscore_series(s):
    s = pd.to_numeric(s, errors="coerce")

    mean = s.mean()
    std = s.std(ddof=0)

    if pd.isna(std) or std <= 1e-9:
        return pd.Series(0.0, index=s.index)

    return (s - mean) / std


def first_matching_column(df, keywords, exclude=None):
    if df is None:
        return None

    exclude = set(exclude or [])

    for c in df.columns:
        if c.lower() in exclude:
            continue

        if not pd.api.types.is_numeric_dtype(df[c]):
            continue

        if any(k in c.lower() for k in keywords):
            return c

    return None


def find_matching_columns(df, keywords, exclude=None):
    if df is None:
        return []

    exclude = set(exclude or [])
    cols = []

    for c in df.columns:
        if c.lower() in exclude:
            continue

        if not pd.api.types.is_numeric_dtype(df[c]):
            continue

        if any(k in c.lower() for k in keywords):
            cols.append(c)

    return cols


def average_z_intensity(df, cols):
    if df is None or not cols:
        return pd.Series(0.0, index=df.index if df is not None else [])

    # Vectorized group transforms avoid repeated index writes for every unique
    # coordinate. Satellite swath files can contain tens of thousands of groups.
    z_values = []
    for col in cols:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        group = values.groupby(df["region"], sort=False)
        means = group.transform("mean")
        stds = group.transform("std", ddof=0)
        z = ((values - means) / stds.where(stds > 1e-9)).replace([np.inf, -np.inf], np.nan)
        # Constant groups have zero anomaly, matching zscore_series behavior.
        z_values.append(z.fillna(0.0).rename(col))

    if not z_values:
        return pd.Series(0.0, index=df.index)

    z_frame = pd.concat(z_values, axis=1)
    intensity = z_frame.mean(axis=1)
    intensity = intensity.fillna(0.0).clip(lower=0.0)

    return intensity


def limit_map_points(df, max_points=10000):
    if df is None:
        return df

    if len(df) > max_points:
        return df.sample(n=max_points, random_state=0)

    return df


def count_positive_points(point_summary, col):
    if point_summary is None:
        return 0

    if col not in point_summary.columns:
        return 0

    return int((point_summary[col] > 0).sum())


def find_predictor_script():
    for name in PREDICTOR_SCRIPT_CANDIDATES:
        if os.path.exists(name):
            return name

    return None


# ============================================================
# LOCAL DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_all_data():
    files = []

    patterns = [
        "*.csv",
        "*.cvv",
    ]

    for pattern in patterns:
        files.extend(
            glob.glob(
                os.path.join(DATA_DIR, "**", pattern),
                recursive=True
            )
        )

    files = sorted(files)

    ignore_words = [
        "readme",
        "template",
        "coordinates",
        "archive",
        "backup",
    ]

    files = [
        f for f in files
        if not any(word in os.path.basename(f).lower() for word in ignore_words)
    ]

    frames = []
    summaries = []
    logs = []

    if not files:
        logs.append("No .csv or .cvv files found in data/")

    for path in files:
        name = os.path.basename(path)

        try:
            df = pd.read_csv(path)

            if df.empty:
                logs.append(f"Skipped {name}: file is empty")
                continue

            for c in list(df.columns):
                lc = c.lower()

                if lc in {"latitude", "lat"} and "lat" not in df.columns:
                    df = df.rename(columns={c: "lat"})

                elif lc in {"longitude", "lon", "long"} and "lon" not in df.columns:
                    df = df.rename(columns={c: "lon"})

            if "date" not in df.columns:
                date_candidates = [
                    c for c in df.columns
                    if c.lower() in DATE_CANDIDATES
                ]

                if date_candidates:
                    df = df.rename(columns={date_candidates[0]: "date"})
                else:
                    logs.append(f"Skipped {name}: no date column detected")
                    continue

            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            before = len(df)
            df = df.dropna(subset=["date"])
            after = len(df)

            if before != after:
                logs.append(
                    f"{name}: removed {before - after} rows with invalid dates"
                )

            if df.empty:
                logs.append(f"Skipped {name}: no valid dates after parsing")
                continue

            if "region" not in df.columns:
                region_candidates = [
                    c for c in df.columns
                    if c.lower() in REGION_CANDIDATES
                ]

                if region_candidates:
                    df = df.rename(columns={region_candidates[0]: "region"})

            if "region" not in df.columns:
                if {"lat", "lon"}.issubset(df.columns):
                    lat = pd.to_numeric(df["lat"], errors="coerce").round(2)
                    lon = pd.to_numeric(df["lon"], errors="coerce").round(2)

                    df["region"] = lat.astype(str) + "," + lon.astype(str)
                else:
                    df["region"] = "global"

            for c in ["lat", "lon"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            if "label" in df.columns and "disaster_label" not in df.columns:
                df["disaster_label"] = df["label"]

            if "disaster_label" in df.columns:
                df["disaster_label"] = (
                    pd.to_numeric(df["disaster_label"], errors="coerce")
                    .fillna(0)
                )

            df["__file__"] = name

            numeric_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c.lower() not in {"lat", "lon", "disaster_label", "label"}
            ]

            summaries.append(
                {
                    "file": name,
                    "rows": len(df),
                    "numeric_variables": len(numeric_cols),
                    "regions": df["region"].nunique(),
                    "date_min": df["date"].min(),
                    "date_max": df["date"].max(),
                    "has_coordinates": bool({"lat", "lon"}.issubset(df.columns)),
                    "disaster_labels": int(df["disaster_label"].sum())
                    if "disaster_label" in df.columns
                    else 0,
                }
            )

            frames.append(df)
            logs.append(f"Loaded {name}: {len(df)} rows")

        except Exception as exc:
            logs.append(f"Skipped {name}: {exc}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("date")
    else:
        combined = None

    summary = pd.DataFrame(summaries)

    return combined, summary, logs


@st.cache_data(show_spinner=False)
def load_output_csv(filename):
    path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    return None


@st.cache_data(show_spinner=False)
def load_coordinates():
    if not os.path.exists(COORD_PATH):
        return None

    try:
        coords = pd.read_csv(COORD_PATH)

        if "region" not in coords.columns:
            return None

        if "lat" not in coords.columns:
            coords["lat"] = np.nan

        if "lon" not in coords.columns:
            coords["lon"] = np.nan

        coords["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
        coords["lon"] = pd.to_numeric(coords["lon"], errors="coerce")

        coords = coords.groupby("region", as_index=False)[["lat", "lon"]].median()

        return coords

    except Exception:
        return None


def apply_coordinates(raw_data):
    if raw_data is None:
        return raw_data

    if "lat" not in raw_data.columns:
        raw_data["lat"] = np.nan

    if "lon" not in raw_data.columns:
        raw_data["lon"] = np.nan

    raw_data["lat"] = pd.to_numeric(raw_data["lat"], errors="coerce")
    raw_data["lon"] = pd.to_numeric(raw_data["lon"], errors="coerce")

    coords = load_coordinates()

    if coords is None:
        return raw_data

    missing = raw_data["lat"].isna() | raw_data["lon"].isna()

    if missing.any():
        raw_data = raw_data.merge(
            coords,
            on="region",
            how="left",
            suffixes=("", "_coord")
        )

        raw_data["lat"] = raw_data["lat"].fillna(raw_data["lat_coord"])
        raw_data["lon"] = raw_data["lon"].fillna(raw_data["lon_coord"])

        raw_data = raw_data.drop(
            columns=["lat_coord", "lon_coord"],
            errors="ignore"
        )

    return raw_data


# ============================================================
# LOCAL HAZARD DETECTION
# ============================================================

def add_derived_columns(df):
    notes = []

    if df is None:
        return None, notes

    # FIRE: only use direct fire/thermal detection fields. Temperature, soil,
    # and vegetation are not fire observations and must not be plotted as if
    # they were; keep missing rows missing instead of manufacturing a proxy.
    fire_cols = find_matching_columns(
        df,
        FIRE_KEYWORDS,
        exclude=DERIVED_EXCLUDE
    )

    if fire_cols:
        df["fire_activity"] = df[fire_cols].mean(axis=1)
        fire_available = df[fire_cols].notna().any(axis=1)
        df["fire_intensity"] = average_z_intensity(df, ["fire_activity"]).where(fire_available)
        notes.append(f"Fire columns detected: {', '.join(fire_cols)}")
    else:
        df["fire_activity"] = np.nan
        df["fire_intensity"] = np.nan
        notes.append("No direct fire detections found; fire values are left blank rather than inferred from weather or vegetation.")

    # HEAT
    heat_cols = find_matching_columns(
        df,
        HEAT_KEYWORDS,
        exclude=DERIVED_EXCLUDE
    )

    if heat_cols:
        df["heat_intensity"] = average_z_intensity(df, heat_cols)
        notes.append(f"Heat columns detected: {', '.join(heat_cols)}")

    else:
        temp_col = first_matching_column(
            df,
            ["tmax", "temp", "temperature", "heat", "lst", "surface_temperature"],
            exclude=DERIVED_EXCLUDE
        )

        if temp_col:
            df["heat_intensity"] = average_z_intensity(df, [temp_col])
            notes.append("Created heat_intensity proxy from temperature column.")
        else:
            df["heat_intensity"] = 0.0
            notes.append("No heat detection columns found.")

    # CO2
    co2_cols = find_matching_columns(
        df,
        CO2_KEYWORDS,
        exclude=DERIVED_EXCLUDE
    )

    if co2_cols:
        df["co2_intensity"] = average_z_intensity(df, co2_cols)
        notes.append(f"CO2 columns detected: {', '.join(co2_cols)}")

    else:
        if df["fire_intensity"].abs().sum() > 0:
            df["co2_intensity"] = df["fire_intensity"] * 0.35
            notes.append(
                "No CO2 columns found. "
                "Created weak CO2 proxy from fire activity."
            )
        else:
            df["co2_intensity"] = 0.0
            notes.append(
                "No CO2 columns found. "
                "Add OCO-2 / OCO-3 / GOSAT CO2 data for real CO2 tracking."
            )

    # SMOKE
    smoke_cols = find_matching_columns(
        df,
        SMOKE_KEYWORDS,
        exclude=DERIVED_EXCLUDE
    )

    if smoke_cols:
        df["smoke_intensity"] = average_z_intensity(df, smoke_cols)
        notes.append(f"Smoke/aerosol columns detected: {', '.join(smoke_cols)}")

    else:
        if df["fire_intensity"].abs().sum() > 0:
            df["smoke_intensity"] = df["fire_intensity"] * 0.75
            notes.append("Created smoke proxy from fire activity.")
        else:
            df["smoke_intensity"] = 0.0
            notes.append("No smoke/aerosol columns found.")

    # HAZE
    haze_cols = find_matching_columns(
        df,
        HAZE_KEYWORDS,
        exclude=DERIVED_EXCLUDE
    )

    if haze_cols:
        df["haze_intensity"] = average_z_intensity(df, haze_cols)
        notes.append(f"Haze columns detected: {', '.join(haze_cols)}")

    else:
        df["haze_intensity"] = df["smoke_intensity"]
        notes.append("Using smoke signal as haze proxy.")

    # FLOOD
    flood_cols = find_matching_columns(
        df,
        FLOOD_KEYWORDS,
        exclude=DERIVED_EXCLUDE
    )

    if flood_cols:
        df["flood_intensity"] = average_z_intensity(df, flood_cols)
        notes.append(f"Flood columns detected: {', '.join(flood_cols)}")

    else:
        precip_col = first_matching_column(
            df,
            ["precip", "rain", "rainfall"],
            exclude=DERIVED_EXCLUDE
        )

        soil_col = first_matching_column(
            df,
            ["soil", "moisture"],
            exclude=DERIVED_EXCLUDE
        )

        proxy_cols = [c for c in [precip_col, soil_col] if c]

        if proxy_cols:
            df["flood_intensity"] = average_z_intensity(df, proxy_cols)
            notes.append("Created flood proxy from precipitation/soil moisture.")
        else:
            df["flood_intensity"] = 0.0
            notes.append("No flood detection columns found.")

    # EMERGENCY SIGNAL
    emergency_cols = find_matching_columns(
        df,
        EMERGENCY_KEYWORDS,
        exclude=DERIVED_EXCLUDE
    )

    direct_emergency = pd.Series(0.0, index=df.index)

    if emergency_cols:
        direct_emergency = average_z_intensity(df, emergency_cols)
        notes.append(f"Emergency columns detected: {', '.join(emergency_cols)}")

    hazard_cols = [
        "fire_intensity",
        "heat_intensity",
        "smoke_intensity",
        "haze_intensity",
        "flood_intensity",
        "co2_intensity",
    ]

    df["__direct_emergency"] = direct_emergency
    df["emergency_signal"] = df[hazard_cols + ["__direct_emergency"]].max(axis=1)
    df = df.drop(columns=["__direct_emergency"])

    # EMERGENCY LABELS
    if "disaster_label" in df.columns:
        df["emergency_label"] = (
            pd.to_numeric(df["disaster_label"], errors="coerce")
            .fillna(0)
        )

    elif "label" in df.columns:
        df["emergency_label"] = (
            pd.to_numeric(df["label"], errors="coerce")
            .fillna(0)
        )

    else:
        threshold = df["emergency_signal"].quantile(0.95)

        if pd.notna(threshold) and threshold > 0:
            df["emergency_label"] = (
                df["emergency_signal"] >= threshold
            ).astype(float)

            notes.append(
                "No disaster label found. "
                "Created emergency_label from top 5% emergency_signal."
            )
        else:
            df["emergency_label"] = 0.0

    return df, notes


# ============================================================
# LOCAL POINT / REGION TRACKING
# ============================================================

def build_point_summary(df):
    if df is None or df.empty:
        return None

    if "lat" not in df.columns or "lon" not in df.columns:
        return None

    p = df.dropna(subset=["lat", "lon"]).copy()

    if p.empty:
        return None

    p["lat"] = p["lat"].round(3)
    p["lon"] = p["lon"].round(3)

    agg_kwargs = dict(
        rows=("date", "count"),
        last_date=("date", "max"),
    )

    if "emergency_label" in p.columns:
        agg_kwargs["emergency_labels"] = ("emergency_label", "sum")

    intensity_cols = [
        "fire_intensity",
        "heat_intensity",
        "smoke_intensity",
        "haze_intensity",
        "flood_intensity",
        "co2_intensity",
        "emergency_signal",
    ]

    for c in intensity_cols:
        if c in p.columns:
            agg_kwargs[c] = (c, "mean")

    point = p.groupby(
        ["region", "lat", "lon"],
        as_index=False
    ).agg(**agg_kwargs)

    return point


def get_risk_latest(risk):
    if risk is None or risk.empty:
        return None

    if "region" not in risk.columns:
        return None

    risk_col = None

    preferred = [
        "risk_next_horizon",
        "risk",
        "probability",
        "predicted_risk",
        "hazard_risk",
    ]

    for c in preferred:
        if c in risk.columns:
            risk_col = c
            break

    if risk_col is None:
        numeric_cols = risk.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            risk_col = numeric_cols[0]

    if risk_col is None:
        return None

    risk = risk.copy()

    if "date" in risk.columns:
        risk["date"] = pd.to_datetime(risk["date"], errors="coerce")

        latest = (
            risk.sort_values("date")
            .groupby("region", as_index=False)[risk_col]
            .last()
        )
    else:
        latest = risk.groupby("region", as_index=False)[risk_col].mean()

    latest = latest.rename(columns={risk_col: "latest_risk"})

    return latest


def build_region_registry(raw_data, point_summary, risk_latest):
    # Build each regional summary once, then join. Re-filtering all point and
    # raw-data tables inside a loop over every region becomes quadratic for
    # satellite pixel files with tens of thousands of locations.
    parts = []

    if raw_data is not None and "region" in raw_data.columns:
        agg = {"date": "count"} if "date" in raw_data.columns else {}
        if "emergency_label" in raw_data.columns:
            agg["emergency_label"] = "sum"
        if "fire_intensity" in raw_data.columns:
            agg["fire_intensity"] = "mean"
        if agg:
            raw_stats = raw_data.groupby("region", as_index=False).agg(agg)
            raw_stats = raw_stats.rename(columns={
                "date": "rows",
                "emergency_label": "emergency_labels",
            })
            parts.append(raw_stats)
        else:
            parts.append(raw_data[["region"]].drop_duplicates())

    if point_summary is not None and not point_summary.empty:
        point_locations = (
            point_summary.groupby("region", as_index=False)[["lat", "lon"]]
            .median()
            .rename(columns={"lat": "lat_observed", "lon": "lon_observed"})
        )
        parts.append(point_locations)

    coords = load_coordinates()
    if coords is not None and not coords.empty:
        parts.append(coords[["region", "lat", "lon"]].rename(columns={
            "lat": "lat_coord", "lon": "lon_coord"
        }))

    if risk_latest is not None and not risk_latest.empty and "region" in risk_latest:
        parts.append(risk_latest[[c for c in ["region", "latest_risk"] if c in risk_latest]])

    if not parts:
        return pd.DataFrame(columns=[
            "region", "lat", "lon", "rows", "emergency_labels",
            "fire_intensity", "latest_risk",
        ])

    registry = parts[0]
    for part in parts[1:]:
        registry = registry.merge(part, on="region", how="outer")

    # Prefer coordinates observed in the data, then use the coordinate manager.
    for col in ("lat", "lon"):
        observed_col = f"{col}_observed"
        coord_col = f"{col}_coord"
        observed = registry[observed_col] if observed_col in registry.columns else pd.Series(np.nan, index=registry.index)
        saved = registry[coord_col] if coord_col in registry.columns else pd.Series(np.nan, index=registry.index)
        registry[col] = observed.combine_first(saved)
        registry = registry.drop(columns=[observed_col, coord_col], errors="ignore")

    for col in ("rows", "emergency_labels", "fire_intensity"):
        if col not in registry.columns:
            registry[col] = 0
        registry[col] = registry[col].fillna(0)

    if "latest_risk" not in registry.columns:
        registry["latest_risk"] = np.nan

    return registry.sort_values("region", kind="stable").reset_index(drop=True)


def ensure_coordinates_file(registry):
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(COORD_PATH):
        if registry is None or registry.empty:
            pd.DataFrame(columns=["region", "lat", "lon"]).to_csv(
                COORD_PATH,
                index=False
            )
        else:
            registry[["region", "lat", "lon"]].to_csv(
                COORD_PATH,
                index=False
            )


# ============================================================
# LIVE NASA EONET / FIRMS
# ============================================================

def classify_live_hazard(title, categories):
    text = (str(title) + " " + str(categories)).lower()

    if any(k in text for k in ["wildfire", "fire", "burn"]):
        return "Fire"

    if any(k in text for k in ["heatwave", "heat wave", "temperature extremes", "extreme heat", "heat"]):
        return "Heat"

    if any(k in text for k in ["flood", "inundation", "flash flood"]):
        return "Flood"

    if any(k in text for k in ["storm", "cyclone", "hurricane", "typhoon", "severe storm"]):
        return "Storm"

    if any(k in text for k in ["dust", "haze", "smoke", "aerosol", "air quality"]):
        return "Smoke/Haze"

    if any(k in text for k in ["volcano", "volcanic"]):
        return "Volcano"

    if any(k in text for k in ["ice", "iceberg", "sea ice"]):
        return "Ice"

    return "Other"


def parse_live_geometry(geom):
    """Return a representative point for EONET GeoJSON geometry."""
    try:
        gtype = geom.get("type")
        coords = geom.get("coordinates")

        if coords is None:
            return None, None

        if gtype == "Point":
            lon, lat = coords[0], coords[1]
            return lat, lon

        def flatten_positions(value):
            if isinstance(value, (list, tuple)) and len(value) >= 2 and all(
                isinstance(v, (int, float)) for v in value[:2]
            ):
                yield value
            elif isinstance(value, (list, tuple)):
                for child in value:
                    yield from flatten_positions(child)

        positions = list(flatten_positions(coords))
        if positions:
            # Polygon borders can cross the dateline; selecting an observed
            # coordinate is safer than averaging longitudes across +/-180.
            lon, lat = positions[len(positions) // 2][:2]
            return float(lat), float(lon)

    except Exception:
        return None, None

    return None, None


@st.cache_data(show_spinner=False, ttl=120)
def fetch_eonet_events(limit=200):
    """Fetch live NASA EONET events; never substitute fabricated events."""

    # ---------- NASA EONET v3 (free, no auth) ----------

    if HAS_REQUESTS:
        params = {
            "status": "open",
            "limit": limit,
        }

        try:
            r = requests.get(EONET_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            events = data.get("events", [])

            if not events:
                return pd.DataFrame(columns=[
                    "event_id", "title", "categories", "hazard", "date", "lat", "lon", "link"
                ]), None

            rows = []

            for event in events:
                event_id = event.get("id")
                title = event.get("title", "")
                link = event.get("link", "")

                categories = [
                    c.get("title", "")
                    for c in event.get("categories", [])
                ]

                hazard = classify_live_hazard(title, categories)

                # EONET v3 names this field `geometry` (singular). Accept the
                # older/plural spelling too, but never drop an event just
                # because its current record lacks coordinates.
                geometries = event.get("geometry", event.get("geometries", [])) or []
                if isinstance(geometries, dict):
                    geometries = [geometries]

                source_items = event.get("sources", []) or []
                source_text = "; ".join(
                    " — ".join(str(part) for part in [
                        source.get("id") or source.get("title"),
                        source.get("url") or source.get("source") or source.get("link"),
                    ] if part)
                    for source in source_items
                    if isinstance(source, dict)
                )
                base_row = {
                    "event_id": event_id,
                    "title": title,
                    "description": event.get("description", ""),
                    "categories": ", ".join(categories),
                    "hazard": hazard,
                    "date": None,
                    "lat": np.nan,
                    "lon": np.nan,
                    "sources": source_text,
                    "link": link,
                }

                event_rows = []
                for geom in geometries:
                    if not isinstance(geom, dict):
                        continue
                    lat, lon = parse_live_geometry(geom)
                    row = dict(base_row)
                    row["date"] = geom.get("date")
                    if lat is not None and lon is not None:
                        row["lat"], row["lon"] = float(lat), float(lon)
                    event_rows.append(row)

                if event_rows:
                    rows.extend(event_rows)
                else:
                    rows.append(base_row)

            df = pd.DataFrame(rows)

            if not df.empty:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df.attrs["event_count"] = len(events)
                return df, None
            return df, None

        except Exception as exc:
            if HAS_REQUESTS and isinstance(exc, requests.exceptions.ConnectionError):
                return None, "This app's Python process could not connect to NASA EONET. Your PowerShell check confirms NASA is reachable there; Python may use different firewall/proxy settings. EONET is public and does not use the FIRMS MAP_KEY."
            return None, f"NASA EONET request failed: {type(exc).__name__}."

    return None, "Live NASA EONET data requires the requests package and internet access."


def get_firms_map_key():
    """Read the FIRMS key from the environment or Streamlit secrets."""
    key = os.getenv("NASA_FIRMS_MAP_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("NASA_FIRMS_MAP_KEY", "")).strip()
    except Exception:
        return ""


@st.cache_data(show_spinner=False, ttl=600)
def fetch_firms_url(url):
    if not url:
        return None, "No FIRMS URL provided."

    try:
        df = pd.read_csv(url)

        if df.empty:
            return None, "FIRMS URL returned an empty table."

        for c in list(df.columns):
            lc = c.lower()

            if lc in {"latitude", "lat"} and "lat" not in df.columns:
                df = df.rename(columns={c: "lat"})

            elif lc in {"longitude", "lon", "long"} and "lon" not in df.columns:
                df = df.rename(columns={c: "lon"})

        if "date" not in df.columns:
            date_candidates = [
                c for c in df.columns
                if c.lower() in {
                    "date",
                    "acq_date",
                    "acquisition_date",
                    "time",
                    "datetime",
                    "timestamp",
                }
            ]

            if date_candidates:
                df = df.rename(columns={date_candidates[0]: "date"})

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for c in ["lat", "lon"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["lat", "lon"])

        if df.empty:
            return None, "No valid latitude/longitude rows found in FIRMS data."

        return df, None

    except Exception as exc:
        return None, f"Could not load the FIRMS CSV URL: {type(exc).__name__}."


def show_eonet_connection_help(message):
    st.error(message)
    st.link_button("Open NASA EONET directly", "https://eonet.gsfc.nasa.gov/")
    st.caption("EONET is a separate NASA service and does not use the FIRMS MAP_KEY.")


@st.cache_data(show_spinner=False, ttl=600)
def fetch_firms_api(map_key, source="VIIRS_SNPP_NRT", day_range=1):
    """Fetch recent global fire detections from NASA FIRMS area CSV API."""
    if not map_key:
        return None, "Enter a FIRMS MAP_KEY or set NASA_FIRMS_MAP_KEY."
    if not HAS_REQUESTS:
        return None, "Install requests to fetch FIRMS data."

    # Keep the key out of displayed errors and use the official FIRMS area API.
    url = (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{map_key}/{source}/world/{int(day_range)}"
    )
    try:
        response = requests.get(url, timeout=45)
        if response.status_code != 200:
            return None, f"NASA FIRMS returned HTTP {response.status_code}. Check the MAP_KEY and source."
        if response.text.strip().lower().startswith(("invalid", "error")):
            return None, "NASA FIRMS rejected the MAP_KEY or satellite feed. Check both values."
        df = pd.read_csv(StringIO(response.text))
        if df.empty:
            return None, "NASA FIRMS returned no fire detections for this period."
        for old, new in (("latitude", "lat"), ("longitude", "lon"), ("acq_date", "date")):
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ("lat", "lon", "frp", "bright_ti4"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["lat", "lon"])
        if df.empty:
            return None, "FIRMS response did not contain valid latitude/longitude rows."
        return df, None
    except Exception as exc:
        if isinstance(exc, requests.exceptions.ConnectionError):
            return None, "This app could not connect to NASA FIRMS. Check that Python is allowed through your firewall/VPN/proxy."
        return None, f"Could not fetch NASA FIRMS data: {type(exc).__name__}."


def local_firms_snapshot(raw_data):
    """Return locally saved NASA VIIRS FIRMS rows as an explicitly dated snapshot."""
    if raw_data is None or raw_data.empty or "__file__" not in raw_data.columns:
        return None
    source_mask = raw_data["__file__"].astype(str).str.contains(
        r"J1_VIIRS|FIRMS", case=False, regex=True
    )
    snapshot = raw_data.loc[source_mask].copy()
    if snapshot.empty or not {"lat", "lon"}.issubset(snapshot.columns):
        return None
    snapshot["lat"] = pd.to_numeric(snapshot["lat"], errors="coerce")
    snapshot["lon"] = pd.to_numeric(snapshot["lon"], errors="coerce")
    if "frp" in snapshot.columns:
        snapshot["frp"] = pd.to_numeric(snapshot["frp"], errors="coerce")
    if "date" in snapshot.columns:
        snapshot["date"] = pd.to_datetime(snapshot["date"], errors="coerce")
    return snapshot.dropna(subset=["lat", "lon"])


# ============================================================
# LOAD LOCAL PREDICTION SYSTEM
# ============================================================

raw_data, file_summary, logs = load_all_data()

raw_data = apply_coordinates(raw_data)
raw_data, derived_notes = add_derived_columns(raw_data)

point_summary = build_point_summary(raw_data)

risk = load_output_csv("future_risk.csv")
risk_latest = get_risk_latest(risk)
trends = load_output_csv("trend_report.csv")

registry = build_region_registry(raw_data, point_summary, risk_latest)
ensure_coordinates_file(registry)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="earth-badge">🌍</div>
        <h1>Earth Emergency Command</h1>
        <p>
            Live NASA events, fire feed, local hazard detection,
            coordinates, and disaster prediction in one dashboard.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🛰️ Command Panel")

    if st.button("🔄 Refresh All Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    st.markdown("### 🧠 Prediction Engine")

    predictor_script = find_predictor_script()

    if predictor_script:
        st.caption(f"Detected predictor: `{predictor_script}`")

        if st.button("▶️ Run Prediction"):
            with st.spinner("Running prediction script..."):
                result = subprocess.run(
                    [sys.executable, predictor_script],
                    capture_output=True,
                    text=True
                )

            if result.stdout:
                st.code(result.stdout[-3000:])

            if result.returncode != 0:
                st.error("Prediction script failed.")
                if result.stderr:
                    st.code(result.stderr[-3000:])
            else:
                st.success("Prediction script finished.")

            st.cache_data.clear()
            st.rerun()

    else:
        st.warning(
            "No prediction script found. "
            "Expected one of: "
            + ", ".join(PREDICTOR_SCRIPT_CANDIDATES)
        )

    st.markdown("---")

    auto_refresh = st.checkbox("Auto refresh live data", value=True)

    refresh_interval = st.slider(
        "Refresh interval seconds",
        min_value=30,
        max_value=900,
        value=120,
        step=30
    )

    if HAS_AUTOREFRESH and auto_refresh:
        st_autorefresh(
            interval=refresh_interval * 1000,
            key="unified_dashboard_refresh"
        )
    elif auto_refresh:
        st.warning("Automatic refresh needs streamlit-autorefresh. Install the project requirements and restart the dashboard.")

    st.markdown("---")

    event_limit = st.slider(
        "Max live EONET events",
        min_value=25,
        max_value=500,
        value=200,
        step=25
    )

    show_all_geometry = st.checkbox(
        "Show full EONET geometry history",
        value=False
    )

    st.markdown("---")

    st.markdown("### NASA FIRMS Live Fires")

    firms_key = get_firms_map_key()
    if not firms_key:
        firms_key = st.text_input(
            "FIRMS MAP_KEY",
            value="",
            type="password",
            help="Used for this dashboard session only. Or set NASA_FIRMS_MAP_KEY before launching."
        ).strip()
    else:
        st.caption("FIRMS MAP_KEY loaded from environment or Streamlit secrets.")

    firms_source = st.selectbox(
        "Satellite feed",
        ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"],
        help="Near real-time active fire detections from NASA FIRMS."
    )
    firms_days = st.slider("FIRMS lookback days", 1, 10, 1)
    include_low_confidence = st.checkbox(
        "Include low-confidence detections",
        value=False,
        help="Off by default to reduce likely false alarms. Nominal detections are still algorithm detections, not confirmed ground reports."
    )
    firms_url = st.text_input(
        "Optional FIRMS CSV URL override",
        value="",
        help="Leave blank to use the MAP_KEY above and fetch from the FIRMS API."
    ).strip()

    st.markdown("### Data Table Size")
    table_size = st.selectbox(
        "Rows to show in Data Tables",
        ["100% Size", "50% Size", "25% Size", "All"],
        index=0,
        help="The percentage options show that share of the first 1,000 rows. All shows every loaded row."
    )

    st.markdown("---")

    st.markdown("### Derived Signal Notes")

    if derived_notes:
        for note in derived_notes:
            st.caption(note)
    else:
        st.caption("No derived signal notes.")


# ============================================================
# FETCH LIVE DATA
# ============================================================

live_events, live_error = fetch_eonet_events(limit=event_limit)

if firms_url:
    firms_df, firms_error = fetch_firms_url(firms_url)
    firms_is_live = False
elif firms_key:
    firms_df, firms_error = fetch_firms_api(firms_key, firms_source, firms_days)
    firms_is_live = firms_df is not None
else:
    firms_df, firms_error = None, (
        "No NASA FIRMS MAP_KEY reached this dashboard process. Enter it in the "
        "password field in the sidebar, or launch Streamlit from the PowerShell "
        "session where NASA_FIRMS_MAP_KEY is set."
    )
    firms_is_live = False

if firms_df is None:
    local_firms_df = local_firms_snapshot(raw_data)
    if local_firms_df is not None and not local_firms_df.empty:
        firms_df = local_firms_df
        latest_local_observation = firms_df["date"].max() if "date" in firms_df.columns else pd.NaT
        snapshot_date = (
            latest_local_observation.strftime("%Y-%m-%d %H:%M UTC")
            if pd.notna(latest_local_observation)
            else "date unavailable"
        )
        firms_error = (
            f"Live FIRMS unavailable ({firms_error}). Showing saved NASA VIIRS "
            f"CSV snapshot instead; latest observation: {snapshot_date}."
        )

if firms_df is not None and not include_low_confidence and "confidence" in firms_df.columns:
    # VIIRS confidence is categorical (low/nominal/high); retain unknown values
    # so alternative FIRMS feeds with different schemas are not silently lost.
    confidence = firms_df["confidence"].astype(str).str.strip().str.lower()
    firms_df = firms_df.loc[confidence.ne("low")].copy()

if live_events is not None and not live_events.empty and not show_all_geometry:
    live_events = (
        live_events.sort_values("date")
        .groupby("event_id", as_index=False)
        .tail(1)
    )


# ============================================================
# METRICS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

files_loaded = 0 if file_summary.empty else len(file_summary)

if raw_data is not None:
    total_rows = len(raw_data)
    total_regions = raw_data["region"].nunique()
    total_points = 0 if point_summary is None else len(point_summary)
    emergency_labels = int(raw_data["emergency_label"].sum()) if "emergency_label" in raw_data.columns else 0
else:
    total_rows = 0
    total_regions = 0
    total_points = 0
    emergency_labels = 0

max_risk = 0.0

if risk_latest is not None and "latest_risk" in risk_latest.columns:
    max_risk = float(risk_latest["latest_risk"].max())

live_count = 0 if live_events is None else len(live_events)
live_fire_count = 0
live_flood_count = 0

if live_events is not None and not live_events.empty:
    live_fire_count = int((live_events["hazard"] == "Fire").sum())
    live_flood_count = int((live_events["hazard"] == "Flood").sum())

with c1:
    st.metric("📁 Local Files", files_loaded)

with c2:
    st.metric("📊 Local Rows", total_rows)

with c3:
    st.metric("🌍 Regions", total_regions)

with c4:
    st.metric("📍 Points", total_points)

with c5:
    st.metric("🔮 Max Predicted Risk", f"{max_risk:.2f}")


d1, d2, d3, d4, d5 = st.columns(5)

with d1:
    st.metric("🛰️ Live Events", live_count)

with d2:
    st.metric("🔥 Live Fire Events", live_fire_count)

with d3:
    st.metric("🌊 Live Flood Events", live_flood_count)

with d4:
    st.metric("🚨 Local Emergency Labels", emergency_labels)

with d5:
    st.metric("🕒 Checked", datetime.datetime.now().strftime("%H:%M:%S"))


# ============================================================
# TABS
# ============================================================

tab_command, tab_tables, tab_global_map, tab_prediction, tab_local_hazards, tab_atmosphere, tab_live_events, tab_live_fire, tab_coordinates, tab_sources = st.tabs(
    [
        "🌍 Mission Control",
        "📋 Data Tables",
        "🌐 Global Map",
        "🔮 Prediction",
        "🛰️ Local Hazards",
        "🌫️ Air & Atmosphere",
        "🚨 Live Events",
        "🔥 Live Fire",
        "🧭 Coordinates",
        "📚 Sources",
    ]
)


# ============================================================
# TAB: MISSION CONTROL
# ============================================================

with tab_command:
    st.markdown("## 🌍 Mission Control")

    st.markdown("### Local Input Files")

    if file_summary.empty:
        st.warning(
            "No local input files found. Put `.csv` or `.cvv` files into `data/`."
        )
    else:
        st.dataframe(file_summary, width="stretch")

    st.markdown("### Region Registry")

    if registry.empty:
        st.info("No regions detected.")
    else:
        st.dataframe(registry, width="stretch")

    st.markdown("### Loading Log")

    if logs:
        for log in logs:
            st.text(log)
    else:
        st.info("No log messages.")


# ============================================================
# TAB: GLOBAL MAP
# ============================================================

with tab_global_map:
    st.markdown("## 🌐 Global Map")

    layer_options = [
        "Live EONET Events",
        "FIRMS Fires",
        "Predicted Risk",
        "Local Fire proxy",
        "Local Heat proxy",
        "Local Smoke proxy",
        "Local Haze proxy",
        "Local Flood proxy",
        "Local CO₂ proxy",
    ]

    layer = st.selectbox("Map layer", layer_options)

    map_style = st.radio(
        "Map style",
        ["Globe", "Flat world"],
        horizontal=True,
        key="global_map_style"
    )

    projection = "orthographic" if map_style == "Globe" else "natural earth"

    if layer == "Live EONET Events":
        if live_events is None:
            show_eonet_connection_help(live_error)
        elif live_error:
            st.warning(f"⚠️ {live_error}")

        elif live_events.empty:
            st.info("No live EONET events found.")

        else:
            hazards = sorted(live_events["hazard"].unique().tolist())

            selected_hazards = st.multiselect(
                "Filter hazards",
                hazards,
                default=hazards
            )

            map_df = live_events[live_events["hazard"].isin(selected_hazards)].copy()

            if map_df.empty:
                st.info("No hazards selected.")

            else:
                mapped_events = map_df.dropna(subset=["lat", "lon"])
                if not mapped_events.empty:
                    fig = px.scatter_geo(
                        mapped_events,
                        lat="lat",
                        lon="lon",
                        color="hazard",
                        hover_name="title",
                        hover_data=["categories", "date"],
                        color_discrete_map=HAZARD_COLORS,
                        projection=projection,
                        height=600,
                    )
                    fig = make_geo_fig(fig)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("These events have no coordinates to plot. Their event records are shown below.")

                st.dataframe(map_df, width="stretch")

    elif layer == "FIRMS Fires":
        if firms_df is None:
            st.info(firms_error)
        else:
            if not firms_is_live:
                st.warning(firms_error)
            fig = make_thermal_fire_map(firms_df, projection=projection, height=640)
            st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})
            st.caption("NASA FIRMS thermal anomaly detections, filtered to nominal/high confidence by default. These pixels are not all confirmed wildfires.")

            st.dataframe(firms_df.head(1000), width="stretch")

    else:
        base_map = None

        if point_summary is not None and not point_summary.empty:
            base_map = point_summary.copy()
        elif registry is not None and not registry.empty:
            base_map = registry.copy()

        if base_map is None or base_map.empty:
            st.warning("No local coordinate data available.")

        else:
            color = None
            size = None
            color_scale = "Viridis"

            if layer == "Predicted Risk":
                if risk_latest is None:
                    st.warning("No prediction found. Run the prediction script.")

                else:
                    base_map = base_map.merge(
                        risk_latest,
                        on="region",
                        how="left"
                    )

                    color = "latest_risk"
                    size = "latest_risk"
                    color_scale = "YlOrRd"

            elif layer == "Local Fire proxy":
                color = "fire_intensity"
                size = "fire_intensity"
                color_scale = "YlOrRd"

            elif layer == "Local Heat proxy":
                color = "heat_intensity"
                size = "heat_intensity"
                color_scale = "Hot"

            elif layer == "Local Smoke proxy":
                color = "smoke_intensity"
                size = "smoke_intensity"
                color_scale = "Greys"

            elif layer == "Local Haze proxy":
                color = "haze_intensity"
                size = "haze_intensity"
                color_scale = "PuBu"

            elif layer == "Local Flood proxy":
                color = "flood_intensity"
                size = "flood_intensity"
                color_scale = "Blues"

            elif layer == "Local CO₂ proxy":
                color = "co2_intensity"
                size = "co2_intensity"
                color_scale = "Greens"

            if color and color not in base_map.columns:
                st.info("Selected layer is not available in local data.")

            else:
                map_df = base_map.dropna(subset=["lat", "lon"])
                map_df = limit_map_points(map_df)

                fig = scatter_geo_safe(
                    map_df,
                    lat="lat",
                    lon="lon",
                    color=color,
                    size=size,
                    hover_name="region",
                    projection=projection,
                    color_continuous_scale=color_scale,
                    height=600,
                )

                fig = make_geo_fig(fig)
                st.plotly_chart(fig, width="stretch")

                st.dataframe(map_df, width="stretch")


# ============================================================
# TAB: PREDICTION
# ============================================================

with tab_prediction:
    st.markdown("## 🔮 Prediction")

    st.markdown("### Prediction Status")

    risk_exists = os.path.exists(os.path.join(OUTPUT_DIR, "future_risk.csv"))
    trend_exists = os.path.exists(os.path.join(OUTPUT_DIR, "trend_report.csv"))
    model_exists = os.path.exists(os.path.join(OUTPUT_DIR, "risk_model.pkl"))
    scaler_exists = os.path.exists(os.path.join(OUTPUT_DIR, "scaler.pkl"))

    st.write(f"future_risk.csv: {'✅' if risk_exists else '❌'}")
    st.write(f"trend_report.csv: {'✅' if trend_exists else '❌'}")
    st.write(f"risk_model.pkl: {'✅' if model_exists else '❌'}")
    st.write(f"scaler.pkl: {'✅' if scaler_exists else '❌'}")

    if risk is None:
        st.warning("No prediction found. Run the prediction script from the sidebar.")

    else:
        risk_col = None

        preferred = [
            "risk_next_horizon",
            "risk",
            "probability",
            "predicted_risk",
            "hazard_risk",
        ]

        for c in preferred:
            if c in risk.columns:
                risk_col = c
                break

        if risk_col is None:
            numeric_cols = risk.select_dtypes(include=[np.number]).columns.tolist()

            if numeric_cols:
                risk_col = numeric_cols[0]

        if risk_col is None:
            st.warning("No numeric risk column found.")

        else:
            st.markdown("### Saved Model Risk Scores")
            st.caption("This is output from the local prediction file, not a live NASA observation or a verified disaster probability.")

            risk_values = pd.to_numeric(risk[risk_col], errors="coerce").dropna()
            if not risk_values.empty and ((risk_values <= 1e-5).any() or (risk_values >= 1 - 1e-5).any()):
                st.warning("Some model scores are almost exactly 0 or 1. Treat these as uncalibrated model outputs, not certainty that an event will or will not happen.")

            if "date" in risk.columns:
                risk_plot = risk.copy()
                risk_plot["date"] = pd.to_datetime(risk_plot["date"], errors="coerce")
                valid_dates = risk_plot["date"].dropna()
                if not valid_dates.empty and valid_dates.max().date() < datetime.date.today():
                    st.warning(
                        f"This saved output ends on {valid_dates.max().date()}; it is historical and is not a current forecast."
                    )
                if not valid_dates.empty:
                    fig = px.line(
                        risk_plot.dropna(subset=["date", risk_col]),
                        x="date",
                        y=risk_col,
                        color="region" if "region" in risk_plot.columns else None,
                        height=480,
                        title="Saved model scores by date",
                    )
                    fig = make_dark_fig(fig)
                    st.plotly_chart(fig, width="stretch")

            st.markdown("### Most Recent Saved Model Score")

            if risk_latest is not None:
                st.dataframe(
                    risk_latest.sort_values("latest_risk", ascending=False),
                    width="stretch"
                )

            st.markdown("### Full Risk Table")
            st.dataframe(risk, width="stretch")

    st.markdown("### Trend Report")

    if trends is None:
        st.info("No trend_report.csv found.")

    else:
        if "significant" in trends.columns:
            significant_trends = trends[trends["significant"] == True]
            st.dataframe(significant_trends, width="stretch")
        else:
            st.dataframe(trends, width="stretch")


# ============================================================
# TAB: LOCAL HAZARDS
# ============================================================

with tab_local_hazards:
    st.markdown("## 🛰️ Local Signal Proxies")
    st.info(
        "These charts summarize your local files and derived indicators. "
        "They are not live atmospheric gas measurements. Use Air & Atmosphere "
        "for NASA satellite observation layers."
    )

    if raw_data is None:
        st.warning("No local data loaded.")

    else:
        hazard_options = {
            "Fire": ("fire_intensity", "YlOrRd"),
            "Heat": ("heat_intensity", "Hot"),
            "Smoke": ("smoke_intensity", "Greys"),
            "Haze": ("haze_intensity", "PuBu"),
            "Flood": ("flood_intensity", "Blues"),
            "CO2": ("co2_intensity", "Greens"),
        }

        selected_hazard = st.selectbox(
            "Select hazard",
            list(hazard_options.keys())
        )

        col, color_scale = hazard_options[selected_hazard]

        if col not in raw_data.columns:
            st.warning(f"{selected_hazard} signal not available.")

        else:
            st.markdown(f"### {selected_hazard} Timeline")
            timeline_source = raw_data
            if selected_hazard == "Fire":
                timeline_source = raw_data.dropna(subset=[col])
                st.caption("Shows only rows with direct fire/thermal measurements; weather and vegetation are not treated as fire detections.")

            if timeline_source.empty:
                st.info(f"No direct {selected_hazard.lower()} measurements are available in the loaded local data.")
            else:
                timeline = (
                    timeline_source.groupby(pd.Grouper(key="date", freq="D"))[col]
                    .mean()
                    .dropna()
                    .reset_index()
                )
                if timeline.empty:
                    st.info(f"No dated {selected_hazard.lower()} measurements are available to chart.")
                else:
                    fig = px.line(timeline, x="date", y=col, height=380)
                    fig = make_dark_fig(fig)
                    st.plotly_chart(fig, width="stretch")

            st.markdown(f"### {selected_hazard} Locations")

            if point_summary is not None and col in point_summary.columns:
                top_points = point_summary.sort_values(col, ascending=False).head(100)

                st.dataframe(top_points, width="stretch")

                map_df = top_points.dropna(subset=["lat", "lon"])

                if not map_df.empty:
                    map_style = st.radio(
                        "Map style",
                        ["Globe", "Flat world"],
                        horizontal=True,
                        key=f"local_{col}_map_style"
                    )

                    projection = "orthographic" if map_style == "Globe" else "natural earth"

                    fig = scatter_geo_safe(
                        map_df,
                        lat="lat",
                        lon="lon",
                        color=col,
                        size=col,
                        hover_name="region",
                        projection=projection,
                        color_continuous_scale=color_scale,
                        height=560,
                    )

                    fig = make_geo_fig(fig)
                    st.plotly_chart(fig, width="stretch")


# ============================================================
# TAB: AIR & ATMOSPHERE
# ============================================================

with tab_atmosphere:
    st.markdown("## 🌫️ Air & Atmosphere")
    st.markdown(
        "Explore NASA satellite products for atmospheric composition and "
        "aerosols. These are satellite column or aerosol observations, not "
        "street-level air-quality readings."
    )

    atmosphere_products = [
        {
            "name": "Carbon monoxide (CO)",
            "instrument": "MOPITT · daily total column",
            "layer": "MOPITT_CO_Daily_Total_Column_Day",
            "description": "Maps CO through the atmospheric column. Wildfires and transport can create broad plumes.",
            "period": "Daily satellite product",
        },
        {
            "name": "Carbon dioxide (CO₂)",
            "instrument": "OCO-2 · total-column average",
            "layer": "OCO-2_Carbon_Dioxide_Total_Column_Average",
            "description": "Satellite XCO₂ observations show regional patterns. Coverage is sparse between satellite overpasses.",
            "period": "Satellite observations; revisit and cloud gaps apply",
        },
        {
            "name": "Aerosols / haze",
            "instrument": "OMI · aerosol optical depth",
            "layer": "OMI_Aerosol_Optical_Depth",
            "description": "Aerosol optical depth is a satellite haze proxy; it is not a direct PM₂.₅ or visibility measurement.",
            "period": "Satellite retrieval; clouds and surface conditions affect coverage",
        },
        {
            "name": "Nitrogen dioxide (NO₂)",
            "instrument": "Aura / OMI · tropospheric column",
            "layer": "OMI_Nitrogen_Dioxide_Tropo_Column",
            "description": "Tracks a major air-pollution gas associated with combustion and industrial activity.",
            "period": "Daily satellite retrieval; clouds can limit coverage",
        },
        {
            "name": "Sulfur dioxide (SO₂)",
            "instrument": "Aura / OMI · lower troposphere",
            "layer": "OMI_SO2_Lower_Troposphere",
            "description": "Highlights SO₂ plumes from volcanoes and combustion sources where satellite retrievals are available.",
            "period": "Satellite retrieval; revisit and cloud gaps apply",
        },
    ]

    cards = st.columns(3)
    for card, product in zip(cards, atmosphere_products):
        with card:
            st.markdown(f"### {product['name']}")
            st.caption(product["instrument"])
            st.write(product["description"])
            st.caption(product["period"])
            worldview_url = (
                "https://worldview.earthdata.nasa.gov/?l="
                f"{product['layer']},Coastlines,Reference_Features,Reference_Labels"
            )
            st.link_button("Explore in NASA Worldview", worldview_url, width="stretch")

    st.markdown("### Interpreting these layers")
    st.markdown(
        "NASA Worldview provides the interactive, time-selectable satellite "
        "maps above. Product timing and coverage differ: CO is a daily product, "
        "CO₂ is a sparse satellite retrieval, and aerosol optical depth is an "
        "indicator of airborne particles rather than a ground monitor."
    )


# ============================================================
# TAB: LIVE EVENTS
# ============================================================

with tab_live_events:
    st.markdown("## 🚨 Live NASA EONET Events")

    if live_events is None:
        show_eonet_connection_help(live_error)
    elif live_error:
        st.warning(f"⚠️ {live_error}")
    elif live_events.empty:
        st.info("No live events found.")

    else:
        hazards = sorted(live_events["hazard"].unique().tolist())

        selected_hazards = st.multiselect(
            "Filter event types",
            hazards,
            default=hazards,
            key="live_event_filter"
        )

        table_df = live_events[live_events["hazard"].isin(selected_hazards)].copy()
        table_df = table_df.sort_values("date", ascending=False)

        st.dataframe(table_df, width="stretch")

        st.markdown("### Event Links")

        for _, row in table_df.head(30).iterrows():
            st.markdown(
                f"- [{row['title']}]({row['link']}) — "
                f"{row['hazard']} — {row['date']}"
            )


# ============================================================
# TAB: LIVE FIRE
# ============================================================

with tab_live_fire:
    st.markdown("## 🔥 Live Fire")

    st.markdown("### EONET Fire Events")

    if live_events is None:
        show_eonet_connection_help(live_error)
    elif live_error:
        st.warning(f"⚠️ {live_error}")
    elif live_events.empty:
        st.info("No live events found.")

    else:
        fire_events = live_events[live_events["hazard"] == "Fire"].copy()

        if fire_events.empty:
            st.info("No live fire events detected by EONET.")
        else:
            st.dataframe(fire_events, width="stretch")
            mapped_fire_events = fire_events.dropna(subset=["lat", "lon"])
            if mapped_fire_events.empty:
                st.info("EONET has no coordinates for these fire events yet; the event records remain available above.")
            else:
                map_style = st.radio(
                    "Fire map style",
                    ["Globe", "Flat world"],
                    horizontal=True,
                    key="live_fire_map_style"
                )

                projection = "orthographic" if map_style == "Globe" else "natural earth"
                fig = px.scatter_geo(
                    mapped_fire_events,
                    lat="lat",
                    lon="lon",
                    color="hazard",
                    hover_name="title",
                    color_discrete_map=HAZARD_COLORS,
                    projection=projection,
                    height=560,
                )
                fig = make_geo_fig(fig)
                st.plotly_chart(fig, width="stretch")

    st.markdown("### FIRMS Fire Feed")

    if firms_df is None:
        st.info(firms_error)
    else:
        if firms_is_live:
            st.success(f"Loaded {len(firms_df)} live fire detections from NASA FIRMS.")
        else:
            st.warning(firms_error)
            st.caption(f"Showing {len(firms_df):,} saved detections; this snapshot is not live.")

        if "date" in firms_df.columns:
            timeline = (
                firms_df.groupby(pd.Grouper(key="date", freq="D"))
                .size()
                .reset_index(name="fire_count")
            )

            fig = px.line(
                timeline,
                x="date",
                y="fire_count",
                height=380
            )

            fig = make_dark_fig(fig)
            st.plotly_chart(fig, width="stretch")

        map_style = st.radio(
            "FIRMS map style",
            ["Globe", "Flat world"],
            horizontal=True,
            key="firms_map_style"
        )

        projection = "orthographic" if map_style == "Globe" else "natural earth"
        fig = make_thermal_fire_map(firms_df, projection=projection, height=620)
        st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})
        st.caption("NASA FIRMS thermal anomaly detections, filtered to nominal/high confidence by default. FRP affects hotspot size; detections are not all confirmed wildfires.")

        st.dataframe(firms_df.head(1000), width="stretch")


# ============================================================
# TAB: DATA TABLES
# ============================================================

with tab_tables:
    st.markdown("## 📋 Data Tables")
    st.caption(f"Table size: {table_size}. Percentage choices use the first 1,000 rows; All shows the complete loaded table.")

    def table_preview(frame):
        if table_size == "All":
            return frame
        limits = {"100% Size": 1000, "50% Size": 500, "25% Size": 250}
        return frame.head(limits[table_size])

    st.markdown("### NASA EONET events")
    if live_events is None:
        show_eonet_connection_help(live_error)
    elif live_error:
        st.warning(live_error)
    elif live_events.empty:
        st.info("NASA EONET returned no open events for this request.")
    else:
        visible_events = table_preview(live_events)
        st.caption(f"Showing {len(visible_events):,} of {len(live_events):,} NASA event records")
        st.dataframe(visible_events, width="stretch", height=420)

    st.markdown("### NASA FIRMS thermal detections")
    if firms_df is None:
        st.info(firms_error)
    else:
        source_label = "live NASA FIRMS feed" if firms_is_live else "saved snapshot (not live)"
        visible_firms = table_preview(firms_df)
        st.caption(f"Showing {len(visible_firms):,} of {len(firms_df):,} rows · {source_label}")
        st.dataframe(visible_firms, width="stretch", height=520)

    st.markdown("### Local project data")
    if raw_data is None or raw_data.empty:
        st.info("No local data rows are loaded.")
    else:
        visible_local = table_preview(raw_data)
        st.caption(f"Showing {len(visible_local):,} of {len(raw_data):,} local rows")
        st.dataframe(visible_local, width="stretch", height=520)


# ============================================================
# TAB: COORDINATES
# ============================================================

with tab_coordinates:
    st.markdown("## 🧭 Coordinate Manager")

    st.markdown(
        """
        Edit coordinates for regions here.

        When you press save, this writes:

        `data/coordinates.csv`

        The dashboard will automatically use these coordinates for regions
        that do not already have latitude/longitude values.
        """
    )

    if registry.empty:
        st.info("No regions detected yet.")

    else:
        coord_df = registry[["region", "lat", "lon"]].copy()

        edited_coords = st.data_editor(
            coord_df,
            num_rows="dynamic",
            width="stretch",
            key="coordinate_editor"
        )

        if st.button("💾 Save coordinates.csv"):
            edited_coords = edited_coords.copy()

            if "lat" not in edited_coords.columns:
                edited_coords["lat"] = np.nan

            if "lon" not in edited_coords.columns:
                edited_coords["lon"] = np.nan

            edited_coords["lat"] = pd.to_numeric(
                edited_coords["lat"],
                errors="coerce"
            )

            edited_coords["lon"] = pd.to_numeric(
                edited_coords["lon"],
                errors="coerce"
            )

            edited_coords = edited_coords.dropna(subset=["region"])

            os.makedirs(DATA_DIR, exist_ok=True)
            edited_coords.to_csv(COORD_PATH, index=False)

            st.cache_data.clear()
            st.success("Saved coordinates.csv")
            st.rerun()

        st.markdown("### Missing Coordinates")

        missing = registry[
            registry["lat"].isna() | registry["lon"].isna()
        ]

        if missing.empty:
            st.success("All detected regions have coordinates.")
        else:
            st.dataframe(missing, width="stretch")


# ============================================================
# TAB: SOURCES
# ============================================================

with tab_sources:
    st.markdown("## 📚 Sources")

    st.markdown(
        """
        ### Live / Near Real-Time

        - NASA EONET: https://eonet.gsfc.nasa.gov/
        - NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/
        - NASA Worldview / GIBS satellite layers: https://worldview.earthdata.nasa.gov/
        - NASA Earthdata: https://search.earthdata.nasa.gov/

        ### Atmospheric Products

        - CO: MOPITT daily total-column observations
        - CO₂: OCO-2 total-column average observations
        - Aerosols / haze: OMI aerosol optical depth
        - NO₂: OMI tropospheric column
        - SO₂: OMI lower-troposphere product

        ### Prediction / Historical

        - Local `.csv` / `.cvv` files inside `data/`
        - `output/future_risk.csv`
        - `output/trend_report.csv`
        - `output/risk_model.pkl`
        - `output/scaler.pkl`

        ### Hazards Tracked

        - Fire
        - Heat
        - Smoke
        - Haze
        - Flood
        - CO₂
        - Storms
        - Volcanoes
        - Other NASA EONET natural events

        ### Important Notes

        - Satellite data is rarely instant. Fire data can be delayed by minutes to hours.
        - CO₂ data is usually not truly real-time.
        - EONET tracks reported natural events, not every raw satellite detection.
        - FIRMS provides active fire detections; configure `NASA_FIRMS_MAP_KEY` or enter the key in the password field.
        - Local hazard layers are derived proxies from local input files, not live gas readings.
        """
    )


# ============================================================
# Automatic refresh is browser-driven by streamlit-autorefresh above. The data
# fetch functions have short cache lifetimes so refreshes recheck NASA safely.
