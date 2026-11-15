import os
import glob
import json
import sys
import subprocess
import time
import datetime
import smtplib
import importlib.util
from email.message import EmailMessage
import json
from pathlib import Path
from io import StringIO
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

ROOT_DIR = Path(__file__).resolve().parent
HOME_ART_PATH = ROOT_DIR / "assets" / "earthguard-home.png"
DATA_DIR = str(ROOT_DIR / "data")
OUTPUT_DIR = str(ROOT_DIR / "output")
COORD_PATH = str(ROOT_DIR / "data" / "coordinates.csv")
# Natural Earth 1:110m Admin-0 country polygons (public domain):
# https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
COUNTRY_BOUNDARIES_PATH = ROOT_DIR / "data" / "countries.geojson"

PREDICTOR_SCRIPT_CANDIDATES = [
    "pipeline.py",
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


@st.cache_data(show_spinner=False)
def load_country_boundaries():
    """Load bundled Natural Earth country polygons for local point matching."""
    try:
        with COUNTRY_BOUNDARIES_PATH.open("r", encoding="utf-8") as boundary_file:
            geojson = json.load(boundary_file)
        return [
            {
                "name": feature.get("properties", {}).get("ADMIN"),
                "bbox": feature.get("bbox"),
                "geometry": feature.get("geometry", {}),
            }
            for feature in geojson.get("features", [])
            if feature.get("properties", {}).get("ADMIN") and feature.get("geometry")
        ]
    except (OSError, ValueError, TypeError):
        return []


def points_in_ring(longitudes, latitudes, ring):
    """Vectorized ray-casting test for points in a polygon ring."""
    coordinates = np.asarray(ring, dtype=float)
    inside = np.zeros(len(longitudes), dtype=bool)
    if len(coordinates) < 3:
        return inside
    x1, y1 = coordinates[:-1, 0], coordinates[:-1, 1]
    x2, y2 = coordinates[1:, 0], coordinates[1:, 1]
    for edge in range(len(x1)):
        crosses = (y1[edge] > latitudes) != (y2[edge] > latitudes)
        if y1[edge] != y2[edge]:
            crossing_lon = (x2[edge] - x1[edge]) * (latitudes - y1[edge]) / (y2[edge] - y1[edge]) + x1[edge]
            inside ^= crosses & (longitudes < crossing_lon)
    return inside


def mask_points_in_country(data_frame, country_name, boundaries):
    """Match FIRMS coordinates to a selected Natural Earth country polygon."""
    feature = next((item for item in boundaries if item["name"] == country_name), None)
    if data_frame is None:
        return pd.Series(dtype=bool)
    if feature is None or data_frame.empty:
        return pd.Series(False, index=data_frame.index)

    longitudes = pd.to_numeric(data_frame["lon"], errors="coerce").to_numpy(dtype=float)
    latitudes = pd.to_numeric(data_frame["lat"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(longitudes) & np.isfinite(latitudes)
    bbox = feature.get("bbox")
    if bbox and len(bbox) >= 4:
        min_lon, min_lat, max_lon, max_lat = bbox[:4]
        valid &= (longitudes >= min_lon) & (longitudes <= max_lon)
        valid &= (latitudes >= min_lat) & (latitudes <= max_lat)

    result = np.zeros(len(data_frame), dtype=bool)
    candidate_indices = np.flatnonzero(valid)
    if candidate_indices.size:
        candidate_lon = longitudes[candidate_indices]
        candidate_lat = latitudes[candidate_indices]
        geometry = feature["geometry"]
        polygons = [geometry.get("coordinates", [])] if geometry.get("type") == "Polygon" else geometry.get("coordinates", [])
        for polygon in polygons:
            if not polygon:
                continue
            in_polygon = points_in_ring(candidate_lon, candidate_lat, polygon[0])
            for hole in polygon[1:]:
                in_polygon &= ~points_in_ring(candidate_lon, candidate_lat, hole)
            result[candidate_indices] |= in_polygon
    return pd.Series(result, index=data_frame.index)


def filter_points_to_country_scope(data_frame, country_names, specific_scope, boundaries):
    """Keep only point records inside one of the chosen countries when scoped."""
    if data_frame is None:
        return data_frame
    if not specific_scope:
        return data_frame.copy()
    if data_frame.empty or not country_names:
        return data_frame.iloc[0:0].copy()
    matches = pd.Series(False, index=data_frame.index)
    for country_name in country_names:
        matches |= mask_points_in_country(data_frame, country_name, boundaries)
    return data_frame.loc[matches].copy()


def filter_points_near_location(data_frame, latitude, longitude, radius_km=100):
    """Return detections within a geodesic radius of a searched city."""
    if data_frame is None or data_frame.empty or not {"lat", "lon"}.issubset(data_frame.columns):
        return data_frame.iloc[0:0].copy() if data_frame is not None else data_frame
    latitudes = pd.to_numeric(data_frame["lat"], errors="coerce")
    longitudes = pd.to_numeric(data_frame["lon"], errors="coerce")
    lat1 = np.radians(float(latitude))
    lat2 = np.radians(latitudes)
    delta_lat = lat2 - lat1
    delta_lon = np.radians(longitudes - float(longitude))
    haversine = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    distance_km = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    return data_frame.loc[distance_km <= float(radius_km)].copy()


def selected_country_focus_bounds(country_names, boundaries):
    """Return padded Leaflet bounds for the selected countries, if any."""
    features = [feature for feature in boundaries if feature["name"] in country_names]
    bboxes = [feature.get("bbox") for feature in features if feature.get("bbox") and len(feature["bbox"]) >= 4]
    if not bboxes:
        return None
    min_lon = min(bbox[0] for bbox in bboxes)
    min_lat = min(bbox[1] for bbox in bboxes)
    max_lon = max(bbox[2] for bbox in bboxes)
    max_lat = max(bbox[3] for bbox in bboxes)
    if max_lon - min_lon > 300:
        longitudes = []

        def collect_longitudes(value):
            if not isinstance(value, list) or not value:
                return
            if len(value) >= 2 and isinstance(value[0], (int, float)):
                longitudes.append(float(value[0]))
                return
            for child in value:
                collect_longitudes(child)

        for feature in features:
            collect_longitudes(feature["geometry"].get("coordinates", []))
        if longitudes:
            normalized = sorted({longitude % 360 for longitude in longitudes})
            if len(normalized) > 1:
                circular = normalized + [normalized[0] + 360]
                gap_index = max(range(len(normalized)), key=lambda index: circular[index + 1] - circular[index])
                west = circular[gap_index + 1] % 360
                east = circular[gap_index]
                if east < west:
                    east += 360
                shift = round(((west + east) / 2) / 360) * 360
                min_lon, max_lon = west - shift, east - shift
    lon_padding = max(0.25, (max_lon - min_lon) * 0.05)
    lat_padding = max(0.25, (max_lat - min_lat) * 0.05)
    return [min_lon - lon_padding, min_lat - lat_padding, max_lon + lon_padding, max_lat + lat_padding]


def country_labels_for_points(data_frame, boundaries):
    """Return a readable country/place label for each coordinate pair."""
    if data_frame is None:
        return pd.Series(dtype=str)
    labels = pd.Series("Unknown area", index=data_frame.index, dtype="object")
    if data_frame.empty or not {"lat", "lon"}.issubset(data_frame.columns):
        return labels

    longitudes = pd.to_numeric(data_frame["lon"], errors="coerce").to_numpy(dtype=float)
    latitudes = pd.to_numeric(data_frame["lat"], errors="coerce").to_numpy(dtype=float)
    unassigned = np.ones(len(data_frame), dtype=bool)
    for feature in boundaries:
        if not unassigned.any():
            break
        bbox = feature.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        min_lon, min_lat, max_lon, max_lat = bbox[:4]
        candidates = unassigned & np.isfinite(longitudes) & np.isfinite(latitudes)
        candidates &= (longitudes >= min_lon) & (longitudes <= max_lon)
        candidates &= (latitudes >= min_lat) & (latitudes <= max_lat)
        candidate_indices = np.flatnonzero(candidates)
        if not candidate_indices.size:
            continue

        geometry = feature["geometry"]
        polygons = [geometry.get("coordinates", [])] if geometry.get("type") == "Polygon" else geometry.get("coordinates", [])
        matched = np.zeros(candidate_indices.size, dtype=bool)
        for polygon in polygons:
            if not polygon:
                continue
            in_polygon = points_in_ring(longitudes[candidate_indices], latitudes[candidate_indices], polygon[0])
            for hole in polygon[1:]:
                in_polygon &= ~points_in_ring(longitudes[candidate_indices], latitudes[candidate_indices], hole)
            matched |= in_polygon
        matched_indices = candidate_indices[matched]
        labels.iloc[matched_indices] = feature["name"]
        unassigned[matched_indices] = False
    return labels


def fire_detection_ids(data_frame):
    """Create stable IDs so repeated refreshes do not repeat old alerts."""
    if data_frame is None or data_frame.empty:
        return pd.Series(dtype=str)
    identity_columns = [column for column in ("date", "acq_time", "satellite", "instrument", "lat", "lon") if column in data_frame.columns]
    if not identity_columns:
        identity_columns = [column for column in ("lat", "lon", "frp", "brightness") if column in data_frame.columns]
    identity_frame = data_frame[identity_columns].astype("string").fillna("")
    return pd.util.hash_pandas_object(identity_frame, index=False)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Earth Emergency Command",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
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

.st-key-home_hero {
    padding: 18px 28px;
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 0 70px rgba(0, 191, 255, 0.14);
    margin-bottom: 18px;
    backdrop-filter: blur(10px);
}

.st-key-home_hero [data-testid="stHorizontalBlock"] {
    align-items: center;
}

.st-key-home_hero h1 {
    margin: 0;
    font-size: 42px;
    font-weight: 900;
    background: linear-gradient(90deg, #7dd3fc, #4ade80, #facc15, #fb7185);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.st-key-home_hero p {
    color: #cbe8ff;
    font-size: 18px;
    margin-bottom: 0;
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

[data-testid="stSidebar"] {
    background: rgba(2, 8, 23, 0.92);
}

/* Keep the secret-entry box readable even when the user's Streamlit theme is dark. */
[data-testid="stSidebar"] input[type="password"] {
    color: #111827 !important;
    background-color: #ffffff !important;
    caret-color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}
[data-testid="stSidebar"] input[type="password"]::placeholder {
    color: #6b7280 !important;
    -webkit-text-fill-color: #6b7280 !important;
}

/* Keep form labels legible against the app's dark background. */
label,
label *,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] *,
[data-testid="stAppViewContainer"] [data-testid="stSelectbox"] label,
[data-testid="stAppViewContainer"] [data-testid="stSelectbox"] label *,
[data-testid="stAppViewContainer"] [data-testid="stTextInput"] label,
[data-testid="stAppViewContainer"] [data-testid="stTextInput"] label *,
[data-testid="stAppViewContainer"] [data-testid="stNumberInput"] label,
[data-testid="stAppViewContainer"] [data-testid="stNumberInput"] label *,
[data-testid="stAppViewContainer"] [data-testid="stDateInput"] label,
[data-testid="stAppViewContainer"] [data-testid="stDateInput"] label *,
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li,
[data-testid="stAppViewContainer"] [data-testid="stMetricLabel"],
[data-testid="stAppViewContainer"] [data-testid="stMetricLabel"] * {
    color: #f3f7fb !important;
    -webkit-text-fill-color: #f3f7fb !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    overflow-x: auto;
    flex-wrap: nowrap;
    scrollbar-width: thin;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding-left: 16px;
    padding-right: 16px;
    white-space: nowrap;
}

.st-key-floating_login {
    position: fixed !important;
    right: 1.25rem !important;
    bottom: 4.5rem !important;
    z-index: 9999 !important;
    width: max-content !important;
    max-width: calc(100vw - 2rem) !important;
    padding: 0.35rem !important;
    border: 1px solid rgba(130, 196, 255, 0.35) !important;
    border-radius: 999px !important;
    background: rgba(10, 22, 40, 0.94) !important;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35) !important;
}
.st-key-floating_login button {
    width: auto !important;
    border-radius: 999px !important;
}

@media (max-width: 700px) {
    [data-testid="stAppViewBlockContainer"] {
        padding: 1rem 0.75rem 2rem;
    }
    .st-key-home_hero {
        padding: 14px;
        border-radius: 18px;
    }
    .st-key-home_hero h1 { font-size: 28px; }
    .st-key-home_hero p { font-size: 15px; }
    h1 { font-size: 1.65rem !important; }
    h2 { font-size: 1.35rem !important; }
    h3 { font-size: 1.1rem !important; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 10px;
        font-size: 0.8rem;
    }
    .st-key-floating_login {
        right: 0.65rem !important;
        bottom: 4.25rem !important;
    }
    [data-testid="stPlotlyChart"] {
        width: 100% !important;
        overflow-x: hidden;
    }
    [data-testid="stDataFrame"] { max-width: 100%; }
    [data-testid="stButton"] button,
    [data-testid="stLinkButton"] a { width: 100%; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def oidc_login_configured():
    """Check whether Streamlit's default OIDC provider is configured."""
    try:
        auth_settings = st.secrets.get("auth", {})
        return all(auth_settings.get(field) for field in (
            "client_id", "client_secret", "server_metadata_url", "redirect_uri", "cookie_secret"
        ))
    except Exception:
        return False


def current_user_logged_in():
    """Return False on versions or deployments without identity configuration."""
    try:
        return bool(getattr(st.user, "is_logged_in", False))
    except Exception:
        return False


def current_phone_user():
    """Return the phone number verified for this active Streamlit session."""
    return st.session_state.get("phone_signin_verified_number", "")


def account_is_logged_in():
    return current_user_logged_in() or bool(current_phone_user())


@st.dialog("Your Earth Emergency Command account")
def show_login_dialog():
    if current_user_logged_in():
        user_name = getattr(st.user, "name", None) or getattr(st.user, "email", None) or "Signed-in user"
        st.success(f"You’re signed in as {user_name}.")
        if st.button("Log out", key="login_dialog_logout", width="stretch"):
            st.logout()
        return
    if current_phone_user():
        st.success(f"Phone verified for this session: {current_phone_user()}")
        if st.button("Log out", key="phone_login_dialog_logout", width="stretch"):
            st.session_state.pop("phone_signin_verified_number", None)
            st.rerun()
        return

    if oidc_login_configured() and hasattr(st, "login"):
        st.write("Sign in with Google or verify your phone number.")
        if st.button("Continue with Google", type="primary", key="login_dialog_continue", width="stretch"):
            st.login()
    else:
        with st.expander("Optional Google sign-in setup"):
            st.caption("Google sign-in needs an OpenID Connect provider in Streamlit secrets.")
            st.code('[auth]\nredirect_uri = "https://testnasa123.streamlit.app/oauth2callback"\ncookie_secret = "<long random secret>"\nclient_id = "<provider client ID>"\nclient_secret = "<provider client secret>"\nserver_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"', language="toml")
            st.caption("For local use, change redirect_uri to http://localhost:8501/oauth2callback.")

    st.markdown("#### Sign in with your phone")
    st.caption("We’ll text a one-time code. Your verified sign-in lasts for this browser session.")
    with st.expander("Phone sign-in setup"):
        st.caption("The app owner must create a Twilio Verify Service and add these private values in Streamlit Secrets:")
        st.code('[twilio]\nTWILIO_ACCOUNT_SID = "AC..."\nTWILIO_AUTH_TOKEN = "..."\nTWILIO_VERIFY_SERVICE_SID = "VA..."', language="toml")
    phone_value = st.text_input(
        "Phone number with country code",
        placeholder="+1 555 123 4567",
        key="phone_signin_number",
    )
    normalized_phone = normalize_phone_number(phone_value)
    if st.button("Text me a verification code", key="phone_signin_send_code"):
        if not normalized_phone:
            st.warning("Enter a valid number in international format, starting with + and the country code.")
        else:
            sent, message = start_phone_verification(normalized_phone)
            if sent:
                st.session_state["phone_signin_pending_number"] = normalized_phone
                st.success("Verification code sent. It expires after 10 minutes.")
            else:
                st.error(message)

    pending_phone = st.session_state.get("phone_signin_pending_number", "")
    if pending_phone:
        st.caption(f"Enter the code sent to {pending_phone}.")
        verification_code = st.text_input("Verification code", key="phone_signin_code", max_chars=10)
        if st.button("Verify and sign in", type="primary", key="phone_signin_verify"):
            if not verification_code.strip():
                st.warning("Enter the verification code from your text message.")
            else:
                verified, message = check_phone_verification(pending_phone, verification_code.strip())
                if verified:
                    st.session_state["phone_signin_verified_number"] = pending_phone
                    st.session_state.pop("phone_signin_pending_number", None)
                    st.success("Phone verified. You’re signed in for this session.")
                else:
                    st.error(message)


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


def sample_map_points(data_frame, max_points=2000, rank_by=None):
    """Use a deterministic high-signal plus world-wide sample for every map style."""
    if data_frame is None or len(data_frame) <= max_points:
        return data_frame.copy() if data_frame is not None else data_frame
    if rank_by and rank_by in data_frame.columns:
        ranked = data_frame.assign(__sample_rank=pd.to_numeric(data_frame[rank_by], errors="coerce").fillna(0))
        strongest = ranked.nlargest(max_points // 2, "__sample_rank").drop(columns="__sample_rank")
    else:
        strongest = data_frame.iloc[:max_points // 2]
    remainder = data_frame.drop(index=strongest.index)
    sampled = remainder.sample(n=min(max_points - len(strongest), len(remainder)), random_state=17)
    return pd.concat([strongest, sampled]).drop_duplicates().copy()


def make_thermal_surface_records(data_frame, value_column=None, pixel_size_meters=1000, max_points=6000):
    """Build a soft surface from real detections, measuring cluster support on all input rows."""
    if data_frame is None or data_frame.empty:
        return [], 0, "nearby point density"

    frame = sample_map_points(data_frame, max_points=max_points, rank_by=value_column)
    work = frame[["lat", "lon"]].copy()
    # FIRMS combines 375 m VIIRS and 1 km MODIS detections. Keep each sensor's
    # native footprint when feeds are combined instead of assigning all points
    # the first sensor's resolution.
    instrument = frame.get("instrument", frame.get("satellite", pd.Series("", index=frame.index)))
    instrument = instrument.astype(str).str.upper()
    work["__pixel_size_meters"] = np.where(instrument.str.contains("MODIS|TERRA|AQUA"), 1000.0, float(pixel_size_meters))
    numeric = (
        pd.to_numeric(frame[value_column], errors="coerce")
        if value_column and value_column in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    has_measurement = numeric.notna().any()
    work["__value"] = numeric if has_measurement else 1.0

    # Estimate neighborhood support from the FULL feed, not the display sample.
    # This avoids losing real regional clusters when a global feed has tens of
    # thousands of detections. Bins are used only for counts; points stay at
    # their satellite-reported coordinates, so no grid is drawn on the map.
    support_step = max(0.01, (float(pixel_size_meters) / 111_195.0) * 3.0)
    full_work = data_frame[["lat", "lon"]].copy()
    full_work["lat"] = pd.to_numeric(full_work["lat"], errors="coerce")
    full_work["lon"] = pd.to_numeric(full_work["lon"], errors="coerce")
    full_work = full_work.dropna(subset=["lat", "lon"])
    full_work["__lat_cell"] = np.floor(full_work["lat"] / support_step).astype("int32")
    full_work["__lon_cell"] = np.floor(full_work["lon"] / support_step).astype("int32")
    bin_counts = full_work.groupby(["__lat_cell", "__lon_cell"], sort=False).size()
    work["__lat_cell"] = np.floor(work["lat"] / support_step).astype("int32")
    work["__lon_cell"] = np.floor(work["lon"] / support_step).astype("int32")
    support = [
        sum(int(bin_counts.get((int(lat_cell) + dlat, int(lon_cell) + dlon), 0))
            for dlat in (-1, 0, 1) for dlon in (-1, 0, 1))
        for lat_cell, lon_cell in work[["__lat_cell", "__lon_cell"]].itertuples(index=False, name=None)
    ]

    values = work["__value"].fillna(0).clip(lower=0)
    if value_column == "frp" or not has_measurement:
        values = np.log1p(values)
    low = float(values.quantile(0.02))
    high = float(values.quantile(0.98))
    stops = [(34, 197, 94), (132, 204, 22), (234, 179, 8), (249, 115, 22), (239, 68, 68)]
    records = []
    for (_, row), local_support, value in zip(work.iterrows(), support, values):
        fraction = 0.5 if high <= low else min(1.0, max(0.0, (float(value) - low) / (high - low)))
        segment = min(len(stops) - 2, int(fraction * (len(stops) - 1)))
        blend = fraction * (len(stops) - 1) - segment
        color = "#" + "".join(
            f"{round(a + (b - a) * blend):02x}" for a, b in zip(stops[segment], stops[segment + 1])
        )
        records.append({
            "lat": float(row["lat"]), "lon": float(row["lon"]), "color": color,
            "strength": 0.28 + 0.48 * fraction,
            "support": int(local_support),
            "pixel_size_meters": float(row["__pixel_size_meters"]),
        })
    records.sort(key=lambda record: record["strength"])
    if value_column == "frp":
        basis = "FRP (relative)"
    elif not has_measurement:
        basis = "nearby event density"
    elif value_column == "latest_risk" or (isinstance(value_column, str) and value_column.endswith("_intensity")):
        basis = "local proxy/risk score (relative)"
    else:
        basis = "selected numeric value"
    return records, max(float(pixel_size_meters) * 0.75, 350), basis


def make_thermal_fire_map(data_frame, projection="orthographic", height=600, focus_bounds=None):
    """Render FIRMS detections as a light, single-layer map of normal markers."""
    if focus_bounds is None:
        focus_bounds = globals().get("country_focus_bounds")
    df = data_frame.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
    if df.empty:
        return scatter_geo_safe(df, projection=projection, height=height)

    total_detections = len(df)
    df = sample_map_points(df, max_points=2000, rank_by="frp")
    if globals().get("country_boundaries"):
        df["place"] = country_labels_for_points(df, country_boundaries)

    frp = pd.to_numeric(df.get("frp", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0).clip(lower=0)
    df["__frp"] = frp

    custom_columns = [c for c in ["place", "__frp", "confidence", "date", "acq_date", "acq_time", "satellite", "daynight"] if c in df.columns]
    custom_data = df[custom_columns].to_numpy() if custom_columns else None
    hover_lines = ["<b>Thermal hotspot</b>"]
    if "place" in custom_columns:
        hover_lines.append(f"Place: %{{customdata[{custom_columns.index('place')}]}}")
    if "__frp" in custom_columns:
        hover_lines.append(f"FRP: %{{customdata[{custom_columns.index('__frp')}]:.1f}} MW")
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

    fig = go.Figure(go.Scattergeo(
        lat=df["lat"],
        lon=df["lon"],
        mode="markers",
        marker=(
            dict(
                size=9,
                color=frp,
                cmin=0,
                cmax=max(float(frp.quantile(0.98)), 1.0),
                colorscale=[[0, "#ffd166"], [0.35, "#ff8c00"], [0.72, "#f04438"], [1, "#fff1b8"]],
                showscale=True,
                colorbar=dict(title="FRP<br>MW", thickness=11, len=0.55),
                line=dict(color="rgba(8, 15, 25, 0.8)", width=0.4),
            ) if float(frp.max()) > 0 else dict(
                size=9,
                color="#ff8c00",
                line=dict(color="rgba(8, 15, 25, 0.8)", width=0.4),
            )
        ),
        customdata=custom_data,
        hovertemplate=hovertemplate,
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"FIRMS hotspots · {len(df):,} of {total_detections:,} plotted"
                if len(df) < total_detections
                else f"FIRMS hotspots · {total_detections:,}"
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
            projection=dict(
                type=projection,
                scale=(min(100.0, max(1.0, 420.0 / max(focus_bounds[2] - focus_bounds[0], (focus_bounds[3] - focus_bounds[1]) * 1.5, 4.0))) if focus_bounds else 1),
            ),
            center=(
                dict(lat=(focus_bounds[1] + focus_bounds[3]) / 2, lon=(focus_bounds[0] + focus_bounds[2]) / 2)
                if focus_bounds else dict(lat=10, lon=0)
            ),
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
        uirevision="firms-fire-map",
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


def make_geo_fig(fig, focus_bounds=None):
    # Keep global hazard layers framed on the whole Earth. Plotly otherwise
    # auto-fits the view to the current event coordinates, which can make a
    # global EONET feed look like only a small part of the planet is covered.
    if focus_bounds is None:
        focus_bounds = globals().get("country_focus_bounds")
    if focus_bounds:
        min_lon, min_lat, max_lon, max_lat = focus_bounds
        span = max(max_lon - min_lon, (max_lat - min_lat) * 1.5, 5.0)
        fig.update_geos(
            fitbounds=False,
            scope="world",
            projection_scale=min(100.0, max(1.0, 420.0 / span)),
            center=dict(lon=(min_lon + max_lon) / 2, lat=(min_lat + max_lat) / 2),
        )
    else:
        fig.update_geos(fitbounds=False, scope="world", projection_scale=1)
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


def render_nasa_atmosphere_map(layer_id, observation_date, height=590, focus_bounds=None):
    """Show a date-selectable NASA GIBS atmosphere layer inside the dashboard."""
    if focus_bounds is None:
        focus_bounds = globals().get("country_focus_bounds")
    html = """<!doctype html>
<html><head><meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
html,body,#map{height:100%;width:100%;margin:0;background:#071426}
.status{position:absolute;z-index:1000;left:10px;top:10px;max-width:calc(100% - 78px);padding:7px 10px;border-radius:7px;background:#101827eF;color:#f3f6fa;font:13px/1.35 sans-serif;box-shadow:0 1px 6px #0008}
.leaflet-container{font:13px sans-serif}
@media(max-width:600px){.status{font-size:12px}.leaflet-control-zoom a{width:38px!important;height:38px!important;line-height:38px!important}}
</style></head><body><div id="map"></div><div id="status" class="status">Loading NASA satellite layer…</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const nasaLayer = "__LAYER__";
const nasaDate = "__DATE__";
const focusBounds = __FOCUS_BOUNDS__;
const map = L.map('map', {zoomControl:false, tap:true, scrollWheelZoom:false, worldCopyJump:false, minZoom:1});
if(focusBounds) map.fitBounds([[focusBounds[1],focusBounds[0]],[focusBounds[3],focusBounds[2]]], {padding:[8,8],maxZoom:10});
else map.setView([18, 0], 1.6);
L.control.zoom({position:'topright'}).addTo(map);
const baseTiles=L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  minZoom:1, maxZoom:18, noWrap:true, bounds:[[-85,-180],[85,180]],
  attribution: '&copy; OpenStreetMap contributors'
});
let baseTileErrors=0, fallbackBaseAdded=false;
baseTiles.on('tileerror',()=>{baseTileErrors+=1;if(baseTileErrors>=3&&!fallbackBaseAdded){fallbackBaseAdded=true;baseTiles.remove();L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',{minZoom:1,maxZoom:19,noWrap:true,bounds:[[-85,-180],[85,180]],attribution:'&copy; OpenStreetMap contributors &copy; CARTO'}).addTo(map)}});
baseTiles.addTo(map);
const status = document.getElementById('status');
let loaded = false, errors = 0;
let nasa;
if (nasaLayer === 'MODIS_Terra_Land_Surface_Temp_Day') {
  const thermalTiles = 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/' + nasaLayer + '/default/' + nasaDate + '/GoogleMapsCompatible_Level7/{z}/{y}/{x}.png';
  nasa = L.tileLayer(thermalTiles, {
    minZoom:1, maxZoom:18, maxNativeZoom:7, noWrap:true, opacity:0.82,
    bounds:[[-85,-180],[85,180]],
    updateWhenZooming:false, updateWhenIdle:true, keepBuffer:3,
    attribution:'NASA EOSDIS GIBS'
  }).addTo(map);
} else {
  nasa = L.tileLayer.wms('https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi', {
    layers: nasaLayer, format:'image/png', transparent:true, version:'1.1.1',
    time:nasaDate, tiled:true, noWrap:true, opacity:0.78, bounds:[[-85,-180],[85,180]],
    attribution:'NASA EOSDIS GIBS'
  }).addTo(map);
}
nasa.on('tileload', () => { loaded = true; status.textContent = 'Satellite tiles loaded · coverage varies by product, date, and clouds'; });
nasa.on('tileerror', () => { errors += 1; if (!loaded && errors >= 4) status.textContent = 'NASA returned no usable tiles for this layer/date. Try another date or product; satellite coverage can be sparse.'; });
setTimeout(() => { if (!loaded && errors === 0) status.textContent = 'Waiting for NASA GIBS imagery…'; }, 8000);
const resizeGibsMap=()=>map.invalidateSize({pan:false});requestAnimationFrame(()=>requestAnimationFrame(resizeGibsMap));setTimeout(resizeGibsMap,300);new ResizeObserver(resizeGibsMap).observe(document.getElementById('map'));
</script></body></html>"""
    html = html.replace("__LAYER__", str(layer_id)).replace("__DATE__", str(observation_date)).replace("__FOCUS_BOUNDS__", json.dumps(focus_bounds))
    st.iframe(html, height=height, alt="Interactive NASA atmospheric observation map")


def render_leaflet_points(data_frame, *, color_by=None, category_colors=None,
                          popup_columns=None, height=520, point_limit=2000, thermal=False,
                          pixel_size_meters=1000, key="map", focus_bounds=None):
    """Render touch-friendly markers and, optionally, an aggregated thermal surface."""
    if focus_bounds is None:
        focus_bounds = globals().get("country_focus_bounds")
    if data_frame is None or data_frame.empty or not {"lat", "lon"}.issubset(data_frame.columns):
        st.info("No mappable coordinates are available for this layer.")
        return

    frame = data_frame.copy()
    frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
    frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
    frame = frame.dropna(subset=["lat", "lon"])
    frame = frame[frame["lat"].between(-90, 90) & frame["lon"].between(-180, 180)]
    if frame.empty:
        st.info("No valid latitude/longitude coordinates are available for this layer.")
        return

    total = len(frame)
    surface_rows, surface_radius_meters, surface_basis = ([], 0, "")
    if thermal:
        surface_rows, surface_radius_meters, surface_basis = make_thermal_surface_records(
            frame, color_by, pixel_size_meters=pixel_size_meters
        )
    frame = sample_map_points(frame, max_points=point_limit, rank_by=color_by)
    if globals().get("country_boundaries"):
        frame["place"] = country_labels_for_points(frame, country_boundaries)

    if popup_columns is None:
        popup_columns = [c for c in ("title", "place", "hazard", "categories", "date", "region", "frp", "confidence", "satellite") if c in frame.columns]
    popup_columns = [c for c in popup_columns if c in frame.columns and c not in {"lat", "lon"}]

    records = []
    category_colors = category_colors or {}
    numeric_values = pd.to_numeric(frame[color_by], errors="coerce") if color_by and color_by in frame.columns else None
    thermal_basis = "reported intensity"
    if thermal and (numeric_values is None or not numeric_values.notna().any()):
        density_bins = pd.DataFrame({"lat_bin": np.floor(frame["lat"]), "lon_bin": np.floor(frame["lon"])}, index=frame.index)
        numeric_values = density_bins.groupby(["lat_bin", "lon_bin"])["lat_bin"].transform("size").astype(float)
        thermal_basis = "nearby event density"
    if numeric_values is not None and numeric_values.notna().any():
        lo = float(numeric_values.quantile(0.02)) if thermal else float(numeric_values.min())
        hi = float(numeric_values.quantile(0.98)) if thermal else float(numeric_values.max())
    else:
        lo, hi = 0.0, 0.0
    ramp = ["#fde047", "#fb923c", "#f04438", "#b91c1c"]
    thermal_ramp = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"]
    thermal_values = numeric_values.to_numpy() if numeric_values is not None else None
    for row_number, (_, row) in enumerate(frame.iterrows()):
        color = "#ff5a5f"
        radius = 5
        if color_by and color_by in frame.columns:
            value = row[color_by]
            if thermal:
                intensity = thermal_values[row_number]
                fraction = 0 if pd.isna(intensity) or hi <= lo else min(1.0, max(0.0, (float(intensity) - lo) / (hi - lo)))
                segment = min(len(thermal_ramp) - 2, int(fraction * (len(thermal_ramp) - 1)))
                blend = fraction * (len(thermal_ramp) - 1) - segment
                start = tuple(int(thermal_ramp[segment][i:i + 2], 16) for i in (1, 3, 5))
                end = tuple(int(thermal_ramp[segment + 1][i:i + 2], 16) for i in (1, 3, 5))
                color = "#" + "".join(f"{round(a + (b - a) * blend):02x}" for a, b in zip(start, end))
                radius = 2.5
            elif category_colors:
                color = category_colors.get(str(value), "#ff5a5f")
            elif numeric_values is not None:
                number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                fraction = 0 if pd.isna(number) or hi <= lo else min(0.999, max(0, (float(number) - lo) / (hi - lo)))
                color = ramp[int(fraction * len(ramp))]
        details = {column: ("—" if pd.isna(row[column]) else str(row[column])) for column in popup_columns}
        if "frp" in frame.columns:
            details["Interpretation"] = "Satellite thermal anomaly; wildfire not confirmed"
        records.append({
            "lat": float(row["lat"]), "lon": float(row["lon"]),
            "color": "#f8fafc" if thermal else color,
            "radius": radius,
            "opacity": 0.55 if thermal else 0.82,
            "details": details,
        })

    payload = json.dumps(records, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    surface_payload = json.dumps(surface_rows, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = f"""<!doctype html><!-- {key} --><html><head><meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>
html,body,#map{{height:100%;width:100%;margin:0;background:#071426}}.leaflet-container{{font:13px sans-serif}}
.map-note{{position:absolute;z-index:1000;left:10px;top:10px;padding:6px 9px;border-radius:7px;background:#101827eF;color:#f3f6fa;font:12px/1.3 sans-serif;box-shadow:0 1px 6px #0008}}
.thermal-scale{{display:flex;align-items:center;gap:6px;margin-top:5px;font-size:10px;color:#e5e7eb}}
.thermal-scale i{{display:block;width:94px;height:8px;border-radius:8px;background:linear-gradient(90deg,#22c55e,#84cc16,#eab308,#f97316,#ef4444)}}
.thermal-canvas{{pointer-events:none}}
@media(max-width:600px){{.leaflet-control-zoom a{{width:38px!important;height:38px!important;line-height:38px!important}}}}
</style></head><body><div id="map"></div><div class="map-note">__NOTE____LEGEND__</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const rows={payload}; const thermalCells={surface_payload}; const thermalRadiusMeters={surface_radius_meters};
const fireSurface={str(color_by in {"frp", "fire_intensity", "latest_risk"}).lower()};
const thermal={str(thermal).lower()};
const focusBounds={json.dumps(focus_bounds)};
const map=L.map('map',{{zoomControl:false,tap:true,scrollWheelZoom:true,worldCopyJump:false,minZoom:1}});
if(focusBounds)map.fitBounds([[focusBounds[1],focusBounds[0]],[focusBounds[3],focusBounds[2]]],{{padding:[8,8],maxZoom:10}});else map.setView([18,0],1.6);
L.control.zoom({{position:'topright'}}).addTo(map);
const baseTiles=L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{minZoom:1,maxZoom:19,noWrap:true,bounds:[[-85,-180],[85,180]],attribution:'&copy; OpenStreetMap contributors'}});
let baseTileErrors=0, fallbackBaseAdded=false;
baseTiles.on('tileerror',()=>{{baseTileErrors+=1;if(baseTileErrors>=3&&!fallbackBaseAdded){{fallbackBaseAdded=true;baseTiles.remove();L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png',{{minZoom:1,maxZoom:19,noWrap:true,bounds:[[-85,-180],[85,180]],attribution:'&copy; OpenStreetMap contributors &copy; CARTO'}}).addTo(map)}}}});
baseTiles.addTo(map);
if(thermalCells.length){{
 const ThermalSurface=L.Layer.extend({{
  onAdd:function(target){{this._map=target;this._canvas=L.DomUtil.create('canvas','thermal-canvas');this._canvas.style.position='absolute';target.getPanes().overlayPane.insertBefore(this._canvas,target.getPanes().overlayPane.firstChild);target.on('moveend zoomend resize',this._draw,this);this._draw()}},
  onRemove:function(target){{target.off('moveend zoomend resize',this._draw,this);L.DomUtil.remove(this._canvas)}},
  _draw:function(){{const size=this._map.getSize(),origin=this._map.containerPointToLayerPoint([0,0]),dpr=Math.min(window.devicePixelRatio||1,1.25),canvas=this._canvas;canvas.width=size.x*dpr;canvas.height=size.y*dpr;canvas.style.width=size.x+'px';canvas.style.height=size.y+'px';L.DomUtil.setPosition(canvas,origin);const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,size.x,size.y);const heatCanvas=document.createElement('canvas');heatCanvas.width=canvas.width;heatCanvas.height=canvas.height;const heatCtx=heatCanvas.getContext('2d');heatCtx.setTransform(dpr,0,0,dpr,0,0);const zoom=this._map.getZoom();
   // Draw localized supported values; the surface does not invent values between distant points.
   heatCtx.globalCompositeOperation='source-over';
  const cellRadius=cell=>{{if(fireSurface)return 14;const metersPerPixel=156543.03392*Math.max(0.08,Math.cos(cell.lat*Math.PI/180))/Math.pow(2,zoom),physicalRadius=thermalRadiusMeters/metersPerPixel,baseRadius=Math.max(12,physicalRadius*3.5);return Math.min(30,baseRadius)}};
  for(const cell of thermalCells){{const p=this._map.latLngToLayerPoint([cell.lat,cell.lon]).subtract(origin),r=cellRadius(cell);if(p.x < -r || p.y < -r || p.x > size.x+r || p.y > size.y+r)continue;const grad=heatCtx.createRadialGradient(p.x,p.y,0,p.x,p.y,r);grad.addColorStop(0,cell.color+'dd');grad.addColorStop(.42,cell.color+'75');grad.addColorStop(1,cell.color+'00');heatCtx.globalAlpha=cell.strength*(fireSurface ? 0.9 : 1);heatCtx.fillStyle=grad;heatCtx.beginPath();heatCtx.arc(p.x,p.y,r,0,Math.PI*2);heatCtx.fill()}}heatCtx.globalAlpha=1;heatCtx.globalCompositeOperation='source-over';ctx.save();ctx.setTransform(1,0,0,1,0,0);ctx.filter='blur(5px)';ctx.drawImage(heatCanvas,0,0);ctx.filter='none';ctx.restore()}}
 }});
 new ThermalSurface().addTo(map);
}}
if(!thermal)for(const item of rows){{const marker=L.circleMarker([item.lat,item.lon],{{radius:item.radius,color:item.color,fillColor:item.color,fillOpacity:item.opacity,weight:thermalCells.length ? 0.6 : 1}}).addTo(map);
const box=document.createElement('div');for(const [label,value] of Object.entries(item.details)){{const line=document.createElement('div');line.textContent=label+': '+value;box.appendChild(line)}}
if(!Object.keys(item.details).length){{const line=document.createElement('div');line.textContent=item.lat.toFixed(3)+', '+item.lon.toFixed(3);box.appendChild(line)}}marker.bindPopup(box)}}
</script></body></html>""".replace(
        "__NOTE__",
        (f"Soft heat map · green → red = {surface_basis} · no supported points = transparent · {len(surface_rows):,} mapped locations" if thermal
         else f"{len(frame):,} of {total:,} mapped · drag to pan · pinch or use +/− to zoom")
    ).replace(
        "__LEGEND__",
        '<div class="thermal-scale"><span>LOW</span><i></i><span>HIGH</span></div>' if thermal else ""
    )
    st.iframe(html, height=height, width="stretch", tab_index=0, alt="Interactive zoomable map")


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
        path = ROOT_DIR / name
        if path.is_file():
            return str(path)

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
        "historical_hazards",
        "j1_viirs_c2_global_48h",
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
    if not url.lower().startswith("https://"):
        return None, "FIRMS CSV URL must use HTTPS."
    if not HAS_REQUESTS:
        return None, "Install requests to fetch a FIRMS CSV URL."

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))

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


def show_eonet_event_browser(events, key="eonet_events"):
    """Compact, filterable EONET event list that works well on narrow screens."""
    if events is None or events.empty:
        st.info("No EONET event records to show.")
        return

    hazards = ["All event types"] + sorted(events["hazard"].dropna().astype(str).unique().tolist())
    selected_hazard = st.selectbox("Event type", hazards, key=f"{key}_hazard")
    search = st.text_input("Search event title or location", key=f"{key}_search").strip().casefold()

    filtered = events.copy()
    if selected_hazard != "All event types":
        filtered = filtered[filtered["hazard"] == selected_hazard]
    if search:
        searchable = filtered[[c for c in ["title", "categories", "description", "lat", "lon"] if c in filtered.columns]]
        mask = searchable.fillna("").astype(str).agg(" ".join, axis=1).str.casefold().str.contains(search, regex=False)
        filtered = filtered.loc[mask]

    if "date" in filtered.columns:
        filtered = filtered.sort_values("date", ascending=False, na_position="last")
    st.caption(f"{len(filtered):,} matching event records")
    if filtered.empty:
        st.info("No events match those filters.")
        return

    page_size = st.selectbox("Events per page", [10, 20, 50], index=0, key=f"{key}_page_size")
    page_count = max(1, (len(filtered) + page_size - 1) // page_size)
    page_options = [f"Page {i + 1} of {page_count}" for i in range(page_count)]
    page_label = st.selectbox("Page", page_options, key=f"{key}_page_{len(filtered)}_{page_size}")
    page_number = page_options.index(page_label)
    first = page_number * page_size
    visible = filtered.iloc[first:first + page_size]

    for _, event in visible.iterrows():
        title = str(event.get("title") or "Untitled NASA EONET event")
        with st.expander(f"{title} · {event.get('hazard', 'Other')}"):
            details = [str(event.get("categories") or "Uncategorized")]
            place = event.get("place")
            if isinstance(place, str) and place and place != "Unknown area":
                details.append(place)
            event_date = event.get("date")
            if pd.notna(event_date):
                details.append(pd.Timestamp(event_date).strftime("%d %b %Y UTC"))
            lat, lon = event.get("lat"), event.get("lon")
            if pd.notna(lat) and pd.notna(lon):
                details.append(f"{float(lat):.3f}, {float(lon):.3f}")
            st.caption(" · ".join(details))
            description = event.get("description")
            if isinstance(description, str) and description.strip():
                st.write(description)
            sources = event.get("sources")
            if isinstance(sources, str) and sources.strip():
                st.caption(f"Sources: {sources}")
            event_url = event.get("link")
            if isinstance(event_url, str) and event_url.startswith("https://"):
                st.link_button("Open NASA event record", event_url, width="stretch")


@st.cache_data(show_spinner=False, ttl=600)
def fetch_firms_api(map_key, source="VIIRS_SNPP_NRT", day_range=1):
    """Fetch recent global fire detections from NASA FIRMS area CSV API."""
    if not map_key:
        return None, "Enter a FIRMS MAP_KEY or set NASA_FIRMS_MAP_KEY."
    if not HAS_REQUESTS:
        return None, "Install requests to fetch FIRMS data."
    allowed_sources = {
        "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"
    }
    if source not in allowed_sources:
        return None, "Choose a supported NASA FIRMS near-real-time feed."
    try:
        day_range = int(day_range)
    except (TypeError, ValueError):
        return None, "FIRMS lookback must be from 1 to 5 days."
    if not 1 <= day_range <= 5:
        return None, "FIRMS lookback must be from 1 to 5 days."

    # Keep the key out of displayed errors and use the official FIRMS area API.
    url = (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{map_key}/{source}/world/{day_range}"
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


@st.cache_data(show_spinner=False, ttl=3600)
def local_firms_snapshot():
    """Load the saved FIRMS sample only when the live feed is unavailable."""
    candidates = sorted(
        glob.glob(os.path.join(DATA_DIR, "*J1_VIIRS_C2_Global_48h*.csv"))
        + glob.glob(os.path.join(DATA_DIR, "*FIRMS*.csv"))
    )
    if not candidates:
        return None
    try:
        snapshot = pd.read_csv(candidates[0])
    except Exception:
        return None
    for old, new in (("latitude", "lat"), ("longitude", "lon"), ("acq_date", "date")):
        if old in snapshot.columns and new not in snapshot.columns:
            snapshot = snapshot.rename(columns={old: new})
    if not {"lat", "lon"}.issubset(snapshot.columns):
        return None
    snapshot["lat"] = pd.to_numeric(snapshot["lat"], errors="coerce")
    snapshot["lon"] = pd.to_numeric(snapshot["lon"], errors="coerce")
    if "frp" in snapshot.columns:
        snapshot["frp"] = pd.to_numeric(snapshot["frp"], errors="coerce")
    if "date" in snapshot.columns:
        snapshot["date"] = pd.to_datetime(snapshot["date"], errors="coerce")
    return snapshot.dropna(subset=["lat", "lon"])


@st.cache_data(show_spinner=False, ttl=300)
def fetch_air_quality(lat, lon):
    """Fetch modeled AQI and pollutant forecasts for one WGS84 coordinate."""
    if not HAS_REQUESTS:
        return None, "Install requests to load air quality data."
    endpoint = "https://air-quality-api.open-meteo.com/v1/air-quality"
    current_variables = [
        "us_aqi", "european_aqi", "pm2_5", "pm10", "ozone",
        "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide",
    ]
    try:
        response = requests.get(
            endpoint,
            params={
                "latitude": round(float(lat), 4),
                "longitude": round(float(lon), 4),
                "current": ",".join(current_variables),
                "hourly": ",".join(["us_aqi", "european_aqi", "pm2_5", "pm10", "ozone"]),
                "forecast_days": 3,
                "timezone": "auto",
            },
            timeout=20,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("error"):
            return None, result.get("reason", "The air quality service returned an error.")
        if not result.get("current"):
            return None, "The air quality service returned no current readings for this location."
        return result, None
    except requests.exceptions.RequestException as exc:
        return None, f"Could not reach the air quality service ({type(exc).__name__})."
    except ValueError:
        return None, "The air quality service returned an unreadable response."


@st.cache_data(show_spinner=False, ttl=3600)
def search_air_quality_places(query):
    """Resolve a city name to a short list of populated-city results."""
    if not HAS_REQUESTS:
        return [], "Install requests to search for locations."
    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": str(query).strip(), "count": 8, "language": "en", "format": "json"},
            timeout=12,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        city_results = [
            place for place in results
            if place.get("feature_code") == "PPL"
            or place.get("feature_code") == "PPLC"
            or str(place.get("feature_code", "")).startswith("PPLA")
        ]
        return city_results, None
    except requests.exceptions.RequestException as exc:
        return [], f"Could not search locations ({type(exc).__name__})."


def searched_air_quality_place_label(place):
    place_label = " · ".join(
        part for part in [place.get("name", "Place"), place.get("admin1"), place.get("country", "")]
        if part
    )
    if place.get("population"):
        place_label += f" · pop. {int(place['population']):,}"
    return place_label


def get_alert_secret(name, default=""):
    """Read alert-provider configuration from environment or Streamlit secrets."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        direct_value = st.secrets.get(name)
        if direct_value:
            return str(direct_value).strip()
        twilio_settings = st.secrets.get("twilio", {})
        return str(twilio_settings.get(name, default)).strip()
    except Exception:
        return default


def normalize_phone_number(value):
    """Normalize common phone formatting to Twilio's E.164 form."""
    raw_value = str(value or "").strip()
    if not raw_value.startswith("+"):
        return None
    digits = "".join(character for character in raw_value if character.isdigit())
    if not 8 <= len(digits) <= 15 or digits.startswith("0"):
        return None
    return f"+{digits}"


def twilio_verify_settings():
    return (
        get_alert_secret("TWILIO_ACCOUNT_SID"),
        get_alert_secret("TWILIO_AUTH_TOKEN"),
        get_alert_secret("TWILIO_VERIFY_SERVICE_SID"),
    )


def start_phone_verification(phone_number):
    account, token, service = twilio_verify_settings()
    if not HAS_REQUESTS:
        return False, "Phone sign-in needs the requests package."
    if not all([account, token, service]):
        return False, "Phone sign-in is not set up yet. Add the Twilio account SID, auth token, and Verify Service SID in app secrets."
    try:
        response = requests.post(
            f"https://verify.twilio.com/v2/Services/{service}/Verifications",
            data={"To": phone_number, "Channel": "sms"},
            auth=(account, token),
            timeout=15,
        )
        response.raise_for_status()
        if response.json().get("status") == "pending":
            return True, None
        return False, "Twilio did not start the phone verification. Check the Verify service settings."
    except Exception:
        return False, "Could not send the verification code. Check the Twilio settings and try again."


def check_phone_verification(phone_number, code):
    account, token, service = twilio_verify_settings()
    if not all([account, token, service]):
        return False, "Phone sign-in is not set up yet. Add the Twilio account SID, auth token, and Verify Service SID in app secrets."
    try:
        response = requests.post(
            f"https://verify.twilio.com/v2/Services/{service}/VerificationCheck",
            data={"To": phone_number, "Code": code},
            auth=(account, token),
            timeout=15,
        )
        response.raise_for_status()
        if response.json().get("status") == "approved":
            return True, None
        return False, "That code was not accepted. Check it and try again before it expires."
    except Exception:
        return False, "Could not verify the code. It may have expired, or the Twilio settings may need checking."


def send_aqi_email(recipient, subject, body):
    host = get_alert_secret("SMTP_HOST")
    username = get_alert_secret("SMTP_USER")
    password = get_alert_secret("SMTP_PASSWORD")
    sender = get_alert_secret("SMTP_FROM", username)
    if not all([host, username, password, sender]):
        return False, "Add SMTP_HOST, SMTP_USER, SMTP_PASSWORD, and SMTP_FROM to app secrets to enable email."
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    try:
        with smtplib.SMTP(host, int(get_alert_secret("SMTP_PORT", "587")), timeout=15) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(message)
        return True, None
    except Exception as exc:
        return False, f"Email delivery failed ({type(exc).__name__})."


def send_aqi_sms(recipient, body):
    account = get_alert_secret("TWILIO_ACCOUNT_SID")
    token = get_alert_secret("TWILIO_AUTH_TOKEN")
    sender = get_alert_secret("TWILIO_FROM_NUMBER")
    if not all([account, token, sender]):
        return False, "Add TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER to app secrets to enable texts."
    try:
        response = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{account}/Messages.json",
            data={"To": recipient, "From": sender, "Body": body},
            auth=(account, token),
            timeout=15,
        )
        response.raise_for_status()
        return True, None
    except Exception as exc:
        return False, f"Text delivery failed ({type(exc).__name__})."


def us_aqi_category(value):
    """Translate the modeled U.S. AQI value into its familiar category."""
    if value is None or pd.isna(value):
        return "Unavailable"
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Moderate"
    if value <= 150:
        return "Unhealthy for sensitive groups"
    if value <= 200:
        return "Unhealthy"
    if value <= 300:
        return "Very unhealthy"
    return "Hazardous"


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

with st.container(key="home_hero"):
    logo_column, title_column = st.columns([1, 5])
    with logo_column:
        if HOME_ART_PATH.is_file():
            st.image(str(HOME_ART_PATH), width=170)
    with title_column:
        st.markdown(
            """
            <h1>Earth Emergency Command</h1>
            <p>NASA satellite data, live fire detections, and environmental monitoring in one place.</p>
            """,
            unsafe_allow_html=True,
        )

with st.container(key="floating_login"):
    account_button_label = "👤 Account" if account_is_logged_in() else "👤 Log in"
    if st.button(account_button_label, key="floating_login_button"):
        show_login_dialog()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🛰️ Command Panel")

    if st.button("🔄 Refresh All Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    st.markdown("### 🧠 VIIRS Estimate Model")

    predictor_script = find_predictor_script()

    if predictor_script:
        st.caption("Uses the bundled VIIRS FIRMS sample; this is not a live forecast.")
        allow_local_training = os.getenv("ALLOW_LOCAL_MODEL_TRAINING", "").strip().lower() in {"1", "true", "yes"}
        ml_dependencies_available = all(
            importlib.util.find_spec(package) is not None
            for package in ("tensorflow", "sklearn", "joblib")
        )
        if not allow_local_training:
            st.caption("Model training is disabled in this dashboard to keep the app responsive. To enable it locally, install `requirements-ml.txt` and set `ALLOW_LOCAL_MODEL_TRAINING=1`.")
        elif not ml_dependencies_available:
            st.caption("Optional training packages are missing. Install `requirements-ml.txt` to run the model locally.")
        elif st.button("▶️ Train & Estimate Sample"):
            with st.spinner("Training the VIIRS model and estimating fire intensity..."):
                result = subprocess.run(
                    [sys.executable, predictor_script],
                    capture_output=True,
                    text=True,
                    cwd=ROOT_DIR,
                    timeout=3600,
                )

            if result.stdout:
                st.code(result.stdout[-3000:])

            if result.returncode != 0:
                st.error("Prediction script failed.")
                if result.stderr:
                    st.code(result.stderr[-3000:])
            else:
                st.success("VIIRS fire intensity estimate finished.")

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

    firms_source_options = [
        "All NRT feeds (recommended)",
        "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT",
    ]
    firms_source = st.selectbox(
        "Satellite feed",
        firms_source_options,
        key="firms_satellite_feed_v2",
        help="Combining all near-real-time sensors helps catch detections missed by one satellite pass."
    )
    firms_days = st.slider("FIRMS lookback days", 1, 5, 2, key="firms_lookback_days_v2")
    include_low_confidence = st.checkbox(
        "Include low-confidence detections",
        value=False,
        help="Off by default to reduce likely false alarms. Nominal detections are still algorithm detections, not confirmed ground reports."
    )

    st.markdown("### New Fire Popups")
    fire_alerts_enabled = st.checkbox(
        "Show popups for new fire detections",
        value=True,
        help="A popup appears when a live FIRMS refresh includes detections not seen in the previous refresh. The first live load sets the baseline."
    )
    fire_sms_enabled = st.checkbox(
        "Text me when new fires are detected",
        key="fire_alert_sms_enabled",
        help="Sends one text for each refresh batch with new detections in your selected country scope. Requires Twilio messaging settings."
    )
    fire_alert_phone = ""
    if fire_sms_enabled:
        fire_alert_phone = st.text_input(
            "Fire alert phone number (include country code)",
            placeholder="+1 555 123 4567",
            key="fire_alert_phone",
        ).strip()
    if st.button("Send test popup", help="Shows a sample notification without contacting NASA or changing your alert baseline."):
        st.toast("🔥 Test popup: fire alerts are working.", icon="🔥")
    if fire_sms_enabled and st.button("Send test fire text", key="test_fire_sms"):
        test_phone = normalize_phone_number(fire_alert_phone)
        if not test_phone:
            st.warning("Enter a valid number in international format, starting with + and the country code.")
        else:
            sent, error = send_aqi_sms(test_phone, "Earth Emergency Command test: fire alert texts are working.")
            if sent:
                st.success("Test text sent.")
            else:
                st.error(error)
    country_boundaries = load_country_boundaries()
    available_countries = sorted({item["name"] for item in country_boundaries})
    country_scope_mode = st.radio(
        "Country scope for maps and alerts",
        ["All countries", "Selected countries"],
        horizontal=True,
        key="country_scope_mode",
    )
    selected_countries = []
    if country_scope_mode == "Selected countries":
        if available_countries:
            selected_countries = st.multiselect(
                "Choose one or more countries",
                available_countries,
                key="fire_alert_countries",
                max_selections=5,
            )
        else:
            st.warning("Country boundaries could not be loaded, so country-specific alerts are unavailable.")
    st.caption("Selected countries filter point maps and new-fire alerts. Satellite imagery stays global but zooms to your selection. FIRMS detections are unconfirmed; country matching uses generalized boundaries.")
    country_focus_bounds = (
        selected_country_focus_bounds(selected_countries, country_boundaries)
        if country_scope_mode == "Selected countries" and selected_countries
        else None
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

    with st.expander("⚙️ Settings & account", expanded=False):
        if current_user_logged_in():
            account_label = getattr(st.user, "name", None) or getattr(st.user, "email", None) or "Signed in"
            st.caption(f"Signed in as {account_label}")
            if st.button("Log out", key="settings_logout", width="stretch"):
                st.logout()
        elif current_phone_user():
            st.caption(f"Phone verified for this session: {current_phone_user()}")
            if st.button("Log out", key="settings_phone_logout", width="stretch"):
                st.session_state.pop("phone_signin_verified_number", None)
                st.rerun()
        else:
            st.caption("Signing in is optional.")
            if st.button("Log in", key="settings_login", width="stretch"):
                show_login_dialog()

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

live_load_status = st.status("Connecting to NASA feeds…", expanded=False)
live_events, live_error = fetch_eonet_events(limit=event_limit)
firms_url_loaded = False

if firms_url:
    firms_df, firms_error = fetch_firms_url(firms_url)
    firms_url_loaded = firms_df is not None
    firms_is_live = False
    firms_source_used = "CSV URL"
    if firms_df is not None:
        firms_error = "Showing the configured FIRMS CSV URL; freshness depends on that source."
elif firms_key:
    all_nrt_sources = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"]
    sources_to_fetch = all_nrt_sources if firms_source == "All NRT feeds (recommended)" else [firms_source]
    available_frames = []
    available_sources = []
    source_errors = []
    for nrt_source in sources_to_fetch:
        source_frame, source_error = fetch_firms_api(firms_key, nrt_source, firms_days)
        if source_frame is not None and not source_frame.empty:
            source_frame = source_frame.copy()
            source_frame["__feed_source"] = nrt_source
            available_frames.append(source_frame)
            available_sources.append(nrt_source)
        elif source_error and source_error != "NASA FIRMS returned no fire detections for this period.":
            source_errors.append(f"{nrt_source}: {source_error}")

    firms_source_used = ", ".join(available_sources)
    if available_frames:
        firms_df = pd.concat(available_frames, ignore_index=True, sort=False)
        # Do not collapse coincident observations from separate sensors: they
        # are independent satellite passes and help validate local clusters.
        dedupe_columns = [column for column in ("lat", "lon", "date", "acq_time", "frp", "satellite", "instrument") if column in firms_df.columns]
        if len(dedupe_columns) >= 3:
            firms_df = firms_df.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)
        firms_error = (
            f"Combined NASA FIRMS detections from {firms_source_used} for the last {firms_days} days."
        )
        if source_errors:
            firms_error += " Some sensor feeds did not respond: " + " · ".join(source_errors)
    else:
        firms_df = None
        firms_source_used = firms_source
        firms_error = (
            "No detections were returned from the selected NASA FIRMS feeds "
            f"for the last {firms_days} days. " + (" ".join(source_errors) if source_errors else "Try a longer lookback period.")
        )
    firms_is_live = firms_df is not None
else:
    firms_df, firms_error = None, (
        "No NASA FIRMS MAP_KEY reached this dashboard process. Enter it in the "
        "password field in the sidebar, or launch Streamlit from the PowerShell "
        "session where NASA_FIRMS_MAP_KEY is set."
    )
    firms_is_live = False
    firms_source_used = firms_source

firms_display_source = (
    f"live NASA FIRMS feed: {firms_source_used}"
    if firms_is_live
    else "configured FIRMS CSV URL"
    if firms_url_loaded
    else "saved local FIRMS snapshot (not live)"
)

if firms_df is None:
    local_firms_df = local_firms_snapshot()
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

firms_data_is_current = bool(firms_is_live)
firms_freshness_note = ""
if firms_url_loaded and firms_df is not None:
    if "date" not in firms_df.columns:
        firms_freshness_note = "The configured CSV has no observation dates, so its hotspots cannot be verified as current."
    else:
        latest_firms_date = pd.to_datetime(firms_df["date"], errors="coerce", utc=True).max()
        age_hours = (pd.Timestamp.now(tz="UTC") - latest_firms_date).total_seconds() / 3600 if pd.notna(latest_firms_date) else float("inf")
        firms_data_is_current = age_hours <= max(24, firms_days * 24)
        if not firms_data_is_current:
            firms_freshness_note = f"The configured CSV is {age_hours / 24:.1f} days old, beyond the selected {firms_days}-day period."
if not firms_data_is_current and not firms_freshness_note:
    firms_freshness_note = "Only a saved historical snapshot is available; it is not plotted as current fire activity."

if live_events is not None and not live_events.empty and not show_all_geometry:
    live_events = (
        live_events.sort_values("date")
        .groupby("event_id", as_index=False)
        .tail(1)
    )
if live_events is not None and not live_events.empty and {"lat", "lon"}.issubset(live_events.columns):
    live_events["place"] = country_labels_for_points(live_events, country_boundaries)

# ============================================================
# NEW FIRE DETECTION POPUPS
# ============================================================

if firms_is_live and firms_df is not None:
    alert_ids = fire_detection_ids(firms_df)
    feed_key = str(firms_source_used)
    seen_by_feed = st.session_state.setdefault("seen_firms_alert_ids", {})
    previous_ids = set(seen_by_feed.get(feed_key, []))
    current_ids = set(alert_ids.tolist())
    is_first_live_snapshot = feed_key not in seen_by_feed

    if not is_first_live_snapshot and not alert_ids.empty:
        newly_seen = ~alert_ids.isin(previous_ids)
        new_detections = firms_df.loc[newly_seen].copy()
        country_scope_active = country_scope_mode == "Selected countries"
        if country_scope_active:
            new_detections = filter_points_to_country_scope(
                new_detections, selected_countries, True, country_boundaries
            )

        if not new_detections.empty:
            latest = new_detections.iloc[0]
            latest_frame = new_detections.iloc[[0]]
            place = country_labels_for_points(latest_frame, country_boundaries).iloc[0]
            coordinates = f"{float(latest['lat']):.2f}, {float(latest['lon']):.2f}"
            location = f"{place} ({coordinates})"
            scope_label = ", ".join(selected_countries) if country_scope_active else "all countries"
            if fire_alerts_enabled:
                st.toast(
                    f"🔥 {len(new_detections):,} new fire detection(s) in {scope_label}. Latest location: {location}.",
                    icon="🔥",
                )
            if fire_sms_enabled:
                sms_phone = normalize_phone_number(fire_alert_phone)
                if sms_phone:
                    fire_message = (
                        f"Earth Emergency Command: {len(new_detections):,} new FIRMS fire detection(s) "
                        f"in {scope_label}. Latest location: {location}. Detections are unconfirmed."
                    )
                    delivered, delivery_error = send_aqi_sms(sms_phone, fire_message)
                    if delivered:
                        st.toast("📱 Fire alert text sent.", icon="📱")
                    else:
                        st.warning(delivery_error)
                else:
                    st.warning("New fires were detected, but the fire alert phone number is missing or invalid.")

    # Keep a rolling baseline per satellite feed so detections do not alert again
    # on every automatic refresh. The limit bounds session memory on busy feeds.
    seen_by_feed[feed_key] = list((previous_ids | current_ids))[-50000:]

live_load_status.update(
    label="NASA feed check complete",
    state="complete",
    expanded=False,
)


# ============================================================
# METRICS
# ============================================================

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

d1, d2, d3 = st.columns(3)

with d1:
    st.metric("🛰️ Live Events", live_count)

with d2:
    st.metric("🔥 Live Fire Events", live_fire_count)

with d3:
    st.metric("🌊 Live Flood Events", live_flood_count)


# ============================================================
# TABS
# ============================================================

tab_global_map, tab_live_events, tab_live_fire, tab_local_hazards, tab_atmosphere, tab_prediction, tab_command, tab_tables, tab_coordinates, tab_sources = st.tabs(
    [
        "🌍 Overview Map",
        "🚨 Live Events",
        "🔥 Live Fire",
        "🛰️ Local Hazards",
        "🌫️ Air & Atmosphere",
        "🔮 Prediction",
        "📁 Local Data",
        "📋 Data Tables",
        "🧭 Coordinates",
        "📚 Sources",
    ]
)


# ============================================================
# TAB: MISSION CONTROL
# ============================================================

with tab_command:
    st.markdown("## 📁 Local Data")
    st.caption("A quick summary of the files and regions loaded from your project.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Local files", files_loaded)
    m2.metric("Rows", total_rows)
    m3.metric("Regions", total_regions)
    m4.metric("Mapped points", total_points)
    st.metric("Highest estimated fire score", f"{max_risk:.2f}")
    st.caption(f"Local emergency labels: {emergency_labels:,}")

    st.markdown("### Local Input Files")

    if file_summary.empty:
        st.warning(
            "No local input files found. Put `.csv` or `.cvv` files into `data/`."
        )
    else:
        st.caption(f"{files_loaded:,} local file(s) loaded")
        with st.expander("Browse local input files"):
            st.dataframe(file_summary, width="stretch")

    st.markdown("### Region Registry")

    if registry.empty:
        st.info("No regions detected.")
    else:
        st.caption(f"{len(registry):,} regions in the local registry")
        with st.expander("Browse region coordinates and signals"):
            st.dataframe(registry, width="stretch")

    with st.expander("Loading details"):
        if logs:
            for log in logs:
                st.text(log)
        else:
            st.caption("No log messages.")


# ============================================================
# TAB: GLOBAL MAP
# ============================================================

# Switch existing sessions from the previous, overly dense heat style to the
# softened heat-only view once. After that, each map's selector remains user controlled.
if not st.session_state.get("_soft_heat_map_default_v1"):
    map_view_keys = ["global_map_view", "live_events_map_view", "firms_map_view"]
    map_view_keys.extend(
        f"local_{column}_map_view"
        for column in ("fire_intensity", "heat_intensity", "smoke_intensity", "haze_intensity", "flood_intensity", "co2_intensity")
    )
    for map_view_key in map_view_keys:
        st.session_state[map_view_key] = "Soft heat map"
    st.session_state["_soft_heat_map_default_v1"] = True

with tab_global_map:
    st.markdown("## 🌍 Earth activity at a glance")
    st.caption("Start with the map. Choose a layer to explore NASA event reports, satellite fire detections, or local environmental signals.")

    layer_options = [
        "Live EONET Events",
        "FIRMS Fires",
        "Estimated Fire Intensity",
        "Local Fire proxy",
        "Local Heat proxy",
        "Local Smoke proxy",
        "Local Haze proxy",
        "Local Flood proxy",
        "Local CO₂ proxy",
    ]

    layer = st.selectbox("Map layer", layer_options)
    map_view = st.radio(
        "Map view", ["Globe / flat map", "Interactive tile map", "Soft heat map"],
        index=2, horizontal=True, key="global_map_view",
    )
    if map_view == "Globe / flat map":
        map_style = st.radio("Map style", ["Globe", "Flat world"], horizontal=True, key="global_map_style")
        projection = "orthographic" if map_style == "Globe" else "natural earth"
    else:
        projection = None

    if layer == "Live EONET Events":
        st.info(
            "EONET shows NASA's curated event records, not every satellite fire hotspot. "
            "For broader fire coverage, choose **FIRMS Fires** in Map layer; FIRMS plots "
            "thermal detections, which are indicators and not all confirmed wildfires."
        )
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
            map_df = filter_points_to_country_scope(
                map_df, selected_countries,
                country_scope_mode == "Selected countries", country_boundaries,
            )

            if map_df.empty:
                st.info("No events match the selected hazards and countries. Choose at least one country or switch to all countries.")

            else:
                mapped_events = map_df.dropna(subset=["lat", "lon"])
                st.caption(
                    f"Showing {len(mapped_events):,} mapped EONET event records out of "
                    f"{len(map_df):,} selected records. Events without coordinates remain in the list below."
                )
                if not mapped_events.empty:
                    if map_view != "Globe / flat map":
                        render_leaflet_points(mapped_events, color_by="hazard", category_colors=HAZARD_COLORS,
                                              popup_columns=["title", "place", "hazard", "categories", "date"], height=460,
                                              thermal=map_view == "Soft heat map",
                                              key="map_eonet_global_leaflet")
                    else:
                        fig = px.scatter_geo(
                            mapped_events,
                            lat="lat",
                            lon="lon",
                            color="hazard",
                            hover_name="title",
                            hover_data=[c for c in ["place", "categories", "date"] if c in mapped_events.columns],
                            color_discrete_map=HAZARD_COLORS,
                            projection=projection,
                            height=460,
                        )
                        fig = make_geo_fig(fig)
                        st.plotly_chart(
                            fig,
                            width="stretch",
                            config={"responsive": True, "displaylogo": False, "scrollZoom": True},
                            key="map_eonet_global",
                        )
                else:
                    st.info("No coordinates are available for these filtered events, so they appear in the event list instead of on the map.")
                show_eonet_event_browser(map_df, key="global_map_eonet")

    elif layer == "FIRMS Fires":
        if firms_df is None:
            st.info(firms_error)
        else:
            firms_map_df = filter_points_to_country_scope(
                firms_df, selected_countries,
                country_scope_mode == "Selected countries", country_boundaries,
            )
            if not firms_data_is_current:
                st.warning(firms_error)
                st.caption(firms_freshness_note)
                firms_map_df = firms_map_df.iloc[0:0].copy()
            if firms_map_df.empty:
                if not firms_data_is_current:
                    st.info("No current fire layer is shown because this data is stale or its date cannot be verified. Historical records remain in the table below.")
                else:
                    st.info("No NASA FIRMS detections are in the selected countries for this feed period.")
            else:
                if map_view != "Globe / flat map":
                    if map_view == "Soft heat map":
                        render_leaflet_points(firms_map_df, color_by="frp", height=460, point_limit=2000,
                                              thermal=True,
                                              pixel_size_meters=1000 if firms_source_used == "MODIS_NRT" else 375,
                                              key="map_firms_global_thermal_leaflet")
                    else:
                        render_leaflet_points(firms_map_df, color_by="frp", height=460, point_limit=2000,
                                              pixel_size_meters=1000 if firms_source_used == "MODIS_NRT" else 375,
                                              key="map_firms_global_leaflet")
                else:
                    fig = make_thermal_fire_map(firms_map_df, projection=projection, height=460)
                    st.plotly_chart(
                        fig,
                        width="stretch",
                        config={"scrollZoom": True, "responsive": True, "displayModeBar": True, "displaylogo": False},
                        key="map_firms_global",
                    )
                if map_view == "Soft heat map":
                    st.caption(f"Data source: {firms_display_source}. Soft heat colors show relative satellite fire radiative power (FRP) across supported clusters, not ground temperature or fire perimeters. Isolated detections remain available in the standard point map.")
                else:
                    st.caption(f"Data source: {firms_display_source}. NASA FIRMS satellite detections are shown as individual points. A single detection does not confirm a wildfire; use the table for the full feed.")

            with st.expander("View fire detection records"):
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
            base_map = filter_points_to_country_scope(
                base_map, selected_countries,
                country_scope_mode == "Selected countries", country_boundaries,
            )
            if base_map.empty:
                st.info("No local points are in the selected countries.")
            color = None
            size = None
            color_scale = "Viridis"

            if layer == "Estimated Fire Intensity":
                if risk_latest is None:
                    st.warning("No fire intensity estimate found. Run the VIIRS workflow from the sidebar.")

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

                if map_view != "Globe / flat map":
                    popup = [c for c in ["region", color, "latest_risk"] if c and c in map_df.columns]
                    render_leaflet_points(map_df, color_by=color, popup_columns=popup, height=560,
                                          thermal=map_view == "Soft heat map",
                                          key="map_local_global_leaflet")
                else:
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
                    st.plotly_chart(
                        fig,
                        width="stretch",
                        config={"responsive": True, "scrollZoom": True, "displayModeBar": True, "displaylogo": False},
                        key="map_local_global",
                    )

                with st.expander("View mapped local records"):
                    st.dataframe(map_df, width="stretch")


# ============================================================
# TAB: PREDICTION
# ============================================================

with tab_prediction:
    st.markdown("## 🔥 Fire Intensity Estimate")

    st.markdown("### Model Status")

    risk_exists = os.path.exists(os.path.join(OUTPUT_DIR, "future_risk.csv"))
    model_exists = os.path.exists(os.path.join(OUTPUT_DIR, "risk_model.keras"))
    scaler_exists = os.path.exists(os.path.join(OUTPUT_DIR, "scaler.pkl"))

    st.write(f"Saved VIIRS intensity estimates: {'✅' if risk_exists else '❌'}")
    st.write(f"Trained model: {'✅' if model_exists else '❌'}")
    st.write(f"Saved feature preprocessing: {'✅' if scaler_exists else '❌'}")

    if risk is None:
        st.warning("No estimate found. Run the VIIRS workflow from the sidebar.")

    else:
        risk_col = None

        preferred = [
            "latest_risk",
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
            st.markdown("### Estimated Fire Intensity")
            st.caption("This model estimates fire radiative power for detections in the bundled VIIRS FIRMS file. The 0–1 map score is FRP / (FRP + 10); it is not a fire probability or a forecast.")

            risk_values = pd.to_numeric(risk[risk_col], errors="coerce").dropna()
            if not risk_values.empty and ((risk_values <= 1e-5).any() or (risk_values >= 1 - 1e-5).any()):
                st.caption("The bounded display score approaches 0 or 1 for very low or high estimated intensity.")

            if "date" in risk.columns:
                risk_plot = risk.copy()
                risk_plot["date"] = pd.to_datetime(risk_plot["date"], errors="coerce")
                valid_dates = risk_plot["date"].dropna()
                if not valid_dates.empty and valid_dates.max().date() < datetime.date.today():
                    st.warning(
                        f"This saved sample ends on {valid_dates.max().date()}; it is historical, not a live estimate."
                    )
                if not valid_dates.empty:
                    fig = px.line(
                        risk_plot.dropna(subset=["date", risk_col]),
                        x="date",
                        y=risk_col,
                        color="region" if "region" in risk_plot.columns else None,
                        height=480,
                        title="Estimated fire intensity score by date",
                    )
                    fig = make_dark_fig(fig)
                    st.plotly_chart(fig, width="stretch", key="chart_prediction_timeline")

            st.markdown("### Most Recent Saved Intensity Score")

            if risk_latest is not None:
                latest_sorted = risk_latest.sort_values("latest_risk", ascending=False)
                if not latest_sorted.empty:
                    st.caption(f"Highest score: {latest_sorted.iloc[0].get('region', 'Unknown region')} · {float(latest_sorted.iloc[0]['latest_risk']):.2f}")
                with st.expander("Browse latest scores by region"):
                    st.dataframe(latest_sorted, width="stretch")

            with st.expander("Browse all saved risk records"):
                st.dataframe(risk, width="stretch")

    st.markdown("### Trend Report")

    if trends is None:
        st.info("No trend_report.csv found.")

    else:
        if "significant" in trends.columns:
            significant_trends = trends[trends["significant"] == True]
            st.caption(f"{len(significant_trends):,} significant trend(s) found")
            with st.expander("Browse significant trends"):
                st.dataframe(significant_trends, width="stretch")
        else:
            with st.expander("Browse trend report"):
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
                    st.plotly_chart(fig, width="stretch", key="chart_local_hazard_timeline")

            st.markdown(f"### {selected_hazard} Locations")

            if point_summary is not None and col in point_summary.columns:
                top_points = point_summary.sort_values(col, ascending=False).head(100)
                top_points = filter_points_to_country_scope(
                    top_points, selected_countries,
                    country_scope_mode == "Selected countries", country_boundaries,
                )
                if top_points.empty:
                    st.info("No locations for this signal are in the selected countries.")
                st.caption(f"Showing the strongest {len(top_points):,} mapped locations for this signal.")
                with st.expander("Browse location records"):
                    st.dataframe(top_points, width="stretch")

                map_df = top_points.dropna(subset=["lat", "lon"])

                if not map_df.empty:
                    map_view = st.radio(
                        "Map view", ["Globe / flat map", "Interactive tile map", "Soft heat map"],
                        index=2, horizontal=True, key=f"local_{col}_map_view",
                    )
                    if map_view != "Globe / flat map":
                        render_leaflet_points(map_df, color_by=col, popup_columns=["region", col],
                                              height=560, thermal=map_view == "Soft heat map",
                                              key=f"map_local_hazard_{col}_leaflet")
                    else:
                        map_style = st.radio(
                            "Map style", ["Globe", "Flat world"], horizontal=True,
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
                        st.plotly_chart(
                            fig,
                            width="stretch",
                            config={"responsive": True, "scrollZoom": True, "displayModeBar": True, "displaylogo": False},
                            key="map_local_hazard",
                        )


# ============================================================
# TAB: AIR & ATMOSPHERE
# ============================================================

with tab_atmosphere:
    st.markdown("## 🌫️ Air & Atmosphere")
    st.markdown("### 🌬️ Current Air Quality")
    st.caption("Search a city or postcode, compare places, and follow AQI changes. Estimates are modelled, not street monitors.")

    with st.form("air_quality_city_search_form"):
        search_col, search_button_col = st.columns([4, 1])
        with search_col:
            aqi_place_query = st.text_input("Find a city", placeholder="e.g. Surabaya, Indonesia", key="air_quality_place_query")
        with search_button_col:
            st.write("")
            aqi_search_clicked = st.form_submit_button("Search city + show AQI & fires", width="stretch")
    if aqi_search_clicked:
        st.session_state.pop("air_quality_place_results", None)
        st.session_state.pop("air_quality_selected_place", None)
        if len(aqi_place_query.strip()) < 2:
            st.warning("Enter at least two characters to search.")
        else:
            with st.spinner("Finding places…"):
                place_results, place_error = search_air_quality_places(aqi_place_query)
            if place_error:
                st.error(place_error)
            elif not place_results:
                st.info("No matching places found. Try adding a country, such as ‘Jakarta, Indonesia’.")
            else:
                st.session_state["air_quality_place_results"] = place_results
                first_place = place_results[0]
                selected_place_label = searched_air_quality_place_label(first_place)
                st.session_state["air_quality_selected_place"] = {
                    "label": selected_place_label,
                    "lat": float(first_place["latitude"]),
                    "lon": float(first_place["longitude"]),
                }
                st.session_state["_air_quality_search_auto_load"] = True
                st.success(f"Found {selected_place_label}. Loading AQI and nearby satellite hotspots below.")

    searched_places = st.session_state.get("air_quality_place_results", [])

    aqi_locations = registry.dropna(subset=["lat", "lon"]).copy() if registry is not None and not registry.empty else pd.DataFrame(columns=["region", "lat", "lon"])
    aqi_location_lookup = {}
    for _, location_row in aqi_locations.iterrows():
        country_name = country_labels_for_points(aqi_locations.loc[[location_row.name]], country_boundaries).iloc[0]
        location_label = f"{location_row['region']} · {country_name} ({float(location_row['lat']):.2f}, {float(location_row['lon']):.2f})"
        aqi_location_lookup[location_label] = (float(location_row["lat"]), float(location_row["lon"]))
    for place in searched_places:
        place_label = searched_air_quality_place_label(place)
        aqi_location_lookup[place_label] = (float(place["latitude"]), float(place["longitude"]))
    selected_air_quality_place = st.session_state.get("air_quality_selected_place")
    if selected_air_quality_place:
        selected_aqi_location = selected_air_quality_place["label"]
        location_name = selected_aqi_location
        aqi_lat = selected_air_quality_place["lat"]
        aqi_lon = selected_air_quality_place["lon"]
        st.caption(f"Selected place: {location_name}")
        st.markdown(f"#### 🔥 NASA fire detections near {location_name}")
        if firms_df is None:
            st.info(f"No current NASA FIRMS detections are available: {firms_error}")
        elif not firms_data_is_current:
            st.warning(
                f"Current fires near this city cannot be verified: {firms_freshness_note} "
                "Add a valid NASA FIRMS MAP_KEY to refresh detections."
            )
        else:
            nearby_fires = filter_points_near_location(firms_df, aqi_lat, aqi_lon, radius_km=75)
            if nearby_fires.empty:
                st.success("No NASA FIRMS satellite hotspots were detected within 75 km in the selected feed period.")
            else:
                st.caption(
                    f"{len(nearby_fires):,} satellite thermal detections within 75 km · {firms_display_source}. "
                    "These are hotspot detections, not confirmed fires or fire boundaries."
                )
                latitude_radius = 75 / 111.195
                longitude_radius = min(180, latitude_radius / max(0.08, np.cos(np.radians(float(aqi_lat)))))
                render_leaflet_points(
                    nearby_fires, color_by="frp", height=380, point_limit=2000, thermal=True,
                    pixel_size_meters=1000 if firms_source_used == "MODIS_NRT" else 375,
                    focus_bounds=[aqi_lon - longitude_radius, aqi_lat - latitude_radius,
                                  aqi_lon + longitude_radius, aqi_lat + latitude_radius],
                    key="aqi_city_nearby_fires_map",
                )
    else:
        selected_aqi_location = None
        location_name = ""
        aqi_lat = aqi_lon = None
        st.info("Search for a city or postcode above to load its current AQI.")

    with st.expander("📍 Compare and follow places", expanded=False):
        st.caption("Search for a city above, then select it here to monitor its AQI. Choose up to five places worldwide. The map’s country filter does not limit AQI search or alerts.")
        place_choices = list(aqi_location_lookup.keys())
        if "air_quality_followed_places" in st.session_state:
            st.session_state["air_quality_followed_places"] = [
                place for place in st.session_state["air_quality_followed_places"] if place in place_choices
            ][:5]
        followed_places = st.multiselect(
            "Places to compare and monitor",
            place_choices,
            default=[selected_aqi_location] if selected_aqi_location in place_choices else [],
            max_selections=5,
            key="air_quality_followed_places",
        )
        compare_clicked = st.button("Compare AQI now", key="air_quality_compare")
        if compare_clicked and followed_places:
            comparison_rows = []
            comparison_error = None
            with st.spinner("Checking AQI across your places…"):
                for place_label in followed_places:
                    lat, lon = aqi_location_lookup[place_label]
                    result, error = fetch_air_quality(lat, lon)
                    if error:
                        comparison_error = f"{place_label}: {error}"
                        continue
                    current = result.get("current", {})
                    comparison_rows.append({
                        "Place": place_label,
                        "U.S. AQI": current.get("us_aqi"),
                        "Category": us_aqi_category(current.get("us_aqi")),
                        "PM₂.₅": current.get("pm2_5"),
                        "PM₁₀": current.get("pm10"),
                        "Ozone": current.get("ozone"),
                        "Checked": current.get("time", ""),
                    })
            if comparison_rows:
                comparison_frame = pd.DataFrame(comparison_rows)
                st.session_state["air_quality_comparison"] = comparison_frame
                for row in comparison_rows:
                    score = row["U.S. AQI"]
                    threshold = st.session_state.get("aqi_alert_threshold", 101)
                    if score is not None and not pd.isna(score) and float(score) >= threshold:
                        sent_alerts = st.session_state.setdefault("air_quality_alerted_places", {})
                        alert_key = row["Place"]
                        delivery_enabled = (
                            st.session_state.get("aqi_alert_email_enabled") and st.session_state.get("aqi_alert_email", "").strip()
                        ) or (
                            st.session_state.get("aqi_alert_sms_enabled") and st.session_state.get("aqi_alert_phone", "").strip()
                        )
                        if delivery_enabled and not sent_alerts.get(alert_key):
                            alert_text = f"AQI alert: {row['Place']} is {row['Category']} (U.S. AQI {int(float(score))})."
                            alert_errors = []
                            delivered = False
                            email_to = st.session_state.get("aqi_alert_email", "").strip()
                            sms_to = st.session_state.get("aqi_alert_phone", "").strip()
                            if st.session_state.get("aqi_alert_email_enabled") and email_to:
                                ok, error = send_aqi_email(email_to, "Air quality alert", alert_text)
                                delivered = delivered or ok
                                if error:
                                    alert_errors.append(error)
                            if st.session_state.get("aqi_alert_sms_enabled") and sms_to:
                                ok, error = send_aqi_sms(sms_to, alert_text)
                                delivered = delivered or ok
                                if error:
                                    alert_errors.append(error)
                            if alert_errors:
                                st.warning(" ".join(alert_errors))
                            sent_alerts[alert_key] = delivered
                    elif score is not None and not pd.isna(score):
                        st.session_state.setdefault("air_quality_alerted_places", {})[row["Place"]] = False
            if comparison_error:
                st.warning(comparison_error)

        comparison_frame = st.session_state.get("air_quality_comparison")
        if comparison_frame is not None and not comparison_frame.empty:
            st.dataframe(comparison_frame, hide_index=True, width="stretch")
            chart_frame = comparison_frame.dropna(subset=["U.S. AQI"])
            if not chart_frame.empty:
                compare_chart = px.bar(chart_frame, x="Place", y="U.S. AQI", color="Category", height=330, title="Current AQI by place")
                st.plotly_chart(make_dark_fig(compare_chart), width="stretch", key="air_quality_comparison_chart")

    with st.expander("🔔 AQI reminders by email or text", expanded=False):
        st.caption("Alerts are checked every five minutes while the dashboard is open with Auto refresh on. Delivery needs your own SMTP/Twilio settings in app secrets.")
        st.number_input("Alert when AQI reaches", min_value=51, max_value=500, value=101, step=10, key="aqi_alert_threshold")
        st.caption("AQI alerts work worldwide; readings use the U.S. AQI scale.")
        email_enabled_col, sms_enabled_col = st.columns(2)
        with email_enabled_col:
            st.checkbox("Email alert", key="aqi_alert_email_enabled")
            st.text_input("Email address", placeholder="you@example.com", key="aqi_alert_email")
        with sms_enabled_col:
            st.checkbox("Text alert", key="aqi_alert_sms_enabled")
            st.text_input("Phone number (include country code)", placeholder="+1 555 123 4567", key="aqi_alert_phone")
        st.caption("AQI reminders follow the places in your watchlist worldwide. The map’s country filter only affects maps and new-fire alerts.")
        with st.expander("One-time provider setup"):
            st.code('SMTP_HOST = "..."\nSMTP_PORT = 587\nSMTP_USER = "..."\nSMTP_PASSWORD = "..."\nSMTP_FROM = "..."\n\n[twilio]\nTWILIO_ACCOUNT_SID = "..."\nTWILIO_AUTH_TOKEN = "..."\nTWILIO_FROM_NUMBER = "..."\nTWILIO_VERIFY_SERVICE_SID = "VA..."', language="toml")
            st.caption("Use Twilio messaging settings for text alerts and a Twilio Verify Service SID for phone sign-in. Store credentials in Streamlit secrets or environment variables. Never paste passwords here.")

    alerts_active = (
        st.session_state.get("aqi_alert_email_enabled") and st.session_state.get("aqi_alert_email", "").strip()
    ) or (
        st.session_state.get("aqi_alert_sms_enabled") and st.session_state.get("aqi_alert_phone", "").strip()
    )
    monitor_time = time.time()
    last_monitor_time = st.session_state.get("air_quality_last_monitor_check", 0)
    if alerts_active and followed_places and auto_refresh and monitor_time - last_monitor_time >= 300:
        st.session_state["air_quality_last_monitor_check"] = monitor_time
        alert_threshold = st.session_state.get("aqi_alert_threshold", 101)
        alerted_places = st.session_state.setdefault("air_quality_alerted_places", {})
        for place_label in followed_places:
            place_lat, place_lon = aqi_location_lookup[place_label]
            monitor_result, monitor_error = fetch_air_quality(place_lat, place_lon)
            if monitor_error:
                continue
            monitor_score = monitor_result.get("current", {}).get("us_aqi")
            if monitor_score is None or pd.isna(monitor_score):
                continue
            if float(monitor_score) < alert_threshold:
                alerted_places[place_label] = False
                continue
            if alerted_places.get(place_label):
                continue
            alert_message = f"AQI alert: {place_label} is {us_aqi_category(monitor_score)} (U.S. AQI {int(float(monitor_score))})."
            delivered = False
            if st.session_state.get("aqi_alert_email_enabled"):
                delivered, _ = send_aqi_email(st.session_state.get("aqi_alert_email", "").strip(), "Air quality alert", alert_message)
            if st.session_state.get("aqi_alert_sms_enabled"):
                sms_delivered, _ = send_aqi_sms(st.session_state.get("aqi_alert_phone", "").strip(), alert_message)
                delivered = delivered or sms_delivered
            alerted_places[place_label] = delivered
    elif alerts_active and followed_places and not auto_refresh:
        st.caption("Turn on Auto refresh in the sidebar to check AQI reminders every five minutes.")

    load_aqi_clicked = st.button("Load current AQI", key="load_air_quality") if selected_aqi_location else False
    auto_load_searched_place = st.session_state.pop("_air_quality_search_auto_load", False)
    if selected_aqi_location and (load_aqi_clicked or auto_load_searched_place):
        with st.spinner(f"Loading air quality for {location_name}…"):
            air_quality_result, air_quality_error = fetch_air_quality(aqi_lat, aqi_lon)
        if air_quality_error:
            st.session_state.pop("air_quality_snapshot", None)
            st.error(air_quality_error)
        else:
            st.session_state["air_quality_snapshot"] = {
                "location": location_name,
                "lat": round(float(aqi_lat), 4),
                "lon": round(float(aqi_lon), 4),
                "data": air_quality_result,
            }
            current_value = air_quality_result.get("current", {}).get("us_aqi")
            if current_value is not None and not pd.isna(current_value):
                history = st.session_state.setdefault("air_quality_history", [])
                now = datetime.datetime.now().astimezone()
                if not history or history[-1]["Location"] != location_name or (now - history[-1]["Checked"]).total_seconds() >= 300:
                    history.append({"Checked": now, "Location": location_name, "U.S. AQI": float(current_value)})
                    st.session_state["air_quality_history"] = history[-500:]

    saved_air_quality = st.session_state.get("air_quality_snapshot")
    if selected_aqi_location and saved_air_quality and saved_air_quality["lat"] == round(float(aqi_lat), 4) and saved_air_quality["lon"] == round(float(aqi_lon), 4):
        air_quality = saved_air_quality["data"]
        current_air = air_quality.get("current", {})
        current_units = air_quality.get("current_units", {})
        us_aqi = current_air.get("us_aqi")
        category = us_aqi_category(us_aqi)
        if category == "Good":
            st.success(f"{saved_air_quality['location']} · {category} air quality")
        elif category in {"Moderate", "Unhealthy for sensitive groups"}:
            st.warning(f"{saved_air_quality['location']} · {category} air quality")
        elif category == "Unavailable":
            st.info(f"Air quality data loaded for {saved_air_quality['location']}.")
        else:
            st.error(f"{saved_air_quality['location']} · {category} air quality")

        def aqi_metric(variable, label, unit=None):
            value = current_air.get(variable)
            if value is None or pd.isna(value):
                return "—"
            number = f"{float(value):.0f}" if variable in {"us_aqi", "european_aqi"} else f"{float(value):.1f}"
            display_unit = unit or current_units.get(variable, "")
            return f"{number} {display_unit}".strip()

        aq1, aq2, aq3, aq4 = st.columns(4)
        aq1.metric("U.S. AQI", aqi_metric("us_aqi", "U.S. AQI"), category)
        aq2.metric("European AQI", aqi_metric("european_aqi", "European AQI"))
        aq3.metric("PM₂.₅", aqi_metric("pm2_5", "PM₂.₅"))
        aq4.metric("PM₁₀", aqi_metric("pm10", "PM₁₀"))

        st.markdown("#### Other pollutants")
        ap1, ap2, ap3, ap4 = st.columns(4)
        ap1.metric("Ozone (O₃)", aqi_metric("ozone", "Ozone"))
        ap2.metric("Nitrogen dioxide (NO₂)", aqi_metric("nitrogen_dioxide", "NO₂"))
        ap3.metric("Sulfur dioxide (SO₂)", aqi_metric("sulphur_dioxide", "SO₂"))
        ap4.metric("Carbon monoxide (CO)", aqi_metric("carbon_monoxide", "CO"))

        hourly_air = air_quality.get("hourly", {})
        hourly_times = pd.to_datetime(hourly_air.get("time", []), errors="coerce")
        chart_options = {
            "U.S. AQI": ("us_aqi", "AQI"),
            "European AQI": ("european_aqi", "AQI"),
            "PM₂.₅": ("pm2_5", "µg/m³"),
            "PM₁₀": ("pm10", "µg/m³"),
            "Ozone (O₃)": ("ozone", "µg/m³"),
        }
        chart_options = {name: spec for name, spec in chart_options.items() if spec[0] in hourly_air}
        if chart_options and len(hourly_times):
            chart_name = st.selectbox("Next 3 days", list(chart_options), key="air_quality_chart_variable")
            chart_variable, chart_unit = chart_options[chart_name]
            chart_frame = pd.DataFrame({"Time": hourly_times, "Value": hourly_air[chart_variable]})
            chart_frame["Value"] = pd.to_numeric(chart_frame["Value"], errors="coerce")
            chart_frame = chart_frame.dropna(subset=["Time", "Value"]).head(72)
            if not chart_frame.empty:
                chart = px.line(chart_frame, x="Time", y="Value", height=330, title=f"{chart_name} · {chart_unit}")
                st.plotly_chart(make_dark_fig(chart), width="stretch", key="air_quality_trend_chart")

        history = st.session_state.get("air_quality_history", [])
        if history:
            st.markdown("#### Your AQI history")
            history_frame = pd.DataFrame(history)
            history_places = history_frame["Location"].drop_duplicates().tolist()
            visible_history_places = st.multiselect(
                "Show history for", history_places,
                default=[saved_air_quality["location"]] if saved_air_quality["location"] in history_places else history_places,
                key="air_quality_history_places",
            )
            visible_history = history_frame[history_frame["Location"].isin(visible_history_places)]
            if not visible_history.empty:
                history_chart = px.line(visible_history, x="Checked", y="U.S. AQI", color="Location", markers=True, height=330, title="AQI readings saved during this session")
                st.plotly_chart(make_dark_fig(history_chart), width="stretch", key="air_quality_history_chart")
            st.caption("History is kept for this browser session; it starts collecting after you check a place.")

        model_lat = air_quality.get("latitude", saved_air_quality["lat"])
        model_lon = air_quality.get("longitude", saved_air_quality["lon"])
        st.caption(f"Modeled estimate for the nearest grid cell ({float(model_lat):.3f}, {float(model_lon):.3f}); current model time: {current_air.get('time', 'unavailable')} {air_quality.get('timezone_abbreviation', '')}.")
    else:
        st.info("Search for a city above to fetch its air-quality estimate.")

    aqi_globe_rows = []
    saved_comparison = st.session_state.get("air_quality_comparison")
    if isinstance(saved_comparison, pd.DataFrame) and not saved_comparison.empty:
        for _, compared_place in saved_comparison.iterrows():
            coordinates = aqi_location_lookup.get(compared_place.get("Place"))
            score = pd.to_numeric(pd.Series([compared_place.get("U.S. AQI")]), errors="coerce").iloc[0]
            if coordinates and pd.notna(score):
                aqi_globe_rows.append({
                    "City": compared_place["Place"], "lat": coordinates[0], "lon": coordinates[1],
                    "U.S. AQI": float(score), "AQI number": f"{float(score):.0f}",
                })
    if selected_aqi_location and saved_air_quality:
        current_globe_aqi = saved_air_quality.get("data", {}).get("current", {}).get("us_aqi")
        if current_globe_aqi is not None and not pd.isna(current_globe_aqi):
            aqi_globe_rows = [row for row in aqi_globe_rows if row["City"] != selected_aqi_location]
            aqi_globe_rows.append({
                "City": selected_aqi_location, "lat": float(aqi_lat), "lon": float(aqi_lon),
                "U.S. AQI": float(current_globe_aqi), "AQI number": f"{float(current_globe_aqi):.0f}",
            })

    st.markdown("### 🌍 AQI globe")
    st.caption("Numbered dots are shown only for cities you have searched or compared. Values are modeled U.S. AQI estimates, not street-monitor readings.")
    if aqi_globe_rows:
        aqi_globe_frame = pd.DataFrame(aqi_globe_rows)
        aqi_globe = px.scatter_geo(
            aqi_globe_frame, lat="lat", lon="lon", color="U.S. AQI", text="AQI number",
            hover_name="City", hover_data={"lat": False, "lon": False, "U.S. AQI": ":.0f", "AQI number": False},
            color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444", "#9333ea"],
            range_color=[0, 200], projection="orthographic", height=460,
        )
        aqi_globe.update_traces(
            marker={"size": 26, "line": {"color": "#ffffff", "width": 1.5}},
            textposition="middle center", textfont={"color": "#102033", "size": 10},
        )
        aqi_globe.update_geos(
            showland=True, landcolor="#e9e5dc", showocean=True, oceancolor="#a9d3df",
            showcountries=True, countrycolor="#a9bcc5", showcoastlines=True,
            coastlinecolor="#78909c", bgcolor="rgba(0,0,0,0)",
        )
        aqi_globe.update_layout(
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar={"title": "U.S. AQI"},
        )
        st.plotly_chart(
            aqi_globe, width="stretch",
            config={"responsive": True, "displaylogo": False, "scrollZoom": True},
            key="air_quality_numbered_globe",
        )
    else:
        st.info("Search for a city and load its AQI, or compare places, to add numbered cities to the globe.")

    st.caption("This is a model-based estimate, not a ground-monitor reading. Forecast coverage is about 11 km in Europe and 45 km globally. Sources: [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) and [Copernicus Atmosphere Monitoring Service (CAMS)](https://atmosphere.copernicus.eu/).")

    st.markdown(
        "View NASA satellite observations directly on this dashboard. These "
        "products measure atmospheric columns or aerosols from space; they are "
        "not street-level air-quality readings."
    )

    atmosphere_products = [
        {
            "name": "Carbon monoxide (CO)",
            "instrument": "MOPITT · daily total column",
            "layer": "MOPITT_CO_Daily_Total_Column_Day",
            "description": "Maps CO through the atmospheric column. Wildfires and transport can create broad plumes.",
            "period": "Daily observations; standard product may lag by 3 weeks to 2 months",
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
            "description": "Satellite aerosol optical depth; not a direct PM₂.₅ or visibility measurement.",
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

    product_names = [product["name"] for product in atmosphere_products]
    selected_product_name = st.selectbox("NASA observation", product_names, key="nasa_atmosphere_product")
    selected_product = next(product for product in atmosphere_products if product["name"] == selected_product_name)
    atmosphere_lag_days = 60 if selected_product_name == "Carbon monoxide (CO)" else 30 if selected_product_name == "Carbon dioxide (CO₂)" else 7
    selected_observation_date = st.date_input(
        "Satellite observation date",
        value=datetime.date.today() - datetime.timedelta(days=atmosphere_lag_days),
        max_value=datetime.date.today(),
        key=f"nasa_atmosphere_date_v2_{product_names.index(selected_product_name)}",
        help="Choose an observation date. CO and CO₂ products can lag; daily satellite swaths and clouds leave gaps. A blank area may mean no valid retrieval rather than zero gas.",
    )
    st.markdown(f"### {selected_product['name']}")
    st.caption(f"{selected_product['instrument']} · {selected_product['period']}")
    st.write(selected_product["description"])
    if selected_product_name == "Carbon dioxide (CO₂)":
        st.info(
            "OCO-2 samples a narrow ground track and needs clear viewing conditions. "
            "At whole-world scale its coverage can look very sparse or nearly blank; "
            "blank areas do not mean zero CO₂. Use the map’s + control or pinch to zoom in."
        )
    render_nasa_atmosphere_map(selected_product["layer"], selected_observation_date.isoformat())
    st.caption(
        "NASA EOSDIS GIBS satellite visualization, shown for the selected observation date. "
        "Blank areas mean no valid retrieval was available for that date (for example, due to orbit coverage or clouds). "
        "The local proxy charts remain under Local Signal Proxies and are labeled separately."
    )

    st.markdown("### What this measurement means")
    st.markdown(
        "The NASA layer is satellite-observed data rendered by GIBS; it is not a "
        "model prediction or a value inferred from the local files. CO and CO₂ "
        "are column observations, so they do not represent the concentration "
        "at a person's breathing height. Aerosol optical depth describes how "
        "aerosols affect light through the atmospheric column, not ground-level PM₂.₅."
    )


# ============================================================
# TAB: LIVE EVENTS
# ============================================================

with tab_live_events:
    st.markdown("## 🚨 Live NASA EONET Events")
    st.caption("Browse NASA’s current open natural-event records. Use the filters and page controls to find events on mobile or desktop.")

    if live_events is None:
        show_eonet_connection_help(live_error)
    elif live_error:
        st.warning(f"⚠️ {live_error}")
    elif live_events.empty:
        st.info("No live events found.")

    else:
        scoped_events = filter_points_to_country_scope(
            live_events, selected_countries,
            country_scope_mode == "Selected countries", country_boundaries,
        )
        mapped_events = scoped_events.dropna(subset=["lat", "lon"])
        if not mapped_events.empty:
            map_view = st.radio(
                "Map view", ["Globe / flat map", "Interactive tile map", "Soft heat map"],
                index=2, horizontal=True, key="live_events_map_view",
            )
            if map_view != "Globe / flat map":
                render_leaflet_points(mapped_events, color_by="hazard", category_colors=HAZARD_COLORS,
                                      popup_columns=["title", "place", "hazard", "categories", "date"], height=460,
                                      thermal=map_view == "Soft heat map",
                                                                            key="map_eonet_live_events_leaflet")
            else:
                map_style = st.radio("Map style", ["Globe", "Flat world"], horizontal=True, key="live_events_map_style")
                projection = "orthographic" if map_style == "Globe" else "natural earth"
                fig = px.scatter_geo(
                    mapped_events,
                    lat="lat",
                    lon="lon",
                    color="hazard",
                    hover_name="title",
                    hover_data=[c for c in ["place", "categories", "date"] if c in mapped_events.columns],
                    color_discrete_map=HAZARD_COLORS,
                    projection=projection,
                    height=420,
                )
                fig = make_geo_fig(fig)
                st.plotly_chart(
                    fig,
                    width="stretch",
                    config={"responsive": True, "displaylogo": False, "scrollZoom": True},
                    key="map_eonet_live_events",
                )
        else:
            st.info("No event locations are available in the selected countries.")
        show_eonet_event_browser(scoped_events, key="live_events_browser")


# ============================================================
# TAB: LIVE FIRE
# ============================================================

with tab_live_fire:
    st.markdown("## 🔥 Live Fire")
    st.markdown("### Live satellite hotspots · NASA FIRMS")
    st.caption("This is the live detection map. Orange/red dots are satellite hotspots; zoom in or click one for details. These are not flame icons or fire boundaries.")

    if firms_df is None:
        st.info(firms_error)
    else:
        firms_map_df = filter_points_to_country_scope(
            firms_df, selected_countries,
            country_scope_mode == "Selected countries", country_boundaries,
        )
        if firms_is_live:
            st.success(f"Loaded {len(firms_df):,} live fire detections from NASA FIRMS ({firms_source_used}).")
            if firms_error:
                st.info(firms_error)
        elif firms_url_loaded and firms_data_is_current:
            st.info(f"Loaded {len(firms_df):,} detections from the configured CSV URL. {firms_error}")
        else:
            st.warning(firms_error)
            st.caption(firms_freshness_note)
            st.caption("Historical detections are available in the records table, but are not drawn as current fire activity.")
            firms_map_df = firms_map_df.iloc[0:0].copy()
        if firms_map_df.empty:
            if firms_data_is_current:
                st.info("No NASA FIRMS detections are in the selected countries for this feed period.")

        map_view = st.radio(
            "Map view", ["Globe / flat map", "Interactive tile map", "Soft heat map"],
            index=2, horizontal=True, key="firms_map_view",
        )
        if map_view != "Globe / flat map":
            if map_view == "Soft heat map":
                render_leaflet_points(firms_map_df, color_by="frp", height=460, point_limit=2000,
                                      thermal=True,
                                      pixel_size_meters=1000 if firms_source_used == "MODIS_NRT" else 375,
                                      key="map_firms_live_thermal_leaflet")
            else:
                render_leaflet_points(firms_map_df, color_by="frp", height=460, point_limit=2000,
                                      pixel_size_meters=1000 if firms_source_used == "MODIS_NRT" else 375,
                                      key="map_firms_live_leaflet")
        else:
            map_style = st.radio("Map style", ["Globe", "Flat world"], horizontal=True, key="firms_map_style")
            projection = "orthographic" if map_style == "Globe" else "natural earth"
            fig = make_thermal_fire_map(firms_map_df, projection=projection, height=460)
            st.plotly_chart(
                fig,
                width="stretch",
                config={"scrollZoom": True, "responsive": True, "displayModeBar": True, "displaylogo": False},
                key="map_firms_live",
            )
        if map_view == "Soft heat map":
            style_label = "Soft heat map"
            st.caption(f"{style_label} colors supported clusters of NASA FIRMS fire detections by relative satellite fire radiative power (FRP), not ground temperature or fire perimeters.")
        else:
            st.caption("NASA FIRMS detections are shown as orange/red points. A detection does not confirm a wildfire; use the table for the full feed.")

        if "date" in firms_df.columns:
            timeline = (
                firms_df.groupby(pd.Grouper(key="date", freq="D"))
                .size()
                .reset_index(name="fire_count")
            )
            fig = px.line(timeline, x="date", y="fire_count", height=380)
            fig = make_dark_fig(fig)
            st.plotly_chart(fig, width="stretch", key="chart_firms_timeline")

        with st.expander("View fire detection records"):
            st.dataframe(firms_df.head(1000), width="stretch")

    st.markdown("### NASA EONET fire reports")
    st.caption("EONET is a curated event list, separate from FIRMS satellite detections. Use the Live Events tab for event maps and full details.")
    if live_events is None:
        st.info(live_error)
    elif live_error:
        st.warning(live_error)
    else:
        fire_events = live_events[live_events["hazard"] == "Fire"].copy()
        if fire_events.empty:
            st.info("No open EONET fire reports in this response.")
        else:
            with st.expander("View EONET fire records"):
                st.dataframe(fire_events, width="stretch")


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
        source_label = firms_display_source
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
        - `output/risk_model.keras`
        - `output/model_metadata.pkl`
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
