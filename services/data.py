import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "car_price_model.pkl"
METRICS_PATH = PROJECT_DIR / "model_metrics.json"
DATA_PATH = PROJECT_DIR / "used_cars.csv"


def get_data_file_signature():
    """Return a small file signature so cached CSV loaders refresh after writes."""
    if not DATA_PATH.exists():
        return None

    file_stat = DATA_PATH.stat()
    return (file_stat.st_mtime_ns, file_stat.st_size)


def normalize_text(value):
    """Match the text cleaning used during training."""
    return str(value).lower().strip()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_listing_data(file_signature=None):
    """Load the CSV listings so the app can count matching cars."""
    if not DATA_PATH.exists():
        return None

    listings = pd.read_csv(DATA_PATH)
    required_columns = ["make", "model", "year"]
    if not set(required_columns).issubset(listings.columns):
        return None

    optional_columns = ["listing_date"] if "listing_date" in listings.columns else []
    listings = listings[required_columns + optional_columns].copy()
    listings["make"] = listings["make"].apply(normalize_text)
    listings["model"] = listings["model"].apply(normalize_text)
    listings["year"] = pd.to_numeric(listings["year"], errors="coerce")
    listings = listings.dropna(subset=["make", "model", "year"])
    listings["year"] = listings["year"].astype(int)

    if "listing_date" in listings.columns:
        listings["listing_date"] = pd.to_datetime(
            listings["listing_date"],
            errors="coerce",
        )

    return listings


@st.cache_data
def load_statistics_data(file_signature=None):
    """Load clean listing fields needed for market statistics."""
    if not DATA_PATH.exists():
        return None

    listings = pd.read_csv(DATA_PATH)
    required_columns = ["make", "model", "year", "price_usd"]
    if not set(required_columns).issubset(listings.columns):
        return None

    if "fuel" not in listings.columns:
        listings["fuel"] = "Unknown"

    listings = listings[required_columns + ["fuel"]].copy()
    for column in ["make", "model", "fuel"]:
        listings[column] = listings[column].fillna("Unknown").astype(str).str.strip()
        listings.loc[listings[column] == "", column] = "Unknown"

    listings["year"] = pd.to_numeric(listings["year"], errors="coerce")
    listings["price_usd"] = pd.to_numeric(listings["price_usd"], errors="coerce")
    listings = listings.dropna(subset=["make", "model", "year", "price_usd"])
    listings = listings[listings["price_usd"] > 0].copy()

    if listings.empty:
        return None

    listings["year"] = listings["year"].astype(int)
    listings["price_usd"] = listings["price_usd"].astype(float)
    return listings


@st.cache_data
def count_csv_rows(file_signature=None):
    if not DATA_PATH.exists():
        return 0

    with open(DATA_PATH, "r", encoding="utf-8-sig") as file:
        return max(sum(1 for _ in file) - 1, 0)


def format_file_size(path):
    if not path.exists():
        return "Missing"

    size = float(path.stat().st_size)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024


def clear_cached_project_data():
    load_model.clear()
    load_metrics.clear()
    load_listing_data.clear()
    load_statistics_data.clear()
    count_csv_rows.clear()
