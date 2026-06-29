import json
import subprocess
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "car_price_model.pkl"
METRICS_PATH = PROJECT_DIR / "model_metrics.json"
DATA_PATH = PROJECT_DIR / "used_cars.csv"


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
def load_listing_data():
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


def count_same_cars_on_sale(listings, car_make, car_model, manufacturing_year):
    """Count listings with the same make, model, and manufacturing year."""
    if listings is None:
        return None

    same_cars = listings[
        (listings["make"] == car_make)
        & (listings["model"] == car_model)
        & (listings["year"] == int(manufacturing_year))
    ]

    return len(same_cars)


def estimate_time_to_sell_days(listings, car_make, car_model, manufacturing_year):
    """Estimate sale time from how long similar active listings have been online."""
    if listings is None or "listing_date" not in listings.columns:
        return None

    listings_with_dates = listings.dropna(subset=["listing_date"]).copy()
    if listings_with_dates.empty:
        return None

    today = pd.Timestamp(datetime.now().date())
    match_options = [
        (
            "same make, model, and year",
            listings_with_dates[
                (listings_with_dates["make"] == car_make)
                & (listings_with_dates["model"] == car_model)
                & (listings_with_dates["year"] == int(manufacturing_year))
            ],
        ),
        (
            "same make and model",
            listings_with_dates[
                (listings_with_dates["make"] == car_make)
                & (listings_with_dates["model"] == car_model)
            ],
        ),
        ("all cars with listing dates", listings_with_dates),
    ]

    for match_description, matching_rows in match_options:
        if matching_rows.empty:
            continue

        listing_ages = (today - matching_rows["listing_date"]).dt.days
        listing_ages = listing_ages[listing_ages >= 0]
        if listing_ages.empty:
            continue

        return {
            "days": int(round(float(listing_ages.median()))),
            "sample_size": int(len(listing_ages)),
            "basis": match_description,
        }

    return None


def show_model_metrics(metrics):
    if not metrics:
        st.markdown(
            '<div class="section-heading">Model Evaluation Metrics</div>',
            unsafe_allow_html=True,
        )
        st.info("Metrics will appear here after you train the model.")
        return

    cars_on_sale_count = metrics.get("cars_on_sale_count")
    rows_after_cleaning = metrics.get("rows_after_cleaning")
    best_metrics = metrics["best_model_metrics"]

    st.markdown(
        '<div class="section-heading">Market And Model Snapshot</div>',
        unsafe_allow_html=True,
    )

    if cars_on_sale_count is not None:
        sale_column, training_column, model_column, mae_column = st.columns(4)
        with sale_column:
            render_metric_card(
                "Listings in dataset",
                f"{cars_on_sale_count:,}",
                "Current CSV rows",
            )
        if rows_after_cleaning is not None:
            with training_column:
                render_metric_card(
                    "Training rows",
                    f"{rows_after_cleaning:,}",
                    "After cleaning",
                )
        with model_column:
            render_metric_card(
                "Best model",
                metrics["best_model"],
                "Selected by MAE",
            )
        with mae_column:
            render_metric_card(
                "MAE",
                format_usd(best_metrics["mae"], decimals=0),
                "Typical absolute error",
            )

    all_metrics_table = []
    for model_name, values in metrics["all_model_metrics"].items():
        all_metrics_table.append(
            {
                "Model": model_name,
                "MAE": values["mae"],
                "RMSE": values["rmse"],
                "R2 Score": values["r2_score"],
            }
        )

    with st.expander("Compare trained models", expanded=False):
        st.dataframe(
            pd.DataFrame(all_metrics_table),
            hide_index=True,
            width="stretch",
        )


def create_prediction_input(car_make, car_model, year, mileage_km):
    """Create one input row in the same format used during training."""
    return pd.DataFrame(
        [
            {
                "make": car_make,
                "model": car_model,
                "year": int(year),
                "mileage_km": float(mileage_km),
            }
        ]
    )


def predict_price(model, car_make, car_model, year, mileage_km):
    """Predict a non-negative vehicle price."""
    input_data = create_prediction_input(car_make, car_model, year, mileage_km)
    predicted_price = float(model.predict(input_data)[0])
    return max(predicted_price, 0)


def create_price_projection(
    model,
    car_make,
    car_model,
    manufacturing_year,
    current_mileage_km,
    expected_monthly_mileage,
):
    """Project prices for each remaining month by increasing mileage."""
    today = datetime.now()
    projection_rows = []

    for month_number in range(today.month, 13):
        months_ahead = month_number - today.month
        projected_mileage = current_mileage_km + (
            expected_monthly_mileage * months_ahead
        )
        predicted_price = predict_price(
            model,
            car_make,
            car_model,
            manufacturing_year,
            projected_mileage,
        )

        projection_rows.append(
            {
                "month": datetime(today.year, month_number, 1).strftime("%B"),
                "projected_mileage_km": round(float(projected_mileage), 0),
                "predicted_price_usd": round(float(predicted_price), 2),
            }
        )

    return pd.DataFrame(projection_rows)


def render_training_log(log_placeholder, log_lines):
    """Render training logs inside a fixed-height scrollable panel."""
    log_text = escape("\n".join(log_lines[-400:]))
    log_placeholder.markdown(
        f"""
        <div class="training-log">
            <pre>{log_text}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_training_with_live_logs(
    log_placeholder,
    max_listings,
    start_page,
    min_delay,
    max_delay,
    allow_partial_scrape,
):
    """Run training and stream terminal output into the Streamlit page."""
    command = [
        sys.executable,
        "-u",
        "train_model.py",
        "used_cars.csv",
        "--max-listings",
        str(int(max_listings)),
        "--start-page",
        str(int(start_page)),
        "--scrape-min-delay",
        str(float(min_delay)),
        "--scrape-max-delay",
        str(float(max_delay)),
    ]
    if allow_partial_scrape:
        command.append("--allow-partial-scrape")

    log_lines = [
        "$ " + " ".join(command),
        "Existing CSV will be cleared before scraping fresh listings.",
        f"Scraper target: up to {int(max_listings):,} listings.",
        f"Starting AUTO.RIA page: {int(start_page)}.",
        f"Delay between pages: {float(min_delay):.1f}-{float(max_delay):.1f} seconds.",
        "Starting model training...",
    ]
    render_training_log(log_placeholder, log_lines)

    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is not None:
        for line in process.stdout:
            log_lines.append(line.rstrip())
            render_training_log(log_placeholder, log_lines)

    return_code = process.wait()
    if return_code == 0:
        log_lines.append("Training process finished successfully.")
    else:
        log_lines.append(f"Training process failed with exit code {return_code}.")

    render_training_log(log_placeholder, log_lines)
    return return_code


def clear_cached_project_data():
    load_model.clear()
    load_metrics.clear()
    load_listing_data.clear()


def format_usd(value, decimals=0):
    """Format a number as a compact USD value."""
    return f"USD {float(value):,.{decimals}f}"


def apply_app_styles():
    st.markdown(
        """
        <style>
            :root {
                --app-bg: #f5f7fa;
                --panel: #ffffff;
                --ink: #15171a;
                --muted: #667085;
                --line: #d9e1ea;
                --green: #087f5b;
                --green-soft: #e8f6f0;
                --gold: #b7791f;
                --gold-soft: #fff4dc;
                --blue: #2563eb;
                --blue-soft: #eaf1ff;
            }

            [data-testid="stAppViewContainer"] {
                background: var(--app-bg);
            }

            [data-testid="stHeader"] {
                background: rgba(245, 247, 250, 0.88);
                backdrop-filter: blur(12px);
            }

            .block-container {
                max-width: 1180px;
                padding-top: 2rem;
                padding-bottom: 3rem;
            }

            h1, h2, h3, p, label, span {
                letter-spacing: 0;
            }

            .hero-panel {
                background: linear-gradient(135deg, #121816 0%, #17433f 58%, #76551c 100%);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 18px;
                color: #ffffff;
                padding: 32px;
                box-shadow: 0 22px 55px rgba(20, 30, 38, 0.18);
                margin-bottom: 18px;
            }

            .hero-eyebrow {
                color: #c7f3df;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 8px;
            }

            .hero-title {
                font-size: clamp(2rem, 4vw, 3.4rem);
                font-weight: 780;
                line-height: 1.03;
                margin: 0;
                letter-spacing: 0;
            }

            .hero-copy {
                max-width: 760px;
                color: #d8efe8;
                font-size: 1.05rem;
                line-height: 1.65;
                margin: 16px 0 0;
            }

            .hero-stats {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 12px;
                margin-top: 26px;
            }

            .hero-stat {
                background: rgba(255, 255, 255, 0.11);
                border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 12px;
                padding: 14px 16px;
            }

            .hero-stat span {
                display: block;
                color: #c7d2d9;
                font-size: 0.78rem;
                font-weight: 700;
                text-transform: uppercase;
            }

            .hero-stat strong {
                display: block;
                color: #ffffff;
                font-size: 1.15rem;
                margin-top: 4px;
                overflow-wrap: anywhere;
            }

            .notice-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 14px;
                margin: 8px 0 26px;
            }

            .notice-card {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 16px 18px;
                box-shadow: 0 12px 30px rgba(18, 24, 38, 0.06);
            }

            .notice-card strong {
                color: var(--ink);
                display: block;
                font-size: 0.94rem;
                margin-bottom: 4px;
            }

            .notice-card span {
                color: var(--muted);
                font-size: 0.9rem;
                line-height: 1.5;
            }

            .section-heading {
                color: var(--ink);
                font-size: 1.35rem;
                font-weight: 760;
                margin: 28px 0 12px;
            }

            .panel-title {
                color: var(--ink);
                font-size: 1rem;
                font-weight: 740;
                margin: 0 0 12px;
            }

            .metric-card {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 17px 18px;
                min-height: 116px;
                box-shadow: 0 12px 30px rgba(18, 24, 38, 0.06);
                margin-bottom: 12px;
            }

            .metric-card.primary {
                background: var(--green-soft);
                border-color: rgba(8, 127, 91, 0.24);
            }

            .metric-card.blue {
                background: var(--blue-soft);
                border-color: rgba(37, 99, 235, 0.2);
            }

            .metric-card.gold {
                background: var(--gold-soft);
                border-color: rgba(183, 121, 31, 0.24);
            }

            .metric-label {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 740;
                text-transform: uppercase;
                margin-bottom: 8px;
            }

            .metric-value {
                color: var(--ink);
                font-size: clamp(1.28rem, 2vw, 1.75rem);
                font-weight: 780;
                line-height: 1.15;
                overflow-wrap: anywhere;
            }

            .metric-note {
                color: var(--muted);
                font-size: 0.85rem;
                margin-top: 8px;
                line-height: 1.45;
            }

            div[data-testid="stForm"] {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 20px 20px 8px;
                box-shadow: 0 12px 30px rgba(18, 24, 38, 0.06);
            }

            div[data-testid="stTextInput"] input,
            div[data-testid="stNumberInput"] input {
                border-radius: 10px;
                border-color: #c9d3df;
            }

            .stButton button {
                border-radius: 10px;
                min-height: 46px;
                font-weight: 740;
                letter-spacing: 0;
            }

            .stDataFrame {
                border-radius: 14px;
                overflow: hidden;
            }

            .training-log {
                background: #111827;
                border: 1px solid #263244;
                border-radius: 14px;
                height: 420px;
                overflow-y: auto;
                padding: 16px;
                box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.03);
            }

            .training-log pre {
                color: #dbeafe;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
                font-size: 0.84rem;
                line-height: 1.52;
                margin: 0;
                white-space: pre-wrap;
                overflow-wrap: anywhere;
            }

            @media (max-width: 760px) {
                .hero-panel {
                    padding: 24px;
                    border-radius: 14px;
                }

                .hero-stats,
                .notice-grid {
                    grid-template-columns: 1fr;
                }

                .metric-card {
                    min-height: auto;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, note="", tone=""):
    tone_class = f" {tone}" if tone else ""
    note_markup = f'<div class="metric-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="metric-card{tone_class}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {note_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(metrics, show_model_stats=False):
    if show_model_stats and metrics:
        listings = f"{metrics.get('cars_on_sale_count', 0):,}"
        best_model = metrics.get("best_model", "Trained model")
        mae = metrics.get("best_model_metrics", {}).get("mae")
        mae_value = format_usd(mae, decimals=0) if mae is not None else "Available"
        stat_labels = ("Listings", "Model", "Typical Error")
        stat_values = (listings, best_model, mae_value)
    else:
        stat_labels = ("Inputs", "Estimate", "Projection")
        stat_values = ("Make + model", "USD price", "Through December")

    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-eyebrow">AI Fundamentals final project</div>
            <h1 class="hero-title">Used Car Price Estimator for Ukraine</h1>
            <p class="hero-copy">
                A cleaner market dashboard for estimating a used car price from
                make, model, year, mileage, and expected monthly driving.
            </p>
            <div class="hero-stats">
                <div class="hero-stat">
                    <span>{stat_labels[0]}</span>
                    <strong>{stat_values[0]}</strong>
                </div>
                <div class="hero-stat">
                    <span>{stat_labels[1]}</span>
                    <strong>{stat_values[1]}</strong>
                </div>
                <div class="hero-stat">
                    <span>{stat_labels[2]}</span>
                    <strong>{stat_values[2]}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_notices():
    st.markdown(
        """
        <div class="notice-grid">
            <div class="notice-card">
                <strong>Educational estimate</strong>
                <span>The prediction is not a guaranteed market price.</span>
            </div>
            <div class="notice-card">
                <strong>Projection limit</strong>
                <span>The future graph is a mileage-based model projection, not a guaranteed forecast.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_file_size(path):
    if not path.exists():
        return "Missing"

    size = float(path.stat().st_size)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024


def render_training_section(metrics):
    st.markdown(
        '<div class="section-heading">Model Training</div>',
        unsafe_allow_html=True,
    )

    training_control_column, training_log_column = st.columns(
        [1, 1.6],
        gap="large",
    )

    with training_control_column:
        st.markdown(
            '<div class="panel-title">Clear, scrape, and train</div>',
            unsafe_allow_html=True,
        )
        default_listing_target = int(
            metrics.get("cars_on_sale_count", 20000) if metrics else 20000
        )
        suggested_listing_target = 50000
        max_listings = st.number_input(
            "Maximum AUTO.RIA listings",
            min_value=100,
            max_value=50000,
            value=suggested_listing_target,
            step=100,
            help=(
                "The scraper will clear the existing CSV, then collect fresh "
                "listings until this target is reached before training."
            ),
        )
        st.caption(
            f"Current CSV has about {default_listing_target:,} listings. "
            "Training rebuilds it from blank."
        )
        start_page = st.number_input(
            "AUTO.RIA start page",
            min_value=1,
            max_value=5000,
            value=1,
            step=1,
            help=(
                "Page 1 starts from the newest listings and rebuilds the dataset "
                "from the beginning."
            ),
        )
        delay_range = st.slider(
            "Delay between scraper pages",
            min_value=0.0,
            max_value=5.0,
            value=(0.3, 0.8),
            step=0.1,
            help=(
                "Lower values are faster, but very aggressive scraping can be "
                "blocked by the website."
            ),
        )
        allow_partial_scrape = st.checkbox(
            "Train even if fewer listings are collected",
            value=True,
            help=(
                "Keep this enabled when using this from the UI. It lets training "
                "continue if AUTO.RIA returns fewer listings than the selected "
                "maximum."
            ),
        )
        render_metric_card(
            "Training command",
            "Fresh scrape + train",
            f"Target: {int(max_listings):,} listings, page {int(start_page)}+",
            tone="blue",
        )
        train_clicked = st.button(
            "Train Model",
            type="primary",
            width="stretch",
        )

    with training_log_column:
        st.markdown(
            '<div class="panel-title">Training logs</div>',
            unsafe_allow_html=True,
        )
        training_log_placeholder = st.empty()
        render_training_log(
            training_log_placeholder,
            ["Training logs will appear here when you start a run."],
        )

    if train_clicked:
        with st.spinner("Training model..."):
            training_exit_code = run_training_with_live_logs(
                training_log_placeholder,
                max_listings,
                start_page,
                delay_range[0],
                delay_range[1],
                allow_partial_scrape,
            )

        if training_exit_code == 0:
            clear_cached_project_data()
            metrics = load_metrics()
            st.success(
                "Training finished. The app is now using the updated model files."
            )
        else:
            st.error("Training failed. Check the log output above.")

    return metrics


def render_estimator_page(metrics):
    render_notices()

    if not MODEL_PATH.exists():
        st.error(
            "The trained model file was not found. Open Model information and "
            "train the model first."
        )
        st.code("python3 train_model.py used_cars.csv --max-listings 20000")
        st.stop()

    model = load_model()
    listing_data = load_listing_data()
    current_year = datetime.now().year

    st.markdown(
        '<div class="section-heading">Estimate A Car</div>',
        unsafe_allow_html=True,
    )

    input_column, guide_column = st.columns([1.45, 1], gap="large")

    with input_column:
        st.markdown(
            '<div class="panel-title">Vehicle details</div>',
            unsafe_allow_html=True,
        )
        with st.form("prediction_form"):
            make_column, model_column = st.columns(2)
            with make_column:
                car_make_input = st.text_input(
                    "Car make",
                    placeholder="Toyota",
                )
            with model_column:
                car_model_input = st.text_input(
                    "Car model",
                    placeholder="Corolla",
                )

            year_column, mileage_column = st.columns(2)
            with year_column:
                manufacturing_year = st.number_input(
                    "Manufacturing year",
                    min_value=1990,
                    max_value=current_year,
                    value=min(2015, current_year),
                    step=1,
                )
            with mileage_column:
                mileage_km = st.number_input(
                    "Current mileage in kilometers",
                    min_value=0,
                    value=100000,
                    step=1000,
                )

            expected_monthly_mileage = st.number_input(
                "Expected monthly mileage in kilometers",
                min_value=0,
                value=1000,
                step=100,
            )

            submitted = st.form_submit_button(
                "Estimate Price",
                type="primary",
                width="stretch",
            )

    with guide_column:
        st.markdown(
            '<div class="panel-title">Estimator guide</div>',
            unsafe_allow_html=True,
        )
        render_metric_card(
            "Typed inputs",
            "Flexible",
            "Make and model are typed manually",
            tone="blue",
        )
        render_metric_card(
            "Mileage",
            "Important",
            "Current mileage drives both estimate and projection",
        )
        render_metric_card(
            "Projection",
            "December",
            "Monthly mileage is added through year end",
            tone="gold",
        )

    if submitted:
        car_make = normalize_text(car_make_input)
        car_model = normalize_text(car_model_input)

        validation_errors = []
        if not car_make:
            validation_errors.append("Make should not be empty.")
        if not car_model:
            validation_errors.append("Model should not be empty.")
        if manufacturing_year < 1990 or manufacturing_year > current_year:
            validation_errors.append(
                f"Year should be between 1990 and {current_year}."
            )
        if mileage_km < 0:
            validation_errors.append("Current mileage should not be negative.")
        if expected_monthly_mileage < 0:
            validation_errors.append("Expected monthly mileage should not be negative.")

        if validation_errors:
            for error in validation_errors:
                st.error(error)
            return

        predicted_price = predict_price(
            model,
            car_make,
            car_model,
            manufacturing_year,
            mileage_km,
        )

        same_cars_count = count_same_cars_on_sale(
            listing_data,
            car_make,
            car_model,
            manufacturing_year,
        )
        sale_time_estimate = estimate_time_to_sell_days(
            listing_data,
            car_make,
            car_model,
            manufacturing_year,
        )

        result_columns = st.columns(3)
        with result_columns[0]:
            render_metric_card(
                "Estimated average price",
                format_usd(predicted_price, decimals=0),
                "Current model estimate",
                tone="primary",
            )
        with result_columns[1]:
            if same_cars_count is None:
                render_metric_card(
                    "Similar active listings",
                    "Unavailable",
                    "CSV could not be loaded",
                    tone="blue",
                )
            else:
                render_metric_card(
                    "Similar active listings",
                    f"{same_cars_count:,}",
                    "Same make, model, and year",
                    tone="blue",
                )
        with result_columns[2]:
            if sale_time_estimate is None:
                render_metric_card(
                    "Estimated sale time",
                    "Unavailable",
                    "Listing dates unavailable",
                    tone="gold",
                )
            else:
                render_metric_card(
                    "Estimated sale time",
                    f"{sale_time_estimate['days']} days",
                    f"{sale_time_estimate['sample_size']} listings sampled",
                    tone="gold",
                )

        if sale_time_estimate is None:
            st.caption(
                "Estimated selling time is unavailable because listing dates "
                "are not available in the CSV file."
            )
        else:
            st.caption(
                "This is based on the median age of active listings using "
                f"{sale_time_estimate['basis']} "
                f"({sale_time_estimate['sample_size']} listings). It is not "
                "confirmed sold-time data."
            )

        projection = create_price_projection(
            model,
            car_make,
            car_model,
            manufacturing_year,
            mileage_km,
            expected_monthly_mileage,
        )

        st.markdown(
            '<div class="section-heading">Projected Price Through December</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            projection.rename(
                columns={
                    "month": "Month",
                    "projected_mileage_km": "Projected Mileage (km)",
                    "predicted_price_usd": "Predicted Price (USD)",
                }
            ),
            hide_index=True,
            width="stretch",
        )

        chart_data = projection.set_index("month")["predicted_price_usd"].rename(
            "Predicted Price (USD)"
        )
        st.line_chart(chart_data)

    st.caption(
        "Unknown typed make or model values can still be estimated, but accuracy may be lower."
    )


def render_model_information_page(metrics):
    metrics = render_training_section(metrics)

    st.markdown(
        '<div class="section-heading">Project Files</div>',
        unsafe_allow_html=True,
    )
    model_column, data_column, metrics_column = st.columns(3)
    with model_column:
        render_metric_card(
            "Model file",
            format_file_size(MODEL_PATH),
            "car_price_model.pkl",
            tone="blue",
        )
    with data_column:
        render_metric_card(
            "Dataset file",
            format_file_size(DATA_PATH),
            "used_cars.csv",
        )
    with metrics_column:
        render_metric_card(
            "Metrics file",
            "Ready" if METRICS_PATH.exists() else "Missing",
            "model_metrics.json",
            tone="gold",
        )

    show_model_metrics(metrics)


st.set_page_config(
    page_title="Used Car Price Estimator for Ukraine",
    layout="wide",
)

apply_app_styles()
metrics = load_metrics()

selected_page = st.sidebar.radio(
    "Page",
    ("Estimate price", "Model information"),
)
st.sidebar.caption("Use Model information to retrain and inspect the model.")

render_hero(metrics, show_model_stats=selected_page == "Model information")

if selected_page == "Estimate price":
    render_estimator_page(metrics)
else:
    render_model_information_page(metrics)
