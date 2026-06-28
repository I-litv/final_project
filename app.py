import json
from datetime import datetime
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
        st.subheader("Model Evaluation Metrics")
        st.info("Metrics will appear here after you train the model.")
        return

    cars_on_sale_count = metrics.get("cars_on_sale_count")
    rows_after_cleaning = metrics.get("rows_after_cleaning")
    if cars_on_sale_count is not None:
        st.subheader("Dataset Summary")
        sale_column, training_column = st.columns(2)
        sale_column.metric(
            "Cars currently on sale in dataset",
            f"{cars_on_sale_count:,}",
        )
        if rows_after_cleaning is not None:
            training_column.metric(
                "Cars used for model training",
                f"{rows_after_cleaning:,}",
            )

        st.caption(
            "This count is based on the number of listing rows in the CSV "
            "dataset used for training."
        )

    st.subheader("Model Evaluation Metrics")
    best_metrics = metrics["best_model_metrics"]
    st.write(f"Best model: {metrics['best_model']}")

    mae_column, rmse_column, r2_column = st.columns(3)
    mae_column.metric("MAE", f"USD {best_metrics['mae']:,.2f}")
    rmse_column.metric("RMSE", f"USD {best_metrics['rmse']:,.2f}")
    r2_column.metric("R2 Score", f"{best_metrics['r2_score']:.4f}")

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

    st.dataframe(pd.DataFrame(all_metrics_table), hide_index=True)


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


st.set_page_config(
    page_title="Used Car Price Estimator for Ukraine",
)

st.title("Used Car Price Estimator for Ukraine")
st.write(
    "This simple machine learning app estimates the average used car price "
    "in Ukraine from the make, model, manufacturing year, and mileage."
)

st.warning(
    "This is an educational estimate for an AI Fundamentals project. "
    "It is not a guaranteed market price."
)
st.warning(
    "The future price graph is a simple projection based mainly on mileage "
    "increase, not a guaranteed market forecast."
)

if not MODEL_PATH.exists():
    st.error(
        "The trained model file was not found. Train the model first by running:"
    )
    st.code("python3 train_model.py path/to/used_cars.csv")
    show_model_metrics(load_metrics())
    st.stop()

model = load_model()
listing_data = load_listing_data()
current_year = datetime.now().year

car_make_input = st.text_input("Car make", placeholder="Example: Toyota")
car_model_input = st.text_input("Car model", placeholder="Example: Corolla")
manufacturing_year = st.number_input(
    "Manufacturing year",
    min_value=1990,
    max_value=current_year,
    value=min(2015, current_year),
    step=1,
)
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

if st.button("Predict Price"):
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
    else:
        predicted_price = predict_price(
            model,
            car_make,
            car_model,
            manufacturing_year,
            mileage_km,
        )

        st.success(f"Current estimated average price: USD {predicted_price:,.0f}")
        same_cars_count = count_same_cars_on_sale(
            listing_data,
            car_make,
            car_model,
            manufacturing_year,
        )
        if same_cars_count is None:
            st.info(
                "Same-car listing count is unavailable because the CSV file "
                "could not be loaded."
            )
        else:
            st.info(
                "Same cars currently on sale in dataset: "
                f"{same_cars_count:,}"
            )

        sale_time_estimate = estimate_time_to_sell_days(
            listing_data,
            car_make,
            car_model,
            manufacturing_year,
        )
        if sale_time_estimate is None:
            st.info(
                "Estimated selling time is unavailable because listing dates "
                "are not available in the CSV file."
            )
        else:
            st.info(
                "Estimated time to sell if listed today: about "
                f"{sale_time_estimate['days']} days."
            )
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

        st.subheader("Projected Price Until The End Of The Current Year")
        st.dataframe(
            projection.rename(
                columns={
                    "month": "Month",
                    "projected_mileage_km": "Projected Mileage (km)",
                    "predicted_price_usd": "Predicted Price (USD)",
                }
            ),
            hide_index=True,
        )

        chart_data = projection.set_index("month")["predicted_price_usd"]
        st.line_chart(chart_data)

st.info(
    "If the typed make or model was not present in the training data, "
    "the prediction may be less accurate."
)

show_model_metrics(load_metrics())
