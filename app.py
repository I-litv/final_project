import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "car_price_model.pkl"
METRICS_PATH = PROJECT_DIR / "model_metrics.json"


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


def show_model_metrics(metrics):
    st.subheader("Model Evaluation Metrics")

    if not metrics:
        st.info("Metrics will appear here after you train the model.")
        return

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
