from datetime import datetime

import streamlit as st

from services.data import MODEL_PATH, get_data_file_signature, load_listing_data
from services.data import load_model, normalize_text
from services.formatting import format_usd
from services.prediction import count_same_cars_on_sale, create_price_projection
from services.prediction import estimate_time_to_sell_days, predict_price
from ui.components import render_metric_card, render_notices


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
    data_signature = get_data_file_signature()
    listing_data = load_listing_data(data_signature)
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
