from datetime import datetime

import pandas as pd


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
