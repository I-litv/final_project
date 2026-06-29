import streamlit as st

from services.data import count_csv_rows, get_data_file_signature, load_statistics_data
from services.formatting import format_price_range
from services.statistics import build_statistics_table
from ui.components import display_statistics_table, render_metric_card


def render_car_statistics_page(metrics):
    data_signature = get_data_file_signature()
    csv_row_count = count_csv_rows(data_signature)
    listings = load_statistics_data(data_signature)

    st.markdown(
        '<div class="section-heading">Car Statistics</div>',
        unsafe_allow_html=True,
    )

    if listings is None:
        st.info("Car statistics will appear after a valid CSV is available.")
        return

    last_trained_rows = metrics.get("cars_on_sale_count") if metrics else None
    if last_trained_rows is not None and int(last_trained_rows) != int(csv_row_count):
        st.warning(
            f"Current CSV has {csv_row_count:,} rows, while the last trained "
            f"model metrics were created from {int(last_trained_rows):,} rows. "
            "The statistics below use the current CSV."
        )

    summary_columns = st.columns(3)
    with summary_columns[0]:
        render_metric_card(
            "Rows in CSV",
            f"{csv_row_count:,}",
            "Current used_cars.csv",
            tone="blue",
        )
    with summary_columns[1]:
        render_metric_card(
            "Rows used for stats",
            f"{len(listings):,}",
            "Valid rows after filtering",
        )
    with summary_columns[2]:
        render_metric_card(
            "Overall price range",
            format_price_range(
                listings["price_usd"].min(),
                listings["price_usd"].max(),
            ),
            "Minimum to maximum",
            tone="gold",
        )

    st.markdown(
        '<div class="section-heading">Top 20 Cars On Sale</div>',
        unsafe_allow_html=True,
    )
    top_cars = build_statistics_table(listings, ["make", "model"], limit=20)
    display_statistics_table(
        top_cars.rename(
            columns={
                "make": "Make",
                "model": "Model",
            }
        )
    )

    st.markdown(
        '<div class="section-heading">Listings By Fuel</div>',
        unsafe_allow_html=True,
    )
    fuel_table = build_statistics_table(listings, ["fuel"])
    display_statistics_table(
        fuel_table.rename(
            columns={
                "fuel": "Fuel",
            }
        )
    )

    st.markdown(
        '<div class="section-heading">Top 20 Make, Model, And Year Groups</div>',
        unsafe_allow_html=True,
    )
    make_year_table = build_statistics_table(
        listings,
        ["make", "model", "year"],
        limit=20,
    )
    display_statistics_table(
        make_year_table.rename(
            columns={
                "make": "Make",
                "model": "Model",
                "year": "Year",
            }
        )
    )
