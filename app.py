import streamlit as st

from services.data import load_metrics
from ui.components import render_hero
from ui.sidebar import render_sidebar
from ui.styles import apply_app_styles
from views.car_statistics import render_car_statistics_page
from views.estimate import render_estimator_page
from views.model_information import render_model_information_page


def main():
    st.set_page_config(
        page_title="Used Car Price Estimator for Ukraine",
        layout="wide",
    )

    apply_app_styles()
    metrics = load_metrics()
    selected_page = render_sidebar(metrics)

    render_hero(
        metrics,
        show_model_stats=selected_page in ("Model information", "Car statistics"),
    )

    if selected_page == "Estimate price":
        render_estimator_page(metrics)
    elif selected_page == "Model information":
        render_model_information_page(metrics)
    else:
        render_car_statistics_page(metrics)


if __name__ == "__main__":
    main()
