import streamlit as st

from services.data import DATA_PATH, METRICS_PATH, MODEL_PATH
from services.data import clear_cached_project_data, format_file_size, load_metrics
from services.training import run_training_with_live_logs
from ui.components import render_metric_card, render_training_log, show_model_metrics


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
        train_button_column, skip_scrape_button_column = st.columns(2)
        with train_button_column:
            train_clicked = st.button(
                "Train Model",
                type="primary",
                width="stretch",
            )
        with skip_scrape_button_column:
            train_without_scraping_clicked = st.button(
                "Train Without Scraping",
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

    if train_clicked or train_without_scraping_clicked:
        skip_scrape = train_without_scraping_clicked and not train_clicked
        spinner_message = (
            "Training from current CSV..."
            if skip_scrape
            else "Scraping fresh listings and training model..."
        )
        with st.spinner(spinner_message):
            training_exit_code = run_training_with_live_logs(
                lambda log_lines: render_training_log(
                    training_log_placeholder,
                    log_lines,
                ),
                max_listings,
                start_page,
                delay_range[0],
                delay_range[1],
                allow_partial_scrape,
                skip_scrape=skip_scrape,
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
