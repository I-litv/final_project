from html import escape

import pandas as pd
import streamlit as st

from services.formatting import format_usd


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


def display_statistics_table(dataframe):
    st.dataframe(
        dataframe,
        hide_index=True,
        width="stretch",
    )


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
