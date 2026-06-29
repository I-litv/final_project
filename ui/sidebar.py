import streamlit as st

from services.data import DATA_PATH, count_csv_rows, format_file_size
from services.data import get_data_file_signature


def render_sidebar(metrics):
    csv_row_count = count_csv_rows(get_data_file_signature())
    if csv_row_count:
        dataset_status = f"{csv_row_count:,} CSV rows"
    elif metrics and metrics.get("cars_on_sale_count") is not None:
        dataset_status = f"{metrics.get('cars_on_sale_count'):,} trained rows"
    else:
        dataset_status = format_file_size(DATA_PATH)

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-row">
                    <div class="sidebar-brand-mark">UC</div>
                    <div class="sidebar-brand-text">
                        <strong>Used Car Market</strong>
                        <span>Final project</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_page = st.radio(
            "Navigation",
            ("Estimate price", "Car statistics", "Model information"),
        )
        st.markdown(
            f"""
            <div class="sidebar-status">
                <span>Dataset</span>
                <strong>{dataset_status}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_page
