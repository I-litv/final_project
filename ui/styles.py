import streamlit as st


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

            [data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid var(--line);
                box-shadow: 10px 0 30px rgba(18, 24, 38, 0.04);
            }

            [data-testid="stSidebar"] > div:first-child {
                padding: 1.35rem 1rem 1.5rem;
            }

            .sidebar-brand {
                border-bottom: 1px solid var(--line);
                margin-bottom: 18px;
                padding: 4px 2px 18px;
            }

            .sidebar-brand-row {
                align-items: center;
                display: flex;
                gap: 10px;
            }

            .sidebar-brand-mark {
                align-items: center;
                background: #0f766e;
                border-radius: 10px;
                color: #ffffff;
                display: flex;
                font-size: 0.85rem;
                font-weight: 800;
                height: 36px;
                justify-content: center;
                width: 36px;
            }

            .sidebar-brand-text strong {
                color: var(--ink);
                display: block;
                font-size: 1rem;
                line-height: 1.15;
            }

            .sidebar-brand-text span {
                color: var(--muted);
                display: block;
                font-size: 0.78rem;
                font-weight: 650;
                margin-top: 3px;
            }

            [data-testid="stSidebar"] [data-testid="stRadio"] > label {
                color: var(--muted);
                font-size: 0.72rem;
                font-weight: 780;
                letter-spacing: 0.08em;
                margin: 0 0 8px;
                text-transform: uppercase;
            }

            [data-testid="stSidebar"] [role="radiogroup"] {
                gap: 7px;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label {
                align-items: center;
                background: transparent;
                border: 1px solid transparent;
                border-radius: 10px;
                color: #344054;
                cursor: pointer;
                display: flex;
                min-height: 44px;
                padding: 10px 12px;
                transition: background 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
                display: none;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label > div:last-child {
                width: 100%;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label:hover {
                background: #f6f8fb;
                border-color: #dbe4ef;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
                background: #eaf1ff;
                border-color: rgba(37, 99, 235, 0.25);
                box-shadow: inset 3px 0 0 #2563eb;
                color: #174ea6;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
                color: #174ea6;
                font-weight: 780;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label p {
                color: inherit;
                font-size: 0.94rem;
                font-weight: 680;
                line-height: 1.2;
            }

            .sidebar-status {
                background: #f7fafc;
                border: 1px solid var(--line);
                border-radius: 12px;
                margin-top: 20px;
                padding: 13px 14px;
            }

            .sidebar-status span {
                color: var(--muted);
                display: block;
                font-size: 0.72rem;
                font-weight: 760;
                text-transform: uppercase;
            }

            .sidebar-status strong {
                color: var(--ink);
                display: block;
                font-size: 0.94rem;
                line-height: 1.35;
                margin-top: 4px;
                overflow-wrap: anywhere;
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
                background: #f7f6ef;
                border: 1px solid #c8d1dc;
                border-radius: 14px;
                height: 420px;
                overflow-y: auto;
                padding: 16px;
                box-shadow:
                    inset 4px 0 0 #4f7cac,
                    0 12px 30px rgba(18, 24, 38, 0.07);
                scrollbar-color: #9fb1c7 #ece9df;
            }

            .training-log pre {
                color: #263238;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
                font-size: 0.84rem;
                font-weight: 560;
                line-height: 1.56;
                margin: 0;
                white-space: pre-wrap;
                overflow-wrap: anywhere;
                text-shadow: 0 1px 0 rgba(255, 255, 255, 0.55);
            }

            .training-log::-webkit-scrollbar {
                width: 10px;
            }

            .training-log::-webkit-scrollbar-track {
                background: #ece9df;
                border-radius: 999px;
            }

            .training-log::-webkit-scrollbar-thumb {
                background: #9fb1c7;
                border: 2px solid #ece9df;
                border-radius: 999px;
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
