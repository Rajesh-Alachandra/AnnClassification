import pickle

import pandas as pd
import streamlit as st
import tensorflow as tf


st.set_page_config(
    page_title="Will a Customer Leave?",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        color-scheme: light;
        --ink: #17233c;
        --muted: #74809a;
        --blue: #4169e1;
        --mint: #25b89a;
        --line: #e8edf5;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 86% 4%, rgba(65, 105, 225, .10), transparent 28%),
            linear-gradient(180deg, #f7f9fd 0%, #f2f5fb 100%);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; color: var(--ink); }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.1rem 2.3rem;
        border: 1px solid rgba(255,255,255,.55);
        border-radius: 24px;
        background: linear-gradient(115deg, #18294c 0%, #294b91 62%, #4169e1 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(34, 57, 108, .17);
        margin-bottom: 1.5rem;
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -62px;
        top: -115px;
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 50%;
        box-shadow: 0 0 0 30px rgba(255,255,255,.035), 0 0 0 62px rgba(255,255,255,.025);
    }
    .hero-kicker {
        color: #b9ccff;
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .45rem;
    }
    .hero-title {
        color: white;
        font: 800 clamp(1.7rem, 3.5vw, 2.5rem)/1.15 'Manrope', sans-serif;
        letter-spacing: -.04em;
        margin: 0;
    }
    .hero-copy {
        max-width: 620px;
        color: #d7e1fb;
        font-size: 1rem;
        line-height: 1.6;
        margin: .65rem 0 0;
    }
    .section-label {
        color: var(--blue);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin: 0 0 .25rem;
    }
    .section-title {
        color: var(--ink);
        font: 700 1.3rem 'Manrope', sans-serif;
        margin: 0 0 .2rem;
    }
    .section-copy { color: var(--muted); font-size: .91rem; margin-bottom: 1rem; }
    .result-card {
        padding: 1.4rem 1.5rem;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: rgba(255,255,255,.9);
        box-shadow: 0 12px 30px rgba(27, 45, 80, .06);
        margin-bottom: 1rem;
    }
    .result-label {
        color: var(--muted);
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
    }
    .result-value {
        color: var(--ink);
        font: 800 2.8rem/1.2 'Manrope', sans-serif;
        letter-spacing: -.05em;
        margin: .25rem 0 .55rem;
    }
    .status {
        padding: .8rem 1rem;
        border-radius: 12px;
        font-size: .95rem;
        font-weight: 700;
        margin: .9rem 0;
    }
    .status-high { color: #a63c43; background: #fff0ef; border: 1px solid #ffdcda; }
    .status-low { color: #16836f; background: #eaf9f5; border: 1px solid #d3f0e8; }
    .note {
        color: var(--muted);
        font-size: .82rem;
        line-height: 1.55;
        padding: .8rem 0 0;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 18px;
        background: rgba(255,255,255,.86);
        box-shadow: 0 12px 30px rgba(27, 45, 80, .045);
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: var(--blue); }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label {
        color: #40506e;
        font-weight: 600;
    }
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        color: var(--ink) !important;
        background-color: #fff !important;
        border-color: #dbe3f0 !important;
    }
    div[data-testid="stNumberInput"] button {
        color: #40506e !important;
        background-color: #f7f9fd !important;
        border-color: #dbe3f0 !important;
    }
    div[data-testid="stSlider"] [role="slider"] {
        background-color: var(--blue) !important;
        border-color: var(--blue) !important;
    }
    div[data-testid="stProgress"] > div > div > div { background: var(--mint); }
    @media (max-width: 700px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero { padding: 1.5rem; border-radius: 18px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("model.h5")
    with open("label_encoder_gender.pkl", "rb") as file:
        gender_encoder = pickle.load(file)
    with open("onehot_encoder_geo.pkl", "rb") as file:
        geography_encoder = pickle.load(file)
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    return model, gender_encoder, geography_encoder, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Bank customer check</div>
        <h1 class="hero-title">Could this customer leave the bank?</h1>
        <p class="hero-copy">
            Enter a customer's details to see how likely they may be to leave.
            Change the details to see how the estimate changes.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

input_column, result_column = st.columns([1.2, 0.8], gap="large")

with input_column:
    with st.container(border=True):
        st.markdown(
            '<p class="section-label">About the customer</p>'
            '<p class="section-title">Customer details</p>'
            '<p class="section-copy">Fill in the details below to get an estimate.</p>',
            unsafe_allow_html=True,
        )

        geography_column, gender_column = st.columns(2)
        with geography_column:
            geography = st.selectbox("Country", onehot_encoder_geo.categories_[0])
        with gender_column:
            gender = st.selectbox("Gender", label_encoder_gender.classes_)

        age_column, credit_column, tenure_column = st.columns(3)
        with age_column:
            age = st.slider("Age", 18, 92, 40)
        with credit_column:
            credit_score = st.number_input(
                "Credit score",
                min_value=0,
                max_value=1000,
                value=650,
                step=1,
                help="A number that shows how well someone has managed borrowing and repayments.",
            )
        with tenure_column:
            tenure = st.slider("Years with this bank", 0, 10, 5)

        balance_column, salary_column = st.columns(2)
        with balance_column:
            balance = st.number_input(
                "Money in the account",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
                format="%.2f",
            )
        with salary_column:
            estimated_salary = st.number_input(
                "Estimated yearly salary",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
                format="%.2f",
            )

        products_column, card_column, active_column = st.columns(3)
        with products_column:
            num_of_products = st.selectbox("Number of bank products", [1, 2, 3, 4], index=1)
        with card_column:
            has_cr_card = st.selectbox("Has credit card", [1, 0], format_func=lambda value: "Yes" if value else "No")
        with active_column:
            is_active_member = st.selectbox(
                "Active bank customer",
                [1, 0],
                format_func=lambda value: "Yes" if value else "No",
            )

with result_column:
    gender_encoded = label_encoder_gender.transform([gender])[0]
    input_data = pd.DataFrame(
        {
            "CreditScore": [credit_score],
            "Gender": [gender_encoded],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
        }
    )

    geography_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geography_columns = onehot_encoder_geo.get_feature_names_out(["Geography"])
    geography_data = pd.DataFrame(geography_encoded, columns=geography_columns)
    input_data = pd.concat([input_data.reset_index(drop=True), geography_data], axis=1)
    input_data_scaled = scaler.transform(input_data)
    churn_probability = float(model.predict(input_data_scaled, verbose=0)[0][0])
    churn_percent = churn_probability * 100

    st.markdown(
        '<p class="section-label">Result</p>'
        '<p class="section-title">Chance the customer may leave</p>'
        '<p class="section-copy">This estimate changes when you update the details.</p>',
        unsafe_allow_html=True,
    )

    if churn_probability > 0.5:
        status_text = "Higher chance of leaving"
        status_class = "status-high"
        detail_text = "The estimate is more than 50%."
    else:
        status_text = "Lower chance of leaving"
        status_class = "status-low"
        detail_text = "The estimate is 50% or less."

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Estimated chance</div>
            <div class="result-value">{churn_percent:.1f}<span style="font-size:1.35rem;color:#74809a">%</span></div>
            <div class="status {status_class}">{status_text}</div>
            <div class="note">{detail_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(churn_probability, text=f"Chance of leaving · {churn_percent:.1f}%")

    st.markdown(
        """
        <div class="note">
            <strong>Please note:</strong> This is only an estimate. It cannot say for
            certain what a customer will do.
        </div>
        """,
        unsafe_allow_html=True,
    )
