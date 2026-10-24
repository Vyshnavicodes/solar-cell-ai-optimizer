
import html
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="SolarCell AI | Material Intelligence",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "solar_cell_model.joblib"
CSV_PATH = BASE_DIR / "Solar Cell data set (1).csv"

BG = "#0A1714"
PANEL = "#12231E"
GOLD = "#E6B85C"
CREAM = "#F4F2E9"
MUTED = "#A5B7AD"


# =========================================================
# 2. PREMIUM WEBSITE STYLING
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0A1714;
        --panel: #12231E;
        --gold: #E6B85C;
        --cream: #F4F2E9;
        --muted: #A5B7AD;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 85% 0%, rgba(48,104,75,.18), transparent 35%),
            radial-gradient(ellipse at 0% 20%, rgba(230,184,92,.06), transparent 25%),
            var(--bg);
        color: var(--cream);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--cream) !important;
        letter-spacing: -.5px;
    }

    p, label {
        color: var(--muted);
    }

    .brandbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 5px 0 24px;
        border-bottom: 1px solid rgba(244,242,233,.10);
        margin-bottom: 28px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        color: var(--cream);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 20px;
        font-weight: 700;
    }

    .brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        border-radius: 14px;
        background: var(--gold);
        color: var(--bg);
        font-size: 25px;
    }

    .brand-sub {
        color: var(--muted);
        font-size: 10px;
        letter-spacing: 2px;
        text-align: right;
    }

    .hero {
        border: 1px solid rgba(230,184,92,.25);
        border-radius: 25px;
        padding: 42px 36px;
        background:
            radial-gradient(circle at 90% 15%, rgba(230,184,92,.17), transparent 25%),
            radial-gradient(circle at 100% 100%, rgba(78,157,111,.22), transparent 42%),
            linear-gradient(120deg, #142921, #0E1C17 75%);
        margin-bottom: 28px;
    }

    .eyebrow {
        display: inline-block;
        border: 1px solid rgba(230,184,92,.35);
        color: var(--gold);
        padding: 8px 12px;
        border-radius: 100px;
        font-size: 11px;
        letter-spacing: 1.5px;
        margin-bottom: 20px;
    }

    .hero h1 {
        font-size: clamp(36px, 5vw, 58px) !important;
        line-height: 1.08 !important;
        margin: 0 0 18px 0 !important;
    }

    .hero h1 span {
        color: var(--gold);
    }

    .hero-copy {
        max-width: 660px;
        font-size: 16px;
        line-height: 1.8;
        color: #BDCCC3;
        margin-bottom: 25px;
    }

    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }

    .pill {
        background: rgba(244,242,233,.06);
        border: 1px solid rgba(244,242,233,.12);
        padding: 9px 12px;
        border-radius: 9px;
        color: #E2E9E2;
        font-size: 12px;
    }

    .kicker {
        color: var(--gold);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .intro {
        color: var(--muted);
        line-height: 1.8;
        margin-bottom: 20px;
    }

    .feature-card {
        background: linear-gradient(145deg, #12231E, #0F1E19);
        border: 1px solid rgba(244,242,233,.09);
        border-radius: 17px;
        padding: 21px;
        min-height: 150px;
    }

    .feature-number {
        color: var(--gold);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .feature-title {
        color: var(--cream);
        font-weight: 700;
        margin-bottom: 8px;
    }

    .feature-desc {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.7;
    }

    div[data-testid="stForm"] {
        background: linear-gradient(145deg, #12231E, #0E1C17);
        border: 1px solid rgba(244,242,233,.10);
        border-radius: 19px;
        padding: 24px;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        color: #DCE6DE !important;
        font-weight: 600;
        font-size: 13px;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #0B1914 !important;
        color: #F4F2E9 !important;
        border-radius: 10px !important;
        border-color: #30473B !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #73887B !important;
    }

    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        background: var(--gold) !important;
        color: #102019 !important;
        border: 0 !important;
        border-radius: 11px !important;
        font-weight: 700 !important;
        min-height: 48px;
    }

    div[data-testid="stFormSubmitButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: #F2CD7A !important;
    }

    .result-panel {
        background:
            radial-gradient(circle at 90% 0%, rgba(230,184,92,.14), transparent 32%),
            linear-gradient(145deg, #18382A, #10221A);
        border: 1px solid rgba(138,212,160,.25);
        border-radius: 21px;
        padding: 28px;
        margin: 20px 0;
    }

    .result-label {
        color: #BDCCC3;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .result-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(38px, 6vw, 60px);
        font-weight: 700;
        color: var(--gold);
        line-height: 1.2;
        padding: 12px 0;
        overflow-wrap: anywhere;
    }

    .result-caption {
        color: #BDCCC3;
        font-size: 13px;
        line-height: 1.8;
    }

    .note {
        border: 1px solid rgba(230,184,92,.20);
        background: rgba(230,184,92,.06);
        padding: 15px;
        border-radius: 11px;
        color: #D7DCD2;
        font-size: 13px;
        line-height: 1.8;
    }

    .footer {
        border-top: 1px solid rgba(244,242,233,.10);
        padding-top: 23px;
        margin-top: 40px;
        color: #82978A;
        font-size: 12px;
        line-height: 1.8;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 700px) {
        .hero { padding: 27px 21px; }
        .brand-sub { display: none; }
        div[data-testid="stForm"] { padding: 15px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. LOAD SAVED MODEL AND OPTIONAL REFERENCE DATA
# =========================================================
@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


@st.cache_data
def load_dataset(path, modified_time):
    return pd.read_csv(path)


def get_feature_names(loaded_model):
    if hasattr(loaded_model, "feature_names_in_"):
        return [str(name) for name in loaded_model.feature_names_in_]

    if hasattr(loaded_model, "named_steps"):
        for step in loaded_model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return [str(name) for name in step.feature_names_in_]

    return None


def get_categorical_columns(loaded_model):
    found = set()

    if not hasattr(loaded_model, "named_steps"):
        return found

    for step in loaded_model.named_steps.values():
        transformers = getattr(step, "transformers_", None)

        if transformers is None:
            continue

        for name, transformer, columns in transformers:
            if name == "remainder" or transformer == "drop":
                continue

            transformer_name = type(transformer).__name__.lower()
            is_categorical = (
                "categorical" in str(name).lower()
                or "onehot" in transformer_name
                or "ordinal" in transformer_name
            )

            if is_categorical and isinstance(
                columns, (list, tuple, np.ndarray, pd.Index)
            ):
                found.update(str(col) for col in columns)

    return found


def pretty_name(name):
    return str(name).replace("_", " ").replace("-", " ").strip().title()


def likely_categorical(name):
    keywords = (
        "material", "composition", "substrate", "architecture",
        "structure", "method", "technique", "process", "solvent",
        "electrode", "additive", "category", "type", "layer",
        "device", "cell_name"
    )
    name = name.lower()
    return any(word in name for word in keywords)


# =========================================================
# 4. HEADER AND HERO
# =========================================================
st.markdown(
    """
    <div class="brandbar">
        <div class="brand">
            <div class="brand-icon">☀</div>
            <div>
                SolarCell <span style="color:#E6B85C">AI</span>
                <div style="font-size:10px;color:#A5B7AD;
                            letter-spacing:2px;font-weight:500">
                    MATERIAL INTELLIGENCE
                </div>
            </div>
        </div>
        <div class="brand-sub">RESEARCH • MATERIALS • PERFORMANCE</div>
    </div>

    <div class="hero">
        <div class="eyebrow">✦ AI-POWERED SOLAR RESEARCH</div>
        <h1>Engineer the next<br>generation of <span>solar.</span></h1>
        <div class="hero-copy">
            Explore solar-cell design parameters with your trained XGBoost
            model. Turn material and process inputs into a model-based
            efficiency estimate.
        </div>
        <div class="hero-pills">
            <span class="pill">⚡ XGBoost prediction</span>
            <span class="pill">◈ Saved Google Colab model</span>
            <span class="pill">↗ No CSV needed to predict</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="kicker">THE WORKFLOW</div>', unsafe_allow_html=True)
st.markdown("## From design parameters to insight")
st.markdown(
    '<div class="intro">Choose from known training-data categories, enter your '
    'design values, and estimate the model output.</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-number">01</div>
            <div class="feature-title">Choose your inputs</div>
            <div class="feature-desc">Select available material categories and enter
            numeric properties without random defaults.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-number">02</div>
            <div class="feature-title">Run the saved model</div>
            <div class="feature-desc">The website loads your exported model using
            joblib instead of training a new model.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-number">03</div>
            <div class="feature-title">Review efficiency</div>
            <div class="feature-desc">See the model's prediction and download
            your inputs and result.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 5. LOAD MODEL
# =========================================================
st.markdown("---")
st.markdown('<div class="kicker">MODEL LAB</div>', unsafe_allow_html=True)
st.markdown("## Solar-cell efficiency studio")
st.markdown(
    '<div class="intro">Fields are generated from the saved model’s expected '
    'input columns. Dataset-based dropdowns appear when the CSV is available.</div>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        "Model not found. Add solar_cell_model.joblib to the same GitHub "
        "repository folder as app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as error:
    st.error("Could not load the saved model.")
    st.code(str(error))
    st.info(
        "Check that the Python and package versions are compatible with those "
        "used when the model was saved in Google Colab."
    )
    st.stop()

feature_names = get_feature_names(model)

if not feature_names:
    st.error(
        "The saved model does not expose its original feature names. "
        "For a safe input form, the original training feature columns are needed."
    )
    st.stop()

categorical_columns = get_categorical_columns(model)

# The CSV is used only for form options and numeric ranges.
reference_data = None
if CSV_PATH.exists():
    try:
        reference_data = load_dataset(str(CSV_PATH), CSV_PATH.stat().st_mtime)
    except Exception:
        reference_data = None

if reference_data is not None:
    st.markdown(
        f"""
        <div class="note">
            <b style="color:#E6B85C">MODEL READY</b><br>
            Saved model loaded successfully. Dropdown choices and observed
            numeric ranges are based on the available CSV.
            Required inputs: {len(feature_names)}.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="note">
            <b style="color:#E6B85C">MODEL READY</b><br>
            Saved model loaded. The CSV is not available, so numeric fields
            and any categorical fields that cannot be inferred will need
            manual input.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 6. BUILD DROPDOWN OPTIONS FROM TRAINING DATA
# =========================================================
feature_options = {}
numeric_ranges = {}

if reference_data is not None:
    for feature in feature_names:
        if feature not in reference_data.columns:
            continue

        series = reference_data[feature].dropna()

        if series.empty:
            continue

        is_text = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )

        if is_text or feature in categorical_columns:
            options = sorted(
                series.astype(str).unique().tolist(),
                key=str.casefold,
            )

            if options:
                feature_options[feature] = options
                categorical_columns.add(feature)

        elif pd.api.types.is_numeric_dtype(series):
            numeric_ranges[feature] = (
                float(series.min()),
                float(series.max()),
            )


# =========================================================
# 7. INPUT FORM
# =========================================================
st.write("")
with st.form("solar_cell_form", clear_on_submit=False):
    st.markdown("### Your solar-cell design")
    st.caption(
        "Choose a dropdown option or enter a number. Blank fields are not "
        "pre-filled with arbitrary values. Use the same units as your dataset."
    )

    values = {}
    left, right = st.columns(2, gap="large")

    for index, feature in enumerate(feature_names):
        label = pretty_name(feature)
        column = left if index % 2 == 0 else right

        with column:
            if feature in feature_options:
                values[feature] = st.selectbox(
                    label,
                    options=feature_options[feature],
                    index=None,
                    placeholder=f"Choose {label.lower()}...",
                    help="Options are taken from the reference CSV.",
                    key=f"field_{index}",
                )

            elif feature in numeric_ranges:
                minimum, maximum = numeric_ranges[feature]

                # Streamlit number_input requires a usable interval.
                values[feature] = st.number_input(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=None,
                    placeholder=f"{minimum:.4g} to {maximum:.4g}",
                    help=(
                        f"Observed dataset range: {minimum:.4g}–{maximum:.4g}. "
                        "Enter a value in the original training units."
                    ),
                    key=f"field_{index}",
                )

            elif feature in categorical_columns or likely_categorical(feature):
                values[feature] = st.text_input(
                    label,
                    placeholder=f"Enter {label.lower()}",
                    help="Enter a category that the model can recognize.",
                    key=f"field_{index}",
                )

            else:
                values[feature] = st.number_input(
                    label,
                    value=None,
                    placeholder="Enter numeric value",
                    help="Use the same units as the training dataset.",
                    key=f"field_{index}",
                )

    st.markdown(
        '<p style="font-size:12px;color:#A5B7AD">'
        'Note: observed ranges describe the historical dataset; they are not '
        'guaranteed physical limits or optimal settings.</p>',
        unsafe_allow_html=True,
    )

    submitted = st.form_submit_button(
        "☀  Estimate Cell Efficiency",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# 8. PREDICT
# =========================================================
if submitted:
    missing = []
    invalid = []
    cleaned = {}

    for feature in feature_names:
        raw = values[feature]

        if raw is None or (
            isinstance(raw, str) and not raw.strip()
        ):
            missing.append(pretty_name(feature))
            continue

        if feature in feature_options or (
            feature in categorical_columns and
            not pd.api.types.is_numeric_dtype(
                reference_data[feature]
            ) if reference_data is not None and feature in reference_data.columns
            else feature in categorical_columns or likely_categorical(feature)
        ):
            cleaned[feature] = str(raw).strip()
        else:
            try:
                number = float(raw)
                if not np.isfinite(number):
                    raise ValueError
                cleaned[feature] = number
            except (ValueError, TypeError):
                invalid.append(pretty_name(feature))

    if missing:
        st.error("Please complete: " + ", ".join(missing))

    elif invalid:
        st.error("Please enter valid numbers for: " + ", ".join(invalid))

    else:
        input_df = pd.DataFrame([cleaned], columns=feature_names)

        try:
            prediction_array = model.predict(input_df)
            prediction = float(np.asarray(prediction_array).reshape(-1)[0])

            if not np.isfinite(prediction):
                raise ValueError("The model returned an invalid number.")

            # Confirm the target and units from your Colab notebook.
            # Set this to True ONLY if your target was PCE in percentage units.
            TARGET_IS_PCE_PERCENT = False

            if TARGET_IS_PCE_PERCENT:
                displayed_prediction = f"{prediction:.2f}%"
                result_label = "Predicted power conversion efficiency"
                result_note = (
                    "Interpreted as PCE in percent because the training target "
                    "was confirmed to use percentage units."
                )
            else:
                displayed_prediction = f"{prediction:.4f}"
                result_label = "Model prediction"
                result_note = (
                    "This is the model's raw output. Confirm the training target "
                    "and its units before interpreting it as efficiency."
                )

            st.markdown("---")
            st.markdown('<div class="kicker">YOUR RESULT</div>', unsafe_allow_html=True)
            st.markdown("## Efficiency estimate")

            st.markdown(
                f"""
                <div class="result-panel">
                    <div class="result-label">{html.escape(result_label)}</div>
                    <div class="result-value">{html.escape(displayed_prediction)}</div>
                    <div class="result-caption">
                        {html.escape(result_note)}<br>
                        Prediction generated by your saved Google Colab model.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### Input summary")
            st.dataframe(input_df, use_container_width=True)

            output_df = input_df.copy()
            output_df["model_prediction"] = prediction

            st.download_button(
                "↓  Download prediction",
                data=output_df.to_csv(index=False).encode("utf-8"),
                file_name="solar_cell_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown(
                """
                <div class="note">
                    <b style="color:#E6B85C">SCIENTIFIC NOTE</b><br>
                    Predictions are estimates based on patterns in the training
                    data. They are not guarantees of real experimental
                    performance and do not prove that a recipe is optimal.
                    Validate promising designs experimentally.
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as error:
            st.error(
                "Prediction failed. Check that the selected categories and "
                "numeric values match the format expected by the saved model."
            )
            st.code(str(error))
            st.caption(
                "If you see an unknown-category error, the selected value may "
                "not have been represented during training."
            )


# =========================================================
# 9. OPTIONAL CSV PREVIEW
# =========================================================
st.markdown("---")
st.markdown('<div class="kicker">OPTIONAL DATA EXPLORER</div>', unsafe_allow_html=True)
st.markdown("## Explore historical records")
st.markdown(
    '<div class="intro">Predictions do not require a CSV upload. '
    'Use this section only if you want to inspect another dataset.</div>',
    unsafe_allow_html=True,
)

uploaded_csv = st.file_uploader(
    "Upload a CSV to preview",
    type=["csv"],
    help="Optional. This upload does not retrain or replace the saved model.",
)

if uploaded_csv is not None:
    try:
        preview_data = pd.read_csv(uploaded_csv)
        m1, m2 = st.columns(2)
        m1.metric("Records", f"{len(preview_data):,}")
        m2.metric("Columns", len(preview_data.columns))
        st.dataframe(preview_data.head(10), use_container_width=True)
    except Exception as error:
        st.error("Unable to read this CSV.")
        st.code(str(error))


# =========================================================
# 10. FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:17px;
                    color:#F4F2E9;font-weight:700;margin-bottom:8px">
            ☀ SolarCell <span style="color:#E6B85C">AI</span>
        </div>
        AI-powered solar-cell material exploration.<br>
        Built with Streamlit, Python, XGBoost and a pretrained joblib model.<br><br>
        Predictions are research estimates. Experimental validation is essential.
    </div>
    """,
    unsafe_allow_html=True,
)
