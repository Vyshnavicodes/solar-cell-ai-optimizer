
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import html

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="SolarCell AI | Material Intelligence",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "solar_cell_model.joblib"

# Premium palette
DARK = "#0A1714"
PANEL = "#12231E"
PANEL_LIGHT = "#182D26"
GOLD = "#E6B85C"
CREAM = "#F4F2E9"
MUTED = "#A5B7AD"
GREEN = "#8AD4A0"


# =========================================================
# CUSTOM DESIGN
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0A1714;
        --panel: #12231E;
        --panel2: #182D26;
        --gold: #E6B85C;
        --cream: #F4F2E9;
        --muted: #A5B7AD;
        --green: #8AD4A0;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 85% 0%, rgba(48, 104, 75, 0.18), transparent 35%),
            radial-gradient(ellipse at 0% 20%, rgba(230, 184, 92, 0.06), transparent 25%),
            var(--bg);
        color: var(--cream);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stAppViewContainer"] > .main {
        background: transparent;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--cream) !important;
        letter-spacing: -0.6px;
    }

    p, label, li {
        color: var(--muted);
    }

    .brandbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 5px 0 25px 0;
        border-bottom: 1px solid rgba(244,242,233,0.10);
        margin-bottom: 28px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        color: var(--cream);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 19px;
        font-weight: 700;
    }

    .brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 40px;
        width: 40px;
        border-radius: 13px;
        background: var(--gold);
        color: var(--bg);
        font-size: 23px;
    }

    .brand-sub {
        color: var(--muted);
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
        text-align: right;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(230,184,92,0.23);
        border-radius: 25px;
        padding: 42px 38px;
        background:
            radial-gradient(circle at 88% 25%, rgba(230,184,92,0.17), transparent 22%),
            radial-gradient(circle at 100% 100%, rgba(78,157,111,0.22), transparent 42%),
            linear-gradient(120deg, #142921, #0E1C17 70%);
        margin-bottom: 26px;
    }

    .eyebrow {
        display: inline-block;
        border: 1px solid rgba(230,184,92,0.35);
        color: var(--gold);
        padding: 7px 12px;
        border-radius: 100px;
        font-size: 11px;
        letter-spacing: 1.7px;
        text-transform: uppercase;
        margin-bottom: 19px;
    }

    .hero h1 {
        font-size: clamp(35px, 5vw, 57px) !important;
        line-height: 1.08 !important;
        max-width: 720px;
        margin: 0 0 18px 0 !important;
    }

    .hero h1 span {
        color: var(--gold);
    }

    .hero-copy {
        max-width: 650px;
        font-size: 16px;
        line-height: 1.8;
        color: #BDCCC3;
        margin-bottom: 25px;
    }

    .hero-foot {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }

    .hero-pill {
        background: rgba(244,242,233,0.06);
        border: 1px solid rgba(244,242,233,0.12);
        padding: 9px 12px;
        border-radius: 9px;
        color: #E2E9E2;
        font-size: 12px;
    }

    .section-kicker {
        color: var(--gold);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .section-intro {
        color: var(--muted);
        line-height: 1.7;
        margin-top: -5px;
        margin-bottom: 20px;
    }

    .feature-card {
        background: linear-gradient(145deg, var(--panel), #0F1E19);
        border: 1px solid rgba(244,242,233,0.09);
        border-radius: 17px;
        padding: 20px;
        min-height: 135px;
    }

    .feature-number {
        color: var(--gold);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 25px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .feature-title {
        color: var(--cream);
        font-weight: 700;
        margin-bottom: 7px;
    }

    .feature-desc {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
    }

    div[data-testid="stForm"] {
        background: linear-gradient(145deg, rgba(18,35,30,0.98), rgba(14,28,23,0.98));
        border: 1px solid rgba(244,242,233,0.10);
        border-radius: 19px;
        padding: 25px;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label {
        color: #DCE6DE !important;
        font-weight: 600;
        font-size: 13px;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        background: #0B1914 !important;
        color: #F4F2E9 !important;
        border: 1px solid #30473B !important;
        border-radius: 10px !important;
        min-height: 43px;
    }

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 1px var(--gold) !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #73887B !important;
    }

    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        background: var(--gold) !important;
        color: #102019 !important;
        border: 0 !important;
        border-radius: 11px !important;
        font-weight: 700 !important;
        min-height: 47px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: #F2CD7A !important;
        transform: translateY(-1px);
    }

    .result-panel {
        background:
            radial-gradient(circle at 90% 0%, rgba(230,184,92,0.14), transparent 32%),
            linear-gradient(145deg, #18382A, #10221A);
        border: 1px solid rgba(138,212,160,0.25);
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
        line-height: 1.15;
        padding: 12px 0;
        overflow-wrap: anywhere;
    }

    .result-caption {
        color: #BDCCC3;
        font-size: 13px;
        line-height: 1.7;
    }

    .note {
        border: 1px solid rgba(230,184,92,0.20);
        background: rgba(230,184,92,0.06);
        padding: 14px 16px;
        border-radius: 11px;
        color: #D7DCD2;
        font-size: 13px;
        line-height: 1.7;
    }

    .footer {
        border-top: 1px solid rgba(244,242,233,0.10);
        padding-top: 22px;
        margin-top: 40px;
        color: #82978A;
        font-size: 12px;
        line-height: 1.8;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 27px 22px;
        }
        .brand-sub {
            display: none;
        }
        div[data-testid="stForm"] {
            padding: 15px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_model(model_path, modified_time):
    # modified_time lets Streamlit reload the model if the file changes.
    return joblib.load(model_path)


def get_feature_names(loaded_model):
    """Find the original feature columns expected by the saved model."""
    if hasattr(loaded_model, "feature_names_in_"):
        return [str(x) for x in loaded_model.feature_names_in_]

    if hasattr(loaded_model, "named_steps"):
        for step in loaded_model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return [str(x) for x in step.feature_names_in_]

    return None


def get_categorical_columns(loaded_model):
    """Try to detect categorical features in a fitted sklearn pipeline."""
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

            transformer_type = type(transformer).__name__.lower()
            is_categorical = (
                "categorical" in str(name).lower()
                or "onehot" in transformer_type
                or "ordinal" in transformer_type
            )

            if is_categorical and isinstance(
                columns, (list, tuple, np.ndarray, pd.Index)
            ):
                found.update(str(c) for c in columns)

    return found


def looks_categorical(name, categorical_columns):
    if name in categorical_columns:
        return True

    keywords = (
        "material", "composition", "substrate", "architecture",
        "structure", "method", "technique", "process", "solvent",
        "electrode", "additive", "category", "type", "layer",
        "cell_name", "device"
    )
    lowered = name.lower()
    return any(word in lowered for word in keywords)


def nice_name(name):
    return str(name).replace("_", " ").replace("-", " ").strip().title()


def safe_error(error):
    return html.escape(str(error))


# =========================================================
# TOP NAVIGATION
# =========================================================
st.markdown(
    """
    <div class="brandbar">
        <div class="brand">
            <div class="brand-icon">☀</div>
            <div>
                SolarCell<span style="color:#E6B85C"> AI</span>
                <div style="font-size:10px;color:#A5B7AD;letter-spacing:2px;font-weight:500">
                    MATERIAL INTELLIGENCE
                </div>
            </div>
        </div>
        <div class="brand-sub">
            RESEARCH • MATERIALS • PERFORMANCE
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HERO SECTION
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">✦ AI-POWERED SOLAR RESEARCH</div>
        <h1>Engineer the next<br>generation of <span>solar.</span></h1>
        <div class="hero-copy">
            Explore your solar-cell design inputs with a trained
            XGBoost model. Turn material and process parameters into
            a model-based prediction to support your research.
        </div>
        <div class="hero-foot">
            <span class="hero-pill">⚡ XGBoost prediction</span>
            <span class="hero-pill">◈ Your saved Colab model</span>
            <span class="hero-pill">↗ CSV not required</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# VALUE PROPOSITION
# =========================================================
st.markdown('<div class="section-kicker">THE WORKFLOW</div>', unsafe_allow_html=True)
st.markdown("## From design parameters to insight")
st.markdown(
    '<div class="section-intro">A simple interface for exploring your trained model — '
    'without retraining it every time you open the website.</div>',
    unsafe_allow_html=True,
)

card1, card2, card3 = st.columns(3)

with card1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-number">01</div>
            <div class="feature-title">Define your design</div>
            <div class="feature-desc">
                Enter the material and process values required by your trained model.
                No hidden default values.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with card2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-number">02</div>
            <div class="feature-title">Run the AI model</div>
            <div class="feature-desc">
                The website loads your saved model from Google Colab using joblib
                and generates a prediction.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with card3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-number">03</div>
            <div class="feature-title">Review the output</div>
            <div class="feature-desc">
                Inspect the result and export the entered parameters with the
                model prediction for your records.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# LOAD SAVED MODEL
# =========================================================
st.markdown("---")
st.markdown('<div class="section-kicker">MODEL LAB</div>', unsafe_allow_html=True)
st.markdown("## Solar-cell prediction studio")
st.markdown(
    '<div class="section-intro">Enter the values used by your training dataset. '
    'Field names are taken from the saved model whenever available.</div>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Upload solar_cell_model.joblib to the same "
        "GitHub repository folder as app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as error:
    st.error("The saved model could not be loaded.")
    st.code(str(error))
    st.markdown(
        '<div class="note">Check that the model file was exported successfully '
        'from Google Colab and that the Python and package versions are compatible.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

feature_names = get_feature_names(model)

if not feature_names:
    st.error(
        "This model does not expose its original feature names. "
        "The form cannot safely guess the required columns. Please check the "
        "Google Colab training notebook and save the feature column names "
        "alongside the model."
    )
    st.stop()

categorical_columns = get_categorical_columns(model)

st.markdown(
    f"""
    <div class="note">
        <b style="color:#E6B85C">READY TO PREDICT</b><br>
        The saved model is loaded. It expects {len(feature_names)} input
        feature(s). Complete the fields below using the same units and
        category names used during training.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# =========================================================
# FORM WITH PLACEHOLDERS
# =========================================================
with st.form("solar_design_form", clear_on_submit=False):
    st.markdown("### Design parameters")
    st.caption(
        "Fields are intentionally blank. Placeholder text is an example of what "
        "to enter, not a recommended material recipe."
    )

    values = {}

    left, right = st.columns(2, gap="large")

    for index, feature in enumerate(feature_names):
        label = nice_name(feature)
        target_column = left if index % 2 == 0 else right

        with target_column:
            if looks_categorical(feature, categorical_columns):
                values[feature] = st.text_input(
                    label,
                    value="",
                    placeholder=f"Enter {label.lower()}",
                    help=(
                        "Use a category that is represented in the data "
                        "used to train the model."
                    ),
                    key=f"feature_{index}",
                )
            else:
                values[feature] = st.text_input(
                    label,
                    value="",
                    placeholder=f"Enter numeric {label.lower()}",
                    help=(
                        "Enter a number in the same units used during training. "
                        "Do not enter the unit in the box."
                    ),
                    key=f"feature_{index}",
                )

    st.write("")
    submitted = st.form_submit_button(
        "✦  Generate AI prediction",
        use_container_width=True,
        type="primary",
    )


# =========================================================
# PREDICTION RESULT
# =========================================================
if submitted:
    missing = []
    invalid = []
    cleaned = {}

    for feature in feature_names:
        raw = str(values[feature]).strip()

        if not raw:
            missing.append(nice_name(feature))
            continue

        if looks_categorical(feature, categorical_columns):
            cleaned[feature] = raw
        else:
            try:
                number = float(raw)
                if not np.isfinite(number):
                    raise ValueError
                cleaned[feature] = number
            except (ValueError, TypeError):
                invalid.append(nice_name(feature))

    if missing:
        st.error("Complete these fields before predicting: " + ", ".join(missing))

    elif invalid:
        st.error("Enter valid numeric values for: " + ", ".join(invalid))

    else:
        # Preserve the feature order expected by the saved model.
        input_df = pd.DataFrame([cleaned], columns=feature_names)

        try:
            raw_prediction = model.predict(input_df)
            prediction = float(np.asarray(raw_prediction).reshape(-1)[0])

            if not np.isfinite(prediction):
                raise ValueError("The model returned a non-finite prediction.")

            st.markdown("---")
            st.markdown('<div class="section-kicker">YOUR RESULT</div>', unsafe_allow_html=True)
            st.markdown("## Prediction generated")

            st.markdown(
                f"""
                <div class="result-panel">
                    <div class="result-label">Model-predicted output</div>
                    <div class="result-value">{prediction:,.4f}</div>
                    <div class="result-caption">
                        Generated by your saved Google Colab model.
                        This number is the model's raw output; confirm that
                        the target trained in Colab is PCE before interpreting
                        it as power conversion efficiency.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.markdown("#### Input summary")
                st.write(f"**Features supplied:** {len(feature_names)}")
                st.write("**Model:** Saved XGBoost pipeline/model")
                st.write("**Prediction status:** Complete")

            with result_col2:
                st.markdown("#### Export your result")
                output_df = input_df.copy()
                output_df["model_prediction"] = prediction

                st.download_button(
                    "↓  Download prediction CSV",
                    data=output_df.to_csv(index=False).encode("utf-8"),
                    file_name="solar_cell_prediction.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.markdown(
                """
                <div class="note">
                    <b style="color:#E6B85C">SCIENTIFIC NOTE</b><br>
                    This prediction is an estimate learned from historical training
                    data. It is not a guarantee of experimental performance and
                    does not independently prove that a material combination is
                    optimal. Validate promising candidates experimentally.
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as error:
            st.error(
                "Prediction failed. One or more values may not match the "
                "format or categories expected by the trained model."
            )
            st.code(str(error))
            st.caption(
                "If the error mentions unknown categories, check the exact "
                "category labels in the original training dataset."
            )


# =========================================================
# OPTIONAL DATA EXPLORER
# =========================================================
st.markdown("---")
st.markdown('<div class="section-kicker">OPTIONAL</div>', unsafe_allow_html=True)
st.markdown("## Explore your dataset")
st.markdown(
    '<div class="section-intro">Your CSV is only needed for this optional '
    'preview — not for model predictions.</div>',
    unsafe_allow_html=True,
)

uploaded_csv = st.file_uploader(
    "Upload a historical solar-cell CSV",
    type=["csv"],
    help="Optional: upload your dataset to preview its first rows.",
)

if uploaded_csv is not None:
    try:
        data = pd.read_csv(uploaded_csv)

        metric1, metric2 = st.columns(2)
        metric1.metric("Records", f"{len(data):,}")
        metric2.metric("Columns", f"{len(data.columns):,}")

        st.dataframe(data.head(10), use_container_width=True)

    except Exception as error:
        st.error("Could not read this CSV file.")
        st.code(str(error))


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:16px;
                    color:#F4F2E9;font-weight:700;margin-bottom:7px">
            ☀ SolarCell <span style="color:#E6B85C">AI</span>
        </div>
        AI-based solar-cell material exploration.<br>
        Built with Streamlit, Python and a pretrained model exported using joblib.<br><br>
        <span style="color:#647B6D">
            Research support only • Predictions depend on the training data and
            model limitations • Experimental validation is essential
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
