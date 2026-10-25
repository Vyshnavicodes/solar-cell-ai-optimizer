```python
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SolarIQ | AI Cell Optimizer",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "solar_cell_model.joblib")
CSV_PATH = os.path.join(BASE_DIR, "Solar Cell data set (1).csv")

TARGET = "JV_default_PCE"

# =========================================================
# PREMIUM RESEARCH-STYLE DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background: #091713;
    color: #F3F4EC;
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #F3F4EC;
}

.hero {
    padding: 35px;
    border: 1px solid #29483B;
    border-radius: 24px;
    background:
        radial-gradient(circle at 85% 15%, #294B35 0%, transparent 40%),
        linear-gradient(135deg, #10251C, #0B1914);
    margin-bottom: 25px;
}

.eyebrow {
    color: #E6BD65;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(32px, 5vw, 54px);
    line-height: 1.1;
    font-weight: 700;
    margin: 15px 0;
}

.hero-copy {
    color: #B5C7BC;
    font-size: 16px;
    max-width: 750px;
    line-height: 1.8;
}

.panel {
    background: #10231B;
    border: 1px solid #29483B;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 15px;
}

.result-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 52px;
    font-weight: 700;
    color: #E6BD65;
}

.muted {
    color: #A9BDB0;
    font-size: 13px;
}

div[data-testid="stMetric"] {
    background: #10231B;
    border: 1px solid #29483B;
    padding: 18px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #B5C7BC;
}

div[data-testid="stMetricValue"] {
    color: #E6BD65;
}

.stButton > button[kind="primary"],
.stFormSubmitButton > button {
    background: #E6BD65 !important;
    color: #102017 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 46px;
    width: 100%;
}

.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button:hover {
    background: #F2D58E !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #12271E;
    border-color: #355746;
    border-radius: 8px;
}

section[data-testid="stSidebar"] {
    background: #0D2018;
    border-right: 1px solid #29483B;
}

hr {
    border-color: #29483B;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD SAVED MODEL AND REFERENCE DATA
# =========================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "solar_cell_model.joblib was not found. "
            "Upload it to the same GitHub folder as app.py."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        return None
    return pd.read_csv(CSV_PATH, low_memory=False)


try:
    model = load_model()
except Exception as exc:
    st.error("Unable to load the saved model.")
    st.code(str(exc))
    st.info(
        "Check that solar_cell_model.joblib is in the repository "
        "and that requirements.txt contains compatible package versions."
    )
    st.stop()

try:
    data = load_data()
except Exception as exc:
    st.warning(f"Dataset could not be loaded: {exc}")
    data = None


if data is not None and TARGET not in data.columns:
    st.warning(
        f"The CSV does not contain the expected target column '{TARGET}'. "
        "Check the actual target column name."
    )
    data = None


# =========================================================
# DETERMINE MODEL INPUT FEATURES
# =========================================================

model_features = getattr(model, "feature_names_in_", None)

if model_features is not None:
    FEATURES = list(model_features)
elif data is not None:
    FEATURES = [c for c in data.columns if c != TARGET]
else:
    st.error(
        "Cannot determine model input features. "
        "Keep the training CSV alongside the saved model."
    )
    st.stop()

if TARGET in FEATURES:
    st.error(
        "The model feature list includes the target PCE column. "
        "Check your training notebook to prevent target leakage."
    )
    st.stop()

if data is not None:
    missing_columns = [c for c in FEATURES if c not in data.columns]
    if missing_columns:
        st.error(
            "These model input columns are missing from the CSV: "
            + ", ".join(missing_columns)
        )
        st.stop()


# =========================================================
# FEATURE TYPE DETECTION
# =========================================================

def is_numeric_feature(column):
    if data is not None and column in data.columns:
        return pd.api.types.is_numeric_dtype(data[column])

    return False


def get_options(column):
    if data is None or column not in data.columns:
        return []

    series = data[column].dropna().astype(str).str.strip()
    series = series[series != ""]
    counts = series.value_counts()

    # Keep dropdowns manageable while preserving common choices.
    return counts.index[:150].tolist()


def make_reference_input():
    """Create a reasonable starting point from observed dataset values."""
    values = {}

    for column in FEATURES:
        if data is None or column not in data.columns:
            values[column] = None
            continue

        series = data[column]

        if is_numeric_feature(column):
            clean = pd.to_numeric(series, errors="coerce").dropna()
            values[column] = (
                float(clean.median()) if not clean.empty else None
            )
        else:
            clean = series.dropna().astype(str).str.strip()
            clean = clean[clean != ""]
            values[column] = (
                clean.mode().iloc[0] if not clean.empty else None
            )

    return values


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">AI MATERIALS RESEARCH · XGBOOST</div>
    <div class="hero-title">
        Discover better<br>solar cell performance.
    </div>
    <div class="hero-copy">
        Predict power conversion efficiency from material and
        fabrication parameters. Explore existing configurations
        and identify promising candidates for further research.
    </div>
</div>
""", unsafe_allow_html=True)

if data is not None:
    a, b, c = st.columns(3)
    a.metric("Dataset records", f"{len(data):,}")
    b.metric("Input features", len(FEATURES))
    c.metric("Target", "PCE (%)")
else:
    st.info(
        "The saved model is loaded, but the CSV is unavailable. "
        "Prediction may still work if all model features are known."
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## ☀️ SolarIQ")
    st.caption("AI-powered solar materials research")
    st.divider()

    st.markdown("### Project")
    st.write("Algorithm: XGBoost")
    st.write("Task: Regression")
    st.write("Output: Power Conversion Efficiency")
    st.divider()

    st.markdown("### How it works")
    st.write("1. Enter cell parameters.")
    st.write("2. Run the saved model.")
    st.write("3. Compare predicted performance.")
    st.write("4. Explore promising candidates.")

    st.caption("Predictions are research estimates, not experimental guarantees.")


# =========================================================
# MAIN TABS
# =========================================================

tab_predict, tab_optimize, tab_compare, tab_about = st.tabs([
    "🔬 PCE Prediction",
    "🚀 Optimization",
    "📊 Comparative Analysis",
    "ℹ️ About the Model",
])


# =========================================================
# TAB 1: PREDICTION FORM
# =========================================================

with tab_predict:
    st.subheader("Solar cell configuration")
    st.write(
        "Complete the model's input features below. "
        "Dropdown choices are taken from your dataset where available."
    )

    if data is None:
        st.error(
            "The CSV is needed to build the input form. "
            "Upload Solar Cell data set (1).csv beside app.py."
        )
    else:
        with st.form("prediction_form"):
            user_values = {}

            numeric_features = [
                f for f in FEATURES if is_numeric_feature(f)
            ]
            categorical_features = [
                f for f in FEATURES if not is_numeric_feature(f)
            ]

            st.markdown("#### Material and fabrication inputs")

            if categorical_features:
                st.caption("Material, process, and categorical features")

                for start in range(0, len(categorical_features), 2):
                    group = categorical_features[start:start + 2]
                    cols = st.columns(len(group))

                    for col, feature in zip(cols, group):
                        options = get_options(feature)

                        with col:
                            if options:
                                selected = st.selectbox(
                                    feature,
                                    options=[None] + options,
                                    index=0,
                                    key=f"cat_{feature}",
                                    help="Choose a value observed in the dataset.",
                                )
                                user_values[feature] = selected
                            else:
                                entered = st.text_input(
                                    feature,
                                    key=f"text_{feature}",
                                )
                                user_values[feature] = (
                                    entered.strip() or None
                                )

            if numeric_features:
                st.markdown("#### Numeric parameters")

                for start in range(0, len(numeric_features), 2):
                    group = numeric_features[start:start + 2]
                    cols = st.columns(len(group))

                    for col, feature in zip(cols, group):
                        series = pd.to_numeric(
                            data[feature], errors="coerce"
                        ).dropna()

                        minimum = (
                            float(series.min()) if not series.empty else None
                        )
                        maximum = (
                            float(series.max()) if not series.empty else None
                        )

                        with col:
                            st.caption(feature)

                            if minimum is not None and maximum is not None:
                                st.caption(
                                    f"Observed range: {minimum:g} to {maximum:g}"
                                )

                            entered = st.number_input(
                                feature,
                                min_value=minimum,
                                max_value=maximum,
                                value=None,
                                step=0.1,
                                key=f"num_{feature}",
                                placeholder="Enter a value",
                                label_visibility="collapsed",
                            )
                            user_values[feature] = entered

            submitted = st.form_submit_button(
                "☀ Estimate Cell Efficiency",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            missing = [
                feature for feature in FEATURES
                if user_values.get(feature) is None
                or user_values.get(feature) == ""
            ]

            if missing:
                st.error(
                    "Please complete these fields: "
                    + ", ".join(missing)
                )
            else:
                input_frame = pd.DataFrame(
                    [[user_values[f] for f in FEATURES]],
                    columns=FEATURES,
                )

                # Restore numeric data types based on the training CSV.
                for feature in FEATURES:
                    if is_numeric_feature(feature):
                        input_frame[feature] = pd.to_numeric(
                            input_frame[feature], errors="coerce"
                        )

                try:
                    prediction = float(
                        np.asarray(model.predict(input_frame)).reshape(-1)[0]
                    )

                    if not np.isfinite(prediction):
                        raise ValueError("The model returned a non-finite value.")

                    st.session_state["last_prediction"] = prediction
                    st.session_state["last_input"] = input_frame.copy()

                    st.markdown("---")
                    st.markdown("### Model prediction")

                    left, right = st.columns([2, 1])

                    with left:
                        st.markdown(
                            f"""
                            <div class="panel">
                                <div class="muted">PREDICTED POWER CONVERSION EFFICIENCY</div>
                                <div class="result-number">{prediction:.2f}%</div>
                                <div class="muted">
                                    Predicted by your saved XGBoost model
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with right:
                        st.metric("Predicted PCE", f"{prediction:.2f}%")
                        if prediction < 0 or prediction > 100:
                            st.warning(
                                "The prediction is outside the physical "
                                "0–100% range. Check the model and target units."
                            )

                    st.info(
                        "This is a model estimate. Actual efficiency depends "
                        "on fabrication quality and experimental conditions."
                    )

                    with st.expander("View submitted feature values"):
                        st.dataframe(input_frame, use_container_width=True)

                except Exception as exc:
                    st.error("Prediction failed.")
                    st.code(str(exc))
                    st.info(
                        "Check that the model expects these exact columns "
                        "and that categorical values are encoded the same "
                        "way as during training."
                    )


# =========================================================
# TAB 2: OPTIMIZATION
# =========================================================

with tab_optimize:
    st.subheader("Find promising configurations")
    st.write(
        "This tool scores existing configurations in your CSV using "
        "the saved model and ranks them by predicted PCE. It does not "
        "retrain the model."
    )

    st.warning(
        "The best-ranked row is a candidate for further investigation, "
        "not proof of a globally optimal or experimentally verified cell."
    )

    if data is None:
        st.error("Upload the CSV to enable optimization.")
    else:
        available = [
            column for column in data.columns
            if column not in FEATURES and column != TARGET
        ]

        limit = st.slider(
            "Number of top candidates to display",
            min_value=5,
            max_value=25,
            value=10,
        )

        if st.button(
            "🚀 Rank Candidate Configurations",
            type="primary",
            use_container_width=True,
        ):
            try:
                candidates = data[FEATURES].copy()

                # Keep rows with observed target values for context,
                # if target is available.
                if TARGET in data.columns:
                    valid_target = pd.to_numeric(
                        data[TARGET], errors="coerce"
                    ).notna()
                    candidates = candidates.loc[valid_target].copy()
                    source_rows = data.loc[valid_target].copy()
                else:
                    source_rows = data.copy()

                if candidates.empty:
                    st.error("No candidate rows are available.")
                else:
                    # Predict in batches to avoid unnecessary memory use.
                    batch_size = 1000
                    prediction_parts = []

                    for start in range(0, len(candidates), batch_size):
                        batch = candidates.iloc[start:start + batch_size]
                        batch_predictions = model.predict(batch)
                        prediction_parts.extend(
                            np.asarray(batch_predictions).reshape(-1).tolist()
                        )

                    source_rows = source_rows.copy()
                    source_rows["Predicted PCE (%)"] = prediction_parts
                    source_rows = source_rows.sort_values(
                        "Predicted PCE (%)", ascending=False
                    )

                    st.session_state["optimization_results"] = (
                        source_rows.head(limit)
                    )

            except Exception as exc:
                st.error("Optimization could not be completed.")
                st.code(str(exc))
                st.info(
                    "The model may require preprocessing or missing-value "
                    "handling consistent with the original training pipeline."
                )

        results = st.session_state.get("optimization_results")

        if results is not None:
            best = float(results["Predicted PCE (%)"].iloc[0])

            st.markdown("### Highest-ranked candidate")
            st.markdown(
                f"""
                <div class="panel">
                    <div class="muted">TOP PREDICTED PCE</div>
                    <div class="result-number">{best:.2f}%</div>
                    <div class="muted">
                        Highest prediction among the candidates evaluated
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.dataframe(results, use_container_width=True)

            st.download_button(
                "⬇ Download ranked candidates (CSV)",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="solar_cell_optimization_candidates.csv",
                mime="text/csv",
                use_container_width=True,
            )


# =========================================================
# TAB 3: COMPARATIVE ANALYSIS
# =========================================================

with tab_compare:
    st.subheader("Prediction and performance analysis")

    if data is None or TARGET not in data.columns:
        st.info(
            "The reference CSV and its JV_default_PCE target column "
            "are required for this analysis."
        )
    else:
        st.write(
            "The following metrics compare the saved model's predictions "
            "against recorded PCE values in the available CSV."
        )

        st.warning(
            "These metrics are calculated on the available reference dataset. "
            "If this is the same data used for training, the scores may be "
            "optimistic and must not be reported as held-out test performance."
        )

        if st.button(
            "📊 Calculate Reference-Set Metrics",
            type="primary",
            use_container_width=True,
        ):
            try:
                eval_data = data.copy()
                actual = pd.to_numeric(
                    eval_data[TARGET], errors="coerce"
                )
                valid = actual.notna()

                eval_x = eval_data.loc[valid, FEATURES]
                eval_y = actual.loc[valid].astype(float)

                if len(eval_x) < 2:
                    st.error("Not enough valid rows to calculate metrics.")
                else:
                    pred_parts = []
                    batch_size = 1000

                    for start in range(0, len(eval_x), batch_size):
                        batch = eval_x.iloc[start:start + batch_size]
                        pred_parts.extend(
                            np.asarray(model.predict(batch)).reshape(-1).tolist()
                        )

                    predicted = np.asarray(pred_parts, dtype=float)

                    mae = mean_absolute_error(eval_y, predicted)
                    rmse = np.sqrt(mean_squared_error(eval_y, predicted))
                    r2 = r2_score(eval_y, predicted)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("MAE", f"{mae:.3f}")
                    m2.metric("RMSE", f"{rmse:.3f}")
                    m3.metric("R² score", f"{r2:.4f}")

                    comparison = pd.DataFrame({
                        "Actual PCE (%)": eval_y.to_numpy(),
                        "Predicted PCE (%)": predicted,
                    })
                    comparison["Absolute Error"] = (
                        comparison["Actual PCE (%)"]
                        - comparison["Predicted PCE (%)"]
                    ).abs()

                    st.markdown("#### Actual vs predicted PCE")
                    st.scatter_chart(
                        comparison,
                        x="Actual PCE (%)",
                        y="Predicted PCE (%)",
                    )

                    st.markdown("#### Sample predictions")
                    st.dataframe(
                        comparison.head(100),
                        use_container_width=True,
                    )

                    st.download_button(
                        "⬇ Download comparison results",
                        data=comparison.to_csv(index=False).encode("utf-8"),
                        file_name="solar_cell_prediction_comparison.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    st.session_state["reference_metrics"] = {
                        "MAE": mae,
                        "RMSE": rmse,
                        "R2": r2,
                    }

            except Exception as exc:
                st.error("Metrics could not be calculated.")
                st.code(str(exc))
                st.info(
                    "If your model was trained on a transformed feature matrix, "
                    "the saved pipeline must include the same preprocessing "
                    "used during training."
                )

        st.markdown("#### What the metrics mean")

        st.markdown("""
        - **MAE:** average absolute difference between predicted and actual PCE.
          Lower is better.
        - **RMSE:** penalizes larger prediction errors more heavily. Lower is better.
        - **R²:** indicates how well predictions explain variation in the target.
          A value closer to 1 is generally better.
        """)

        st.markdown("#### XGBoost versus other algorithms")

        st.write(
            "For a valid algorithm comparison, train Linear Regression and "
            "Random Forest on the same training split and evaluate all models "
            "on the same held-out test split. Use the same preprocessing and "
            "report MAE, RMSE, and R² for each model."
        )

        st.caption(
            "This app does not claim to have comparison scores for models "
            "that have not been trained and evaluated."
        )


# =========================================================
# TAB 4: PROJECT EXPLANATION
# =========================================================

with tab_about:
    st.subheader("About this project")

    st.markdown("""
    ### Objective
    Estimate solar-cell power conversion efficiency from material,
    architecture, and fabrication features.

    ### Algorithm
    **XGBoost Regression** is a gradient-boosted decision-tree method.
    It builds trees sequentially, with later trees helping correct errors
    made by earlier trees.

    ### Workflow
    1. Prepare the solar-cell dataset.
    2. Preprocess features and train XGBoost in the training environment.
    3. Save the trained model as `solar_cell_model.joblib`.
    4. Load that saved model in this Streamlit application.
    5. Predict PCE for user-provided configurations.
    6. Rank candidate configurations by predicted PCE.

    ### Optimization approach
    The current implementation ranks configurations already present in the
    CSV. This is a practical starting point. It does not establish causal
    relationships or guarantee that changing a feature will improve real
    experimental performance.

    ### Important limitations
    - Model predictions are estimates, not laboratory measurements.
    - Dataset-level metrics are not a substitute for held-out test results.
    - Input feature names, order, preprocessing, and units must match training.
    - Do not use the PCE target or other unavailable-at-design-time outcomes
      as input features.
    """)

    st.markdown("### Model files")
    st.write("Required model: `solar_cell_model.joblib`")
    st.write("Reference dataset: `Solar Cell data set (1).csv`")
    st.write("Application: `app.py`")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:15px; color:#8FA69A;">
    <b style="color:#E6BD65;">SolarIQ</b> · AI-Based Solar Cell Material Optimization<br>
    <small>Machine Learning Laboratory Project · XGBoost Regression</small>
</div>
""", unsafe_allow_html=True)
```
