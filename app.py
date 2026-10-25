
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SolarAI | Solar Cell Optimization",
    page_icon="☀️",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "solar_cell_model.joblib")
CSV_PATH = os.path.join(BASE_DIR, "Solar Cell data set (1).csv")
TARGET = "JV_default_PCE"


# =========================================================
# 2. DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background: #091713;
    color: #F3F4EC;
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    color: #F3F4EC;
    font-family: 'Space Grotesk', sans-serif !important;
}

.hero {
    padding: 34px;
    border: 1px solid #29483B;
    border-radius: 22px;
    background:
        radial-gradient(circle at 85% 10%, #294B35, transparent 42%),
        linear-gradient(135deg, #10251C, #0B1914);
    margin-bottom: 24px;
}

.eyebrow {
    color: #E6BD65;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(32px, 5vw, 52px);
    font-weight: 700;
    line-height: 1.12;
    margin: 16px 0;
}

.hero-copy {
    color: #B5C7BC;
    line-height: 1.8;
    max-width: 760px;
}

div[data-testid="stMetric"] {
    background: #10231B;
    border: 1px solid #29483B;
    border-radius: 14px;
    padding: 16px;
}

div[data-testid="stMetricValue"] {
    color: #E6BD65;
}

div[data-testid="stMetricLabel"] {
    color: #B5C7BC;
}

.result-panel {
    background: #10231B;
    border: 1px solid #456B4D;
    border-radius: 18px;
    padding: 26px;
    margin: 12px 0;
}

.result-number {
    color: #E6BD65;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 48px;
    font-weight: 700;
}

.small-muted {
    color: #A9BDB0;
    font-size: 13px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: #E6BD65 !important;
    color: #102017 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 44px;
}

section[data-testid="stSidebar"] {
    background: #0D2018;
    border-right: 1px solid #29483B;
}

hr {
    border-color: #29483B;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD SAVED MODEL AND CSV
# =========================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "solar_cell_model.joblib was not found. "
            "Upload it beside app.py in your GitHub repository."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if not os.path.exists(CSV_PATH):
        return None
    return pd.read_csv(CSV_PATH, low_memory=False)


try:
    model = load_model()
except Exception as exc:
    st.error("The saved model could not be loaded.")
    st.code(str(exc))
    st.info(
        "Check the model file and use compatible scikit-learn, "
        "XGBoost and joblib versions."
    )
    st.stop()

try:
    df = load_dataset()
except Exception as exc:
    st.error("The dataset could not be loaded.")
    st.code(str(exc))
    st.stop()

if df is None:
    st.error(
        "Dataset not found. Upload 'Solar Cell data set (1).csv' "
        "to the same repository folder as app.py."
    )
    st.stop()

if TARGET not in df.columns:
    st.error(f"Expected target column '{TARGET}' is missing.")
    st.stop()


# =========================================================
# 4. IDENTIFY FEATURES
# =========================================================

model_feature_names = getattr(model, "feature_names_in_", None)

if model_feature_names is not None:
    FEATURES = list(model_feature_names)
else:
    FEATURES = [c for c in df.columns if c != TARGET]

if TARGET in FEATURES:
    st.error(
        "The target column is included in the model inputs. "
        "Check your original training code for target leakage."
    )
    st.stop()

missing_features = [c for c in FEATURES if c not in df.columns]

if missing_features:
    st.error(
        "These model features are missing from the CSV: "
        + ", ".join(missing_features)
    )
    st.stop()


# =========================================================
# 5. FEATURE TYPES AND DEFAULTS
# =========================================================

def is_numeric_feature(column):
    """
    Only columns whose actual CSV dtype is numeric are numeric.
    Text such as '10.0 | 100.0' stays categorical.
    Boolean columns stay Boolean.
    """
    return (
        pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
    )


NUMERIC_FEATURES = [c for c in FEATURES if is_numeric_feature(c)]
CATEGORICAL_FEATURES = [c for c in FEATURES if c not in NUMERIC_FEATURES]


def category_values(column):
    """Return actual non-missing values in their original types."""
    values = df[column].dropna().unique().tolist()
    return values


def category_label(value):
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    return str(value)


def category_default(column):
    """Most frequent non-missing value from the dataset."""
    values = df[column].dropna()

    if values.empty:
        return None

    modes = values.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]

    return values.iloc[0]


def numeric_default(column):
    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        return 0.0, 0.0, 0.0

    return (
        float(values.min()),
        float(values.max()),
        float(values.median()),
    )


def validate_input_frame(frame):
    """
    Preserve original categorical values and numeric column types.
    Do not turn pipe-separated text or Boolean values into arbitrary
    numbers. The saved model must accept this original representation.
    """
    frame = frame[FEATURES].copy()

    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


# =========================================================
# 6. HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">MACHINE LEARNING · SOLAR MATERIALS</div>
    <div class="hero-title">SolarAI<br>Smarter Solar Cell Research.</div>
    <div class="hero-copy">
        Predict solar-cell power conversion efficiency, explore promising
        configurations, and evaluate model performance using your saved
        XGBoost model.
    </div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("Dataset records", f"{len(df):,}")
m2.metric("Model input features", len(FEATURES))
m3.metric("Prediction target", "PCE (%)")

with st.sidebar:
    st.markdown("## ☀️ SolarAI")
    st.caption("AI-Based Solar Cell Material Optimization")
    st.divider()
    st.markdown("### Model information")
    st.write("Algorithm: XGBoost Regression")
    st.write("Output: Power Conversion Efficiency")
    st.write("Model source: Saved Joblib file")
    st.divider()
    st.caption("Predictions are estimates, not experimental guarantees.")


# =========================================================
# 7. APPLICATION TABS
# =========================================================

tab_predict, tab_optimize, tab_evaluate, tab_about = st.tabs([
    "🔬 PCE Prediction",
    "🚀 Optimization",
    "📊 Model Evaluation",
    "ℹ️ About",
])


# =========================================================
# 8. PCE PREDICTION
# =========================================================

with tab_predict:
    st.subheader("Solar cell configuration")

    st.write(
        "The form starts with default values from your dataset. "
        "You can change the inputs before predicting."
    )

    with st.form("solar_prediction_form"):
        user_values = {}

        st.markdown("### Material and process features")

        for start in range(0, len(CATEGORICAL_FEATURES), 2):
            group = CATEGORICAL_FEATURES[start:start + 2]
            columns = st.columns(len(group))

            for container, feature in zip(columns, group):
                options = category_values(feature)

                with container:
                    if options:
                        default = category_default(feature)

                        # Find the matching default option.
                        default_index = next(
                            (
                                i for i, value in enumerate(options)
                                if (
                                    type(value) is type(default)
                                    and value == default
                                )
                            ),
                            0,
                        )

                        selected = st.selectbox(
                            feature,
                            options=options,
                            index=default_index,
                            format_func=category_label,
                            key=f"cat_{feature}",
                        )

                        user_values[feature] = selected
                    else:
                        user_values[feature] = st.text_input(
                            feature,
                            value="",
                            key=f"text_{feature}",
                        )

        if NUMERIC_FEATURES:
            st.markdown("### Numeric features")

        for start in range(0, len(NUMERIC_FEATURES), 2):
            group = NUMERIC_FEATURES[start:start + 2]
            columns = st.columns(len(group))

            for container, feature in zip(columns, group):
                minimum, maximum, median = numeric_default(feature)

                with container:
                    st.caption(
                        f"Observed range: {minimum:g} to {maximum:g}"
                    )

                    if minimum == maximum:
                        user_values[feature] = st.number_input(
                            feature,
                            value=median,
                            key=f"num_{feature}",
                        )
                    else:
                        user_values[feature] = st.number_input(
                            feature,
                            min_value=minimum,
                            max_value=maximum,
                            value=median,
                            step=0.1,
                            key=f"num_{feature}",
                        )

        submitted = st.form_submit_button(
            "☀ Estimate Cell Efficiency",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        missing = [
            f for f in FEATURES
            if user_values.get(f) is None
            or (
                isinstance(user_values.get(f), str)
                and user_values.get(f).strip() == ""
            )
        ]

        if missing:
            st.error("Please complete: " + ", ".join(missing))
        else:
            try:
                input_frame = pd.DataFrame(
                    [[user_values[f] for f in FEATURES]],
                    columns=FEATURES,
                )

                input_frame = validate_input_frame(input_frame)

                prediction = float(
                    np.asarray(model.predict(input_frame)).reshape(-1)[0]
                )

                if not np.isfinite(prediction):
                    raise ValueError("The model returned an invalid value.")

                st.session_state["last_prediction"] = prediction
                st.session_state["last_input"] = input_frame.copy()

                st.markdown("---")
                st.subheader("Prediction result")

                st.markdown(
                    f"""
                    <div class="result-panel">
                        <div class="small-muted">
                            PREDICTED POWER CONVERSION EFFICIENCY
                        </div>
                        <div class="result-number">{prediction:.2f}%</div>
                        <div class="small-muted">
                            Output from your saved XGBoost model
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if prediction < 0 or prediction > 100:
                    st.warning(
                        "The prediction is outside 0–100%. Check the target "
                        "units and the training process."
                    )

                st.info(
                    "This result is a model estimate. Actual efficiency "
                    "requires experimental validation."
                )

                with st.expander("View input values"):
                    st.dataframe(input_frame, use_container_width=True)

            except Exception as exc:
                st.error("Prediction failed.")
                st.code(str(exc))
                st.warning(
                    "If the error mentions converting a string or Boolean "
                    "to float, your saved model may expect encoded features "
                    "rather than the raw CSV columns. Use the same feature "
                    "preprocessing from your original Colab training code."
                )


# =========================================================
# 9. OPTIMIZATION
# =========================================================

with tab_optimize:
    st.subheader("Rank promising solar-cell configurations")

    st.write(
        "The saved model scores existing rows from the dataset and ranks "
        "them by predicted PCE. No new model is trained."
    )

    st.warning(
        "This is a ranking of existing records, not a search of every "
        "possible material combination. A high prediction is not proof "
        "of a real-world improvement."
    )

    number_of_candidates = st.slider(
        "Number of candidates to show",
        min_value=5,
        max_value=25,
        value=10,
        key="number_of_candidates",
    )

    if st.button(
        "🚀 Rank Candidate Configurations",
        type="primary",
        use_container_width=True,
    ):
        try:
            candidate_inputs = validate_input_frame(df[FEATURES].copy())

            predictions = []

            for start in range(0, len(candidate_inputs), 500):
                batch = candidate_inputs.iloc[start:start + 500]
                batch_predictions = model.predict(batch)
                predictions.extend(
                    np.asarray(batch_predictions).reshape(-1).tolist()
                )

            ranked = df.copy()
            ranked["Predicted PCE (%)"] = predictions

            ranked = ranked.sort_values(
                "Predicted PCE (%)",
                ascending=False,
            ).head(number_of_candidates)

            st.session_state["ranked_candidates"] = ranked

        except Exception as exc:
            st.error("Optimization failed.")
            st.code(str(exc))
            st.warning(
                "Optimization uses raw dataset features. If the model "
                "expects transformed inputs, use the preprocessing pipeline "
                "saved with the model during training."
            )

    ranked = st.session_state.get("ranked_candidates")

    if ranked is not None:
        best_pce = float(ranked["Predicted PCE (%)"].iloc[0])

        st.markdown(
            f"""
            <div class="result-panel">
                <div class="small-muted">TOP PREDICTED PCE</div>
                <div class="result-number">{best_pce:.2f}%</div>
                <div class="small-muted">
                    Highest model prediction among the ranked records
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(ranked, use_container_width=True)

        st.download_button(
            "⬇ Download ranked candidates",
            data=ranked.to_csv(index=False).encode("utf-8"),
            file_name="solarai_ranked_candidates.csv",
            mime="text/csv",
        )


# =========================================================
# 10. MODEL EVALUATION
# =========================================================

with tab_evaluate:
    st.subheader("Evaluate XGBoost prediction performance")

    st.write(
        "Calculate error metrics by comparing model predictions with "
        "the recorded PCE values in the dataset."
    )

    st.warning(
        "If the model was trained using this same CSV, these scores are "
        "reference-set metrics, not independent test-set performance."
    )

    if st.button(
        "📊 Calculate Evaluation Metrics",
        type="primary",
        use_container_width=True,
    ):
        try:
            actual = pd.to_numeric(df[TARGET], errors="coerce")
            valid = actual.notna()

            X_eval = validate_input_frame(df.loc[valid, FEATURES].copy())
            y_eval = actual.loc[valid].astype(float)

            if len(X_eval) < 2:
                raise ValueError("Not enough valid target rows.")

            predictions = []

            for start in range(0, len(X_eval), 500):
                batch = X_eval.iloc[start:start + 500]
                batch_predictions = model.predict(batch)
                predictions.extend(
                    np.asarray(batch_predictions).reshape(-1).tolist()
                )

            predictions = np.asarray(predictions, dtype=float)

            mae = mean_absolute_error(y_eval, predictions)
            rmse = np.sqrt(mean_squared_error(y_eval, predictions))
            r2 = r2_score(y_eval, predictions)

            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{mae:.4f}")
            c2.metric("RMSE", f"{rmse:.4f}")
            c3.metric("R² score", f"{r2:.4f}")

            comparison = pd.DataFrame({
                "Actual PCE (%)": y_eval.to_numpy(),
                "Predicted PCE (%)": predictions,
            })

            comparison["Absolute Error"] = (
                comparison["Actual PCE (%)"]
                - comparison["Predicted PCE (%)"]
            ).abs()

            st.markdown("### Actual vs predicted PCE")

            st.scatter_chart(
                comparison,
                x="Actual PCE (%)",
                y="Predicted PCE (%)",
            )

            st.markdown("### Prediction examples")
            st.dataframe(
                comparison.head(100),
                use_container_width=True,
            )

            st.download_button(
                "⬇ Download evaluation results",
                data=comparison.to_csv(index=False).encode("utf-8"),
                file_name="solarai_evaluation_results.csv",
                mime="text/csv",
            )

            st.markdown("### How to interpret these metrics")
            st.write("**MAE:** average absolute prediction error; lower is better.")
            st.write("**RMSE:** penalizes larger errors; lower is better.")
            st.write("**R²:** closer to 1 is generally better.")

        except Exception as exc:
            st.error("Model evaluation failed.")
            st.code(str(exc))
            st.warning(
                "Check that the saved model uses the same feature names, "
                "data types, and preprocessing as the CSV."
            )


# =========================================================
# 11. ABOUT THE PROJECT
# =========================================================

with tab_about:
    st.subheader("About SolarAI")

    st.markdown("""
    **Project:** AI-Based Solar Cell Material Optimization Using XGBoost

    **Objective:** Predict solar-cell Power Conversion Efficiency (PCE)
    from material and fabrication parameters.

    **Algorithm:** XGBoost Regression is a gradient-boosting algorithm
    that combines decision trees to predict continuous values.

    **Workflow:**
    1. Prepare the solar-cell dataset.
    2. Train XGBoost in Google Colab.
    3. Save the trained model in a Joblib file.
    4. Load the saved model in this Streamlit website.
    5. Predict PCE for selected inputs.
    6. Rank existing configurations by predicted PCE.

    **Limitations:**
    - Predictions are estimates, not laboratory measurements.
    - Evaluation on training data may overstate model performance.
    - Ranking existing records does not guarantee a global optimum.
    - The deployed model must use the same preprocessing as training.
    """)

    st.markdown("### Required files")
    st.code(
        "app.py\n"
        "solar_cell_model.joblib\n"
        "Solar Cell data set (1).csv\n"
        "requirements.txt"
    )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:16px; color:#8FA69A;">
    <b style="color:#E6BD65;">SolarAI</b><br>
    AI-Based Solar Cell Material Optimization Using XGBoost<br>
    <small>Machine Learning Laboratory Project</small>
</div>
""", unsafe_allow_html=True)
