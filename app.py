
import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SolarIQ | AI Solar Cell Optimizer",
    page_icon="☀️",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "solar_cell_model.joblib")
CSV_PATH = os.path.join(BASE_DIR, "Solar Cell data set (1).csv")
TARGET = "JV_default_PCE"


# =========================================================
# 2. PREMIUM WEBSITE DESIGN
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
    font-family: 'Space Grotesk', sans-serif !important;
    color: #F3F4EC;
}

.hero {
    padding: 34px;
    border: 1px solid #29483B;
    border-radius: 22px;
    background: radial-gradient(circle at 85% 10%, #294B35, transparent 42%),
                linear-gradient(135deg, #10251C, #0B1914);
    margin-bottom: 25px;
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
    line-height: 1.1;
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
    padding: 18px;
}

div[data-testid="stMetricValue"] {
    color: #E6BD65;
}

div[data-testid="stMetricLabel"] {
    color: #B5C7BC;
}

.panel {
    background: #10231B;
    border: 1px solid #29483B;
    border-radius: 16px;
    padding: 24px;
    margin: 10px 0;
}

.result-number {
    color: #E6BD65;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 50px;
    font-weight: 700;
}

.muted {
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
# 3. LOAD MODEL AND DATA
# =========================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "solar_cell_model.joblib not found. "
            "Upload it to the same GitHub folder as app.py."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if not os.path.exists(CSV_PATH):
        return None
    return pd.read_csv(CSV_PATH, low_memory=False)


try:
    model = load_model()
except Exception as error:
    st.error("Could not load the saved model.")
    st.code(str(error))
    st.info(
        "Check that solar_cell_model.joblib is in your repository "
        "and your package versions are compatible."
    )
    st.stop()

try:
    df = load_dataset()
except Exception as error:
    st.error("Could not load the CSV dataset.")
    st.code(str(error))
    st.stop()

if df is None:
    st.error(
        "Dataset not found. Upload 'Solar Cell data set (1).csv' "
        "beside app.py in GitHub."
    )
    st.stop()

if TARGET not in df.columns:
    st.error(f"Target column '{TARGET}' was not found in the CSV.")
    st.stop()


# =========================================================
# 4. GET THE MODEL'S INPUT FEATURES
# =========================================================

model_features = getattr(model, "feature_names_in_", None)

if model_features is not None:
    FEATURES = list(model_features)
else:
    FEATURES = [column for column in df.columns if column != TARGET]

if TARGET in FEATURES:
    st.error(
        "The target PCE column appears among the model input features. "
        "Check your original training process for target leakage."
    )
    st.stop()

missing_columns = [c for c in FEATURES if c not in df.columns]

if missing_columns:
    st.error(
        "The CSV is missing model input columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# =========================================================
# 5. SAFELY DETECT NUMERIC AND CATEGORICAL FEATURES
# =========================================================

def parse_number(value):
    """
    Parse a single number or a pipe-separated numeric list.

    Example:
        '10.0'        -> 10.0
        '10.0 | 100.0' -> 55.0

    IMPORTANT: the average of a list is only a technical fallback.
    It may not represent the physical meaning of that feature.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()

    if not text:
        return np.nan

    # Accept a single number.
    try:
        return float(text)
    except ValueError:
        pass

    # Handle pipe-separated numeric values such as "10.0 | 100.0".
    parts = [part.strip() for part in text.split("|")]

    if len(parts) > 1:
        numbers = []

        for part in parts:
            try:
                numbers.append(float(part))
            except ValueError:
                return np.nan

        if numbers:
            return float(np.mean(numbers))

    return np.nan


def numeric_coverage(series):
    """Return the fraction of non-empty values that parse as numbers."""
    cleaned = series.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]

    if cleaned.empty:
        return 0.0

    parsed = cleaned.map(parse_number)
    return float(parsed.notna().mean())


def is_numeric_feature(column):
    """
    Treat a column as numeric only if all its observed values
    can be interpreted numerically, including numeric lists.
    """
    if column not in df.columns:
        return False

    series = df[column]
    return numeric_coverage(series) == 1.0 and series.notna().any()


NUMERIC_FEATURES = [c for c in FEATURES if is_numeric_feature(c)]
CATEGORICAL_FEATURES = [c for c in FEATURES if c not in NUMERIC_FEATURES]


def get_options(column):
    values = df[column].dropna().astype(str).str.strip()
    values = values[values != ""]
    return values.value_counts().index[:200].tolist()


def clean_model_inputs(frame):
    """
    Clean numeric-looking columns consistently before prediction.
    Columns such as '10.0 | 100.0' are converted to their mean.
    Other columns remain categorical strings.
    """
    frame = frame.copy()

    for column in FEATURES:
        if column in NUMERIC_FEATURES:
            frame[column] = frame[column].map(parse_number)
        else:
            frame[column] = frame[column].where(
                frame[column].notna(), np.nan
            )
            frame[column] = frame[column].map(
                lambda x: str(x).strip() if pd.notna(x) else np.nan
            )

    return frame[FEATURES]


# =========================================================
# 6. HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">AI MATERIALS RESEARCH · XGBOOST</div>
    <div class="hero-title">
        Discover better<br>solar cell performance.
    </div>
    <div class="hero-copy">
        Predict power conversion efficiency from material and fabrication
        parameters. Compare model predictions and explore promising
        configurations using your saved model.
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Dataset records", f"{len(df):,}")
c2.metric("Model input features", len(FEATURES))
c3.metric("Prediction target", "PCE (%)")

with st.sidebar:
    st.markdown("## ☀️ SolarIQ")
    st.caption("AI solar materials research")
    st.divider()
    st.markdown("### Project details")
    st.write("Algorithm: XGBoost Regression")
    st.write("Target: Power Conversion Efficiency")
    st.write("Model: Saved Joblib file")
    st.divider()
    st.caption(
        "Predictions are estimates and must be experimentally validated."
    )


# =========================================================
# 7. TABS
# =========================================================

tab_predict, tab_optimize, tab_compare, tab_about = st.tabs([
    "🔬 PCE Prediction",
    "🚀 Optimization",
    "📊 Model Evaluation",
    "ℹ️ About",
])


# =========================================================
# 8. PCE PREDICTION FORM WITH DEFAULTS
# =========================================================

with tab_predict:
    st.subheader("Solar cell configuration")
    st.write(
        "Inputs are prefilled from your dataset. Change them if needed, "
        "then click Estimate Cell Efficiency."
    )

    with st.form("prediction_form"):
        user_values = {}

        st.markdown("### Material and process parameters")

        for start in range(0, len(CATEGORICAL_FEATURES), 2):
            group = CATEGORICAL_FEATURES[start:start + 2]
            columns = st.columns(len(group))

            for container, feature in zip(columns, group):
                options = get_options(feature)

                with container:
                    if options:
                        # Default to the most frequent observed value.
                        user_values[feature] = st.selectbox(
                            feature,
                            options=options,
                            index=0,
                            key=f"cat_{feature}",
                        )
                    else:
                        user_values[feature] = st.text_input(
                            feature,
                            value="",
                            key=f"text_{feature}",
                        )

        if NUMERIC_FEATURES:
            st.markdown("### Numeric parameters")

        for start in range(0, len(NUMERIC_FEATURES), 2):
            group = NUMERIC_FEATURES[start:start + 2]
            columns = st.columns(len(group))

            for container, feature in zip(columns, group):
                series = df[feature].map(parse_number).dropna()

                with container:
                    if series.empty:
                        user_values[feature] = st.number_input(
                            feature,
                            value=0.0,
                            key=f"num_{feature}",
                        )
                    else:
                        minimum = float(series.min())
                        maximum = float(series.max())
                        median = float(series.median())
                        default_value = min(max(median, minimum), maximum)

                        st.caption(
                            f"Observed range: {minimum:g}–{maximum:g}"
                        )

                        user_values[feature] = st.number_input(
                            feature,
                            min_value=minimum,
                            max_value=maximum,
                            value=default_value,
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
            or user_values.get(f) == ""
        ]

        if missing:
            st.error("Complete these fields: " + ", ".join(missing))
        else:
            input_frame = pd.DataFrame(
                [[user_values[f] for f in FEATURES]],
                columns=FEATURES,
            )

            input_frame = clean_model_inputs(input_frame)

            try:
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
                    <div class="panel">
                        <div class="muted">
                            PREDICTED POWER CONVERSION EFFICIENCY
                        </div>
                        <div class="result-number">{prediction:.2f}%</div>
                        <div class="muted">
                            Prediction from your saved XGBoost model
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if prediction < 0 or prediction > 100:
                    st.warning(
                        "This result is outside the physical 0–100% range. "
                        "Check your target units and model."
                    )

                st.info(
                    "This is a model estimate, not a laboratory measurement."
                )

                with st.expander("View submitted input values"):
                    st.dataframe(input_frame, use_container_width=True)

            except Exception as error:
                st.error("Prediction failed.")
                st.code(str(error))
                st.info(
                    "Your saved model's preprocessing and feature types "
                    "must match the input format used during training."
                )


# =========================================================
# 9. OPTIMIZATION: RANK EXISTING DATASET ROWS
# =========================================================

with tab_optimize:
    st.subheader("Find promising configurations")
    st.write(
        "Score configurations from your dataset and rank them by "
        "predicted PCE using the saved model."
    )

    st.warning(
        "This ranks existing dataset records. It does not guarantee "
        "a globally optimal or experimentally verified solar cell."
    )

    limit = st.slider(
        "Number of candidates",
        min_value=5,
        max_value=25,
        value=10,
        key="candidate_limit",
    )

    if st.button(
        "🚀 Rank Candidate Configurations",
        type="primary",
        use_container_width=True,
    ):
        try:
            candidates = df[FEATURES].copy()
            candidates = clean_model_inputs(candidates)

            # The saved pipeline may impute missing values itself.
            # If it cannot, report the precise error rather than hide it.
            predictions = []

            for start in range(0, len(candidates), 500):
                batch = candidates.iloc[start:start + 500]
                batch_predictions = model.predict(batch)
                predictions.extend(
                    np.asarray(batch_predictions).reshape(-1).tolist()
                )

            ranked = df.copy()
            ranked["Predicted PCE (%)"] = predictions
            ranked = ranked.sort_values(
                "Predicted PCE (%)", ascending=False
            ).head(limit)

            st.session_state["ranked_candidates"] = ranked

        except Exception as error:
            st.error("Optimization failed.")
            st.code(str(error))
            st.info(
                "If this error mentions SimpleImputer, the preprocessing "
                "stored in your model may not match the cleaned inputs. "
                "The training preprocessing may need to be corrected."
            )

    ranked = st.session_state.get("ranked_candidates")

    if ranked is not None:
        best = float(ranked["Predicted PCE (%)"].iloc[0])

        st.markdown(
            f"""
            <div class="panel">
                <div class="muted">HIGHEST PREDICTED PCE IN CANDIDATES</div>
                <div class="result-number">{best:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(ranked, use_container_width=True)

        st.download_button(
            "Download ranked candidates CSV",
            data=ranked.to_csv(index=False).encode("utf-8"),
            file_name="solar_cell_optimization_candidates.csv",
            mime="text/csv",
        )


# =========================================================
# 10. MODEL EVALUATION
# =========================================================

with tab_compare:
    st.subheader("Evaluate the saved XGBoost model")

    st.write(
        "Compare the model's predictions with recorded PCE values "
        "from the dataset."
    )

    st.warning(
        "If the same CSV was used to train the model, these are reference-set "
        "metrics, not unseen test-set scores."
    )

    if st.button(
        "📊 Calculate Metrics",
        type="primary",
        use_container_width=True,
    ):
        try:
            actual = pd.to_numeric(df[TARGET], errors="coerce")
            valid = actual.notna()

            X_eval = clean_model_inputs(df.loc[valid, FEATURES])
            y_eval = actual.loc[valid].astype(float)

            predictions = []

            for start in range(0, len(X_eval), 500):
                batch = X_eval.iloc[start:start + 500]
                predictions.extend(
                    np.asarray(model.predict(batch)).reshape(-1).tolist()
                )

            predictions = np.asarray(predictions, dtype=float)

            mae = mean_absolute_error(y_eval, predictions)
            rmse = np.sqrt(mean_squared_error(y_eval, predictions))
            r2 = r2_score(y_eval, predictions)

            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{mae:.3f}")
            c2.metric("RMSE", f"{rmse:.3f}")
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

            st.dataframe(
                comparison.head(100),
                use_container_width=True,
            )

            st.download_button(
                "Download evaluation CSV",
                data=comparison.to_csv(index=False).encode("utf-8"),
                file_name="xgboost_evaluation.csv",
                mime="text/csv",
            )

        except Exception as error:
            st.error("Could not calculate evaluation metrics.")
            st.code(str(error))
            st.info(
                "The saved model may contain a preprocessing pipeline that "
                "expects different input types or column names."
            )


# =========================================================
# 11. ABOUT
# =========================================================

with tab_about:
    st.subheader("About the project")

    st.markdown("""
    **Objective:** Predict solar-cell power conversion efficiency (PCE)
    from material and fabrication parameters.

    **Algorithm:** XGBoost Regression. XGBoost builds decision trees
    sequentially, with later trees helping correct earlier prediction errors.

    **Workflow:**
    1. Prepare the solar-cell dataset.
    2. Train XGBoost in Google Colab.
    3. Save the model as a Joblib file.
    4. Load the saved model in Streamlit.
    5. Predict PCE for a selected configuration.
    6. Rank existing dataset configurations by predicted PCE.

    **Limitations:**
    - Predictions are estimates, not experimental measurements.
    - Reference-set metrics may be optimistic if training used the same data.
    - Ranking existing rows does not guarantee a global optimum.
    - Model preprocessing must match the original training procedure.
    """)

    st.markdown("### Required repository files")
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
<div style="text-align:center; padding:15px; color:#8FA69A;">
    <b style="color:#E6BD65;">SolarIQ</b><br>
    AI-Based Solar Cell Material Optimization using XGBoost<br>
    <small>Machine Learning Laboratory Project</small>
</div>
""", unsafe_allow_html=True)
