
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# =====================================================
# 1. PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="SolarIQ | AI Solar Cell Optimizer",
    page_icon="☀️",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "solar_cell_model.joblib")
CSV_PATH = os.path.join(BASE_DIR, "Solar Cell data set (1).csv")
TARGET = "JV_default_PCE"


# =====================================================
# 2. WEBSITE DESIGN
# =====================================================

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


# =====================================================
# 3. LOAD MODEL AND DATASET
# =====================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "solar_cell_model.joblib not found. "
            "Upload it beside app.py in GitHub."
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
    st.error("Could not load your saved model.")
    st.code(str(error))
    st.info(
        "Check that solar_cell_model.joblib exists and that "
        "your installed package versions are compatible."
    )
    st.stop()

try:
    df = load_dataset()
except Exception as error:
    st.error("Could not load the CSV dataset.")
    st.code(str(error))
    df = None


# =====================================================
# 4. IDENTIFY INPUT FEATURES
# =====================================================

features = getattr(model, "feature_names_in_", None)

if features is not None:
    FEATURES = list(features)
elif df is not None and TARGET in df.columns:
    FEATURES = [column for column in df.columns if column != TARGET]
else:
    st.error(
        "Cannot identify the model input columns. "
        "Keep the training CSV beside app.py or save the model "
        "with its original feature names."
    )
    st.stop()

if TARGET in FEATURES:
    st.error(
        "The PCE target appears among the model input features. "
        "Check your training process to prevent target leakage."
    )
    st.stop()

if df is not None:
    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        st.error("Dataset is missing model columns: " + ", ".join(missing))
        st.stop()


def is_numeric_feature(column):
    return (
        df is not None
        and column in df.columns
        and pd.api.types.is_numeric_dtype(df[column])
    )


def get_options(column):
    if df is None or column not in df.columns:
        return []

    values = df[column].dropna().astype(str).str.strip()
    values = values[values != ""]
    return values.value_counts().index[:200].tolist()


# =====================================================
# 5. HEADER AND SUMMARY
# =====================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">AI MATERIALS RESEARCH · XGBOOST</div>
    <div class="hero-title">
        Discover better<br>solar cell performance.
    </div>
    <div class="hero-copy">
        Predict power conversion efficiency using solar-cell material
        and fabrication parameters. Compare predictions and explore
        promising candidate configurations using your saved model.
    </div>
</div>
""", unsafe_allow_html=True)

if df is not None:
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


# =====================================================
# 6. TABS
# =====================================================

tab_predict, tab_optimize, tab_compare, tab_about = st.tabs([
    "🔬 PCE Prediction",
    "🚀 Optimization",
    "📊 Model Evaluation",
    "ℹ️ About",
])


# =====================================================
# 7. PCE PREDICTION WITH DEFAULT VALUES
# =====================================================

with tab_predict:
    st.subheader("Solar cell configuration")
    st.write(
        "The fields are prefilled using values from your dataset. "
        "You can keep the defaults or change them."
    )

    if df is None:
        st.error(
            "The CSV is required for default input values. "
            "Upload Solar Cell data set (1).csv beside app.py."
        )
    else:
        with st.form("prediction_form"):
            user_values = {}

            categorical = [
                column for column in FEATURES
                if not is_numeric_feature(column)
            ]
            numeric = [
                column for column in FEATURES
                if is_numeric_feature(column)
            ]

            st.markdown("### Material and process parameters")

            for start in range(0, len(categorical), 2):
                group = categorical[start:start + 2]
                columns = st.columns(len(group))

                for container, feature in zip(columns, group):
                    options = get_options(feature)

                    with container:
                        if options:
                            user_values[feature] = st.selectbox(
                                feature,
                                options=options,
                                index=0,
                                key=f"category_{feature}",
                                help="Options are observed in the dataset.",
                            )
                        else:
                            user_values[feature] = st.text_input(
                                feature,
                                key=f"text_{feature}",
                            )

            if numeric:
                st.markdown("### Numeric parameters")

            for start in range(0, len(numeric), 2):
                group = numeric[start:start + 2]
                columns = st.columns(len(group))

                for container, feature in zip(columns, group):
                    series = pd.to_numeric(
                        df[feature], errors="coerce"
                    ).dropna()

                    with container:
                        if series.empty:
                            user_values[feature] = st.number_input(
                                feature,
                                value=0.0,
                                key=f"numeric_{feature}",
                            )
                        else:
                            minimum = float(series.min())
                            maximum = float(series.max())
                            median = float(series.median())

                            st.caption(
                                f"{feature} · observed range: "
                                f"{minimum:g}–{maximum:g}"
                            )

                            # Default value is the dataset median.
                            # Keep it within the allowed range.
                            default_value = min(
                                max(median, minimum), maximum
                            )

                            user_values[feature] = st.number_input(
                                feature,
                                min_value=minimum,
                                max_value=maximum,
                                value=default_value,
                                step=0.1,
                                key=f"numeric_{feature}",
                            )

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
                input_data = pd.DataFrame(
                    [[user_values[feature] for feature in FEATURES]],
                    columns=FEATURES,
                )

                for feature in numeric:
                    input_data[feature] = pd.to_numeric(
                        input_data[feature], errors="coerce"
                    )

                try:
                    prediction = float(
                        np.asarray(model.predict(input_data)).reshape(-1)[0]
                    )

                    if not np.isfinite(prediction):
                        raise ValueError("Model returned an invalid prediction.")

                    st.session_state["last_prediction"] = prediction
                    st.session_state["last_input"] = input_data.copy()

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
                                Output from your saved XGBoost model
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if prediction < 0 or prediction > 100:
                        st.warning(
                            "This prediction is outside the physical "
                            "0–100% range. Check your target units and model."
                        )

                    st.info(
                        "This is a prediction, not a laboratory measurement. "
                        "It does not guarantee actual cell performance."
                    )

                    with st.expander("View the input values"):
                        st.dataframe(input_data, use_container_width=True)

                except Exception as error:
                    st.error("Prediction failed.")
                    st.code(str(error))
                    st.info(
                        "The model must receive the same feature names, "
                        "data types, and preprocessing used during training."
                    )


# =====================================================
# 8. OPTIMIZATION
# =====================================================

with tab_optimize:
    st.subheader("Find promising configurations")

    st.write(
        "Score configurations already in the dataset and rank them "
        "by the PCE predicted by your saved model."
    )

    st.warning(
        "This ranks existing dataset records. It does not search all "
        "possible material combinations or guarantee experimental success."
    )

    limit = st.slider(
        "Number of candidates",
        min_value=5,
        max_value=25,
        value=10,
        key="candidate_limit",
    )

    if df is None:
        st.error("Upload the CSV to use optimization.")
    elif st.button(
        "🚀 Rank Candidate Configurations",
        type="primary",
        use_container_width=True,
    ):
        try:
            candidates = df[FEATURES].copy()

            if TARGET in df.columns:
                valid_target = pd.to_numeric(
                    df[TARGET], errors="coerce"
                ).notna()
                candidates = candidates.loc[valid_target].copy()
                source_rows = df.loc[valid_target].copy()
            else:
                source_rows = df.copy()

            if candidates.empty:
                st.error("No candidate rows available.")
            else:
                predictions = []

                for start in range(0, len(candidates), 1000):
                    batch = candidates.iloc[start:start + 1000]
                    predictions.extend(
                        np.asarray(model.predict(batch)).reshape(-1).tolist()
                    )

                source_rows = source_rows.copy()
                source_rows["Predicted PCE (%)"] = predictions

                ranked = source_rows.sort_values(
                    "Predicted PCE (%)",
                    ascending=False,
                ).head(limit)

                st.session_state["ranked_candidates"] = ranked

        except Exception as error:
            st.error("Optimization failed.")
            st.code(str(error))

    ranked = st.session_state.get("ranked_candidates")

    if ranked is not None:
        best_prediction = float(ranked["Predicted PCE (%)"].iloc[0])

        st.markdown(
            f"""
            <div class="panel">
                <div class="muted">HIGHEST PREDICTED PCE IN CANDIDATES</div>
                <div class="result-number">{best_prediction:.2f}%</div>
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


# =====================================================
# 9. MODEL EVALUATION
# =====================================================

with tab_compare:
    st.subheader("Evaluate the saved XGBoost model")

    st.write(
        "Compare model predictions with recorded PCE values in the CSV."
    )

    if df is None or TARGET not in df.columns:
        st.info(
            f"The dataset must contain the target column {TARGET} "
            "to calculate these metrics."
        )
    else:
        st.warning(
            "If this CSV was used to train your model, these are not "
            "unseen test-set scores. Use a separate test set for a fair "
            "assessment of generalization."
        )

        if st.button(
            "📊 Calculate Metrics",
            type="primary",
            use_container_width=True,
        ):
            try:
                actual = pd.to_numeric(df[TARGET], errors="coerce")
                valid = actual.notna()

                X_eval = df.loc[valid, FEATURES]
                y_eval = actual.loc[valid].astype(float)

                if len(X_eval) < 2:
                    raise ValueError("Not enough valid rows for evaluation.")

                predictions = []

                for start in range(0, len(X_eval), 1000):
                    batch = X_eval.iloc[start:start + 1000]
                    predictions.extend(
                        np.asarray(model.predict(batch)).reshape(-1).tolist()
                    )

                predictions = np.asarray(predictions)

                mae = mean_absolute_error(y_eval, predictions)
                rmse = np.sqrt(mean_squared_error(y_eval, predictions))
                r2 = r2_score(y_eval, predictions)

                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{mae:.3f}")
                col2.metric("RMSE", f"{rmse:.3f}")
                col3.metric("R² score", f"{r2:.4f}")

                comparison = pd.DataFrame({
                    "Actual PCE (%)": y_eval.to_numpy(),
                    "Predicted PCE (%)": predictions,
                })
                comparison["Absolute Error"] = (
                    comparison["Actual PCE (%)"]
                    - comparison["Predicted PCE (%)"]
                ).abs()

                st.markdown("### Actual vs predicted")
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


# =====================================================
# 10. ABOUT THE PROJECT
# =====================================================

with tab_about:
    st.subheader("Project overview")

    st.markdown("""
    **Objective:** Predict solar-cell power conversion efficiency (PCE)
    from material and fabrication features.

    **Algorithm:** XGBoost Regression, a gradient-boosted decision-tree
    algorithm that builds trees sequentially to improve predictions.

    **Implementation workflow:**
    1. Prepare the solar-cell dataset.
    2. Train XGBoost in Google Colab.
    3. Save the trained model as a Joblib file.
    4. Load the saved model in this Streamlit website.
    5. Predict PCE for a configuration.
    6. Rank existing configurations by predicted PCE.

    **Limitations:**
    - Predicted results are estimates, not experimental measurements.
    - Dataset-level evaluation may be optimistic if the same data was
      used for training.
    - Ranking existing configurations is not proof of a global optimum.
    - Input columns and preprocessing must match the training process.
    """)

    st.markdown("### Required files")
    st.code(
        "app.py\n"
        "solar_cell_model.joblib\n"
        "Solar Cell data set (1).csv\n"
        "requirements.txt"
    )


# =====================================================
# 11. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:15px; color:#8FA69A;">
    <b style="color:#E6BD65;">SolarIQ</b><br>
    AI-Based Solar Cell Material Optimization using XGBoost<br>
    <small>Machine Learning Laboratory Project</small>
</div>
""", unsafe_allow_html=True)
