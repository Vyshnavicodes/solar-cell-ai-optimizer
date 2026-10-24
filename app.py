from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "JV_default_PCE"
LEAKAGE_COLUMNS = ["JV_default_Voc", "JV_default_Jsc", "JV_default_FF", "JV_measured"]
DEFAULT_DATASET = "Solar Cell data set (1).csv"

st.set_page_config(page_title="SolarCell AI | Material Intelligence", page_icon="☀️", layout="wide", initial_sidebar_state="expanded")

# Solar-inspired visual identity: forest green, warm ivory, and solar gold.
st.markdown("""
<style>
:root { --forest: #102B2A; --forest2: #1D4840; --gold: #D9A441; --sage: #E8F0E9; --ivory: #F8F7F2; --ink: #173532; }
.stApp { background: var(--ivory); color: var(--ink); }
[data-testid="stHeader"] { background: rgba(248,247,242,0.92); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #102B2A 0%, #183C35 100%); }
[data-testid="stSidebar"] * { color: #F8F7F2 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] section { background: rgba(255,255,255,0.08); }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1240px; }
.hero { background: linear-gradient(120deg, #102B2A 0%, #1D4840 72%, #315D4B 100%); padding: 2.2rem 2.4rem; border-radius: 24px; color: #F8F7F2; margin-bottom: 1.4rem; border: 1px solid rgba(217,164,65,.35); box-shadow: 0 14px 35px rgba(16,43,42,.12); }
.hero-kicker { color: #E7BD67; text-transform: uppercase; letter-spacing: .16em; font-size: .76rem; font-weight: 700; margin-bottom: .65rem; }
.hero h1 { color: #FFFFFF; font-size: clamp(2rem, 4vw, 3.15rem); line-height: 1.08; letter-spacing: -.035em; margin: 0 0 .8rem 0; }
.hero p { color: #D5E2DC; font-size: 1.03rem; max-width: 760px; margin-bottom: .2rem; }
.pill { display: inline-block; border: 1px solid rgba(217,164,65,.6); color: #F0D28D; border-radius: 999px; padding: .32rem .7rem; font-size: .75rem; margin: .9rem .35rem 0 0; }
.section-label { color: #B48628; text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; font-weight: 800; margin-bottom: .25rem; }
.panel { background: white; border: 1px solid #DCE5DC; border-radius: 18px; padding: 1.1rem 1.25rem; }
h2, h3 { color: #173532; letter-spacing: -.02em; }
[data-testid="stMetric"] { background: white; border: 1px solid #DCE5DC; padding: 1rem 1.1rem; border-radius: 16px; box-shadow: 0 4px 14px rgba(16,43,42,.035); }
[data-testid="stMetricLabel"] { color: #61756A; }
[data-testid="stMetricValue"] { color: #1D4840; }
.stButton > button, .stFormSubmitButton > button, [data-testid="stDownloadButton"] button { background: #D9A441; color: #102B2A; border: 0; border-radius: 11px; font-weight: 750; min-height: 2.8rem; }
.stButton > button:hover, .stFormSubmitButton > button:hover, [data-testid="stDownloadButton"] button:hover { background: #E7BD67; color: #102B2A; border: 0; }
[data-baseweb="select"] > div, [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input { border-radius: 10px; }
[data-testid="stDataFrame"] { border: 1px solid #DCE5DC; border-radius: 14px; overflow: hidden; }
hr { border-color: #DCE5DC; }
footer { visibility: hidden; }
</style>
<div class="hero">
  <div class="hero-kicker">☀ Material intelligence platform</div>
  <h1>Engineer better<br>solar cells.</h1>
  <p>Explore solar-cell recipes with machine learning. Estimate power conversion efficiency, compare promising candidates, and turn historical research data into practical next steps.</p>
  <span class="pill">XGBoost regression</span><span class="pill">PCE prediction</span><span class="pill">Recipe exploration</span>
</div>
""", unsafe_allow_html=True)


def infer_columns(frame: pd.DataFrame, columns):
    numeric, categorical = [], []
    for col in columns:
        original = frame[col]
        converted = pd.to_numeric(original, errors="coerce")
        nonmissing = original.notna().sum()
        ratio = converted.notna().sum() / max(nonmissing, 1)
        if pd.api.types.is_numeric_dtype(original) or (nonmissing > 0 and ratio >= 0.80):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def clean_dataset(raw: pd.DataFrame):
    if TARGET not in raw.columns:
        raise ValueError(f"Dataset must contain the target column '{TARGET}'.")
    df = raw.copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if "JV_measured" in df.columns:
        measured = df["JV_measured"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
        # If the column is interpretable, use only measured rows.
        if measured.any():
            df = df[measured]
    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()
    feature_cols = [c for c in df.columns if c != TARGET and c not in LEAKAGE_COLUMNS]
    if not feature_cols:
        raise ValueError("No usable input feature columns were found.")
    numeric_cols, categorical_cols = infer_columns(df, feature_cols)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical_cols:
        df[col] = df[col].where(df[col].notna(), np.nan)
        df[col] = df[col].map(lambda x: str(x).strip() if pd.notna(x) else np.nan)
        df.loc[df[col].isin(["", "nan", "None", "NaN"]), col] = np.nan
    return df, feature_cols, numeric_cols, categorical_cols


@st.cache_resource(show_spinner="Loading the Colab-trained XGBoost model…")
def load_model():
    model_path = Path(__file__).parent / "solar_cell_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            "solar_cell_model.joblib was not found. Upload it to the same GitHub folder as app.py."
        )
    return joblib.load(model_path)


st.sidebar.markdown("## ☀ SolarCell AI")
st.sidebar.caption("MATERIAL INTELLIGENCE")
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset source")
repo_path = Path(__file__).parent / DEFAULT_DATASET
uploaded = st.sidebar.file_uploader("Upload the solar-cell CSV", type=["csv"])
if uploaded is not None:
    csv_bytes = uploaded.getvalue()
elif repo_path.exists():
    csv_bytes = repo_path.read_bytes()
else:
    st.info(
        f"To run the public website, add **{DEFAULT_DATASET}** to the same GitHub repository as `app.py`. "
        "For a quick local test, upload the CSV using the sidebar."
    )
    st.stop()

try:
    raw = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    df, dataset_feature_cols, numeric_cols, categorical_cols = clean_dataset(raw)
    model = load_model()
    # The saved sklearn pipeline remembers the feature names used during Colab training.
    feature_cols = list(getattr(model, "feature_names_in_", dataset_feature_cols))
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError("The CSV is missing model input columns: " + ", ".join(missing[:8]))
    df = df.copy()
    for col in feature_cols:
        if col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    candidates = df[feature_cols].copy()
    candidates["Predicted_PCE"] = model.predict(candidates[feature_cols])
    candidates = candidates.sort_values("Predicted_PCE", ascending=False)
except Exception as exc:
    st.error(f"Could not load the saved model or dataset: {exc}")
    st.stop()

with st.expander("Model information", expanded=False):
    a, b = st.columns(2)
    a.metric("Usable records", f"{len(df):,}")
    b.metric("Input features", len(feature_cols))
    st.success("Using the XGBoost pipeline saved from Google Colab. The website does not retrain the model on each launch.")
    st.caption("Candidate rankings are predictions for records in the uploaded/repository dataset, not newly discovered materials or proof of experimental improvement.")

st.markdown("<div class='section-label'>01 / Define your experiment</div>", unsafe_allow_html=True)
st.header("Enter your solar-cell recipe")
st.caption("Enter the recipe details you know. Select a value from the dataset or choose “Enter a custom value…” for a material/process not listed. Unknown or blank details are imputed by the trained pipeline; predictions for unfamiliar recipes may be less reliable.")
with st.form("recipe_form"):
    values = {}
    columns = st.columns(2)
    for i, col in enumerate(feature_cols):
        with columns[i % 2]:
            series = df[col]
            if col in numeric_cols:
                numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                if numeric_series.empty:
                    values[col] = st.number_input(col, value=0.0, key=f"input_{col}")
                else:
                    median = float(numeric_series.median())
                    lo, hi = float(numeric_series.min()), float(numeric_series.max())
                    if not np.isfinite(median): median = 0.0
                    if not np.isfinite(lo): lo = median
                    if not np.isfinite(hi): hi = median
                    step = 0.1 if abs(hi - lo) < 100 else 1.0
                    values[col] = st.number_input(
                        col, min_value=lo, max_value=hi, value=min(max(median, lo), hi),
                        step=step, key=f"input_{col}"
                    )
            else:
                counts = series.dropna().astype(str).value_counts()
                options = counts.head(40).index.tolist()
                if not options:
                    options = ["Unknown"]
                custom_choice = "✏️ Enter a custom value…"
                options = options + [custom_choice]
                selected = st.selectbox(col, options=options, index=0, key=f"input_{col}_choice")
                if selected == custom_choice:
                    values[col] = st.text_input(
                        f"Custom value for {col}",
                        value="",
                        placeholder="Type the material or process value",
                        key=f"input_{col}_custom",
                    )
                    if not values[col].strip():
                        values[col] = np.nan
                else:
                    values[col] = selected
    submitted = st.form_submit_button("Predict PCE and find optimization candidates", type="primary", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([{c: values[c] for c in feature_cols}], columns=feature_cols)
    prediction = float(model.predict(input_df)[0])
    st.markdown("<div class='section-label'>02 / Model output</div>", unsafe_allow_html=True)
    st.header("Prediction result")
    st.metric("Predicted power conversion efficiency (PCE)", f"{prediction:.2f}%")
    st.caption("This is a model estimate, not a laboratory measurement. Reliability depends on how similar the entered recipe is to examples in the dataset.")

    st.markdown("<div class='section-label'>03 / Candidate exploration</div>", unsafe_allow_html=True)
    st.header("Optimization suggestions")
    st.write("The table below ranks recipes in your dataset that the saved model predicts will have higher PCE. These are observed dataset examples—not guaranteed improvements or newly discovered materials.")
    higher = candidates[candidates["Predicted_PCE"] > prediction].head(10).copy()
    if higher.empty:
        st.warning("No held-out recipe was predicted to outperform this input. Try checking the details or exploring the highest-ranked recipes below.")
        higher = candidates.head(10).copy()
    shown = higher.copy()
    shown.insert(0, "Candidate", [f"Option {i+1}" for i in range(len(shown))])
    st.dataframe(shown.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Explain which input features differ from each recommended observed recipe.
    st.subheader("What changes in the top recommendation?")
    best = higher.iloc[0]
    changes = []
    for col in feature_cols:
        before, after = values[col], best[col]
        if pd.isna(before) and pd.isna(after):
            continue
        try:
            same = bool(np.isclose(float(before), float(after), equal_nan=True))
        except (TypeError, ValueError):
            same = str(before) == str(after)
        if not same:
            changes.append({"Input feature": col, "Your value": before, "Recommended example": after})
    if changes:
        st.dataframe(pd.DataFrame(changes), use_container_width=True, hide_index=True)
    else:
        st.write("The top recommendation matches the entered feature values, or differences are below numeric comparison precision.")

    st.download_button(
        "Download optimization candidates as CSV",
        data=shown.to_csv(index=False).encode("utf-8"),
        file_name="solar_cell_optimization_candidates.csv",
        mime="text/csv",
    )

    st.markdown("<div class='section-label'>04 / Research notes</div>", unsafe_allow_html=True)
    st.header("Model guidance")
    st.write(
        "Treat these candidates as leads for further investigation. The model learns associations in historical data; it does not prove that changing one parameter causes PCE to increase. "
        "Validate promising recipes experimentally and check that all recommended values are feasible for your fabrication process."
    )

st.divider()
st.caption("SolarCell AI Optimizer · AI-based solar cell material optimization using XGBoost · Research prototype for educational demonstration")
