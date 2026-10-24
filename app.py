from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

TARGET = "JV_default_PCE"
LEAKAGE_COLUMNS = ["JV_default_Voc", "JV_default_Jsc", "JV_default_FF", "JV_measured"]
DEFAULT_DATASET = "Solar Cell data set (1).csv"

st.set_page_config(page_title="Solar Cell AI Optimizer", page_icon="☀️", layout="wide")
st.title("☀️ AI-Based Solar Cell PCE Predictor & Optimizer")
st.write("Enter a solar-cell recipe to estimate its PCE, then explore higher-predicted-PCE recipes found in the dataset.")


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


@st.cache_resource(show_spinner="Training the XGBoost model on your dataset…")
def train_model(csv_bytes: bytes):
    raw = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    df, feature_cols, numeric_cols, categorical_cols = clean_dataset(raw)
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))
    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

    model = Pipeline([
        ("preprocess", preprocess),
        ("xgb", XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=4,
            random_state=42,
        ))
    ])
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, test_predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, test_predictions))),
        "R2": float(r2_score(y_test, test_predictions)),
    }
    test_candidates = X_test.copy()
    test_candidates["Predicted_PCE"] = test_predictions
    test_candidates = test_candidates.sort_values("Predicted_PCE", ascending=False)
    return model, df, feature_cols, numeric_cols, categorical_cols, metrics, test_candidates, X_train.shape[0], X_test.shape[0]


st.sidebar.header("Dataset")
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
    model, df, feature_cols, numeric_cols, categorical_cols, metrics, candidates, n_train, n_test = train_model(csv_bytes)
except Exception as exc:
    st.error(f"Could not load/train the model: {exc}")
    st.stop()

with st.expander("Model information", expanded=False):
    a, b, c, d = st.columns(4)
    a.metric("Usable records", f"{len(df):,}")
    b.metric("Input features", len(feature_cols))
    c.metric("Training rows", f"{n_train:,}")
    d.metric("Test rows", f"{n_test:,}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test MAE", f"{metrics['MAE']:.3f} PCE points")
    m2.metric("Test RMSE", f"{metrics['RMSE']:.3f} PCE points")
    m3.metric("Test R²", f"{metrics['R2']:.3f}")
    st.caption("Metrics use a held-out 20% test split. MAE/RMSE are in PCE percentage points; R² is unitless.")

st.header("1. Enter your solar-cell recipe")
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
    st.header("2. Prediction result")
    st.metric("Predicted power conversion efficiency (PCE)", f"{prediction:.2f}%")
    st.caption("This is a model estimate, not a laboratory measurement. Reliability depends on how similar the entered recipe is to examples in the dataset.")

    st.header("3. Optimization suggestions")
    st.write("The table below ranks recipes from the held-out part of your dataset that the model predicts will have higher PCE. These are observed dataset examples—not guaranteed improvements or newly discovered materials.")
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

    st.header("4. Model guidance")
    st.write(
        "Treat these candidates as leads for further investigation. The model learns associations in historical data; it does not prove that changing one parameter causes PCE to increase. "
        "Validate promising recipes experimentally and check that all recommended values are feasible for your fabrication process."
    )

st.divider()
st.caption("AI-Based Solar Cell Material Optimization using XGBoost | Research prototype for educational demonstration")
