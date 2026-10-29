"""
Solar Cell AI Optimizer
- Trains/loads an XGBoost regression pipeline using the uploaded project's dataset.
- Predicts JV_default_PCE from device/material/process features.
- Performs constrained candidate search for a better configuration for the user's inputs.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

st.set_page_config(page_title="Solar Cell AI Optimizer", page_icon="☀️", layout="wide")

TARGET = "JV_default_PCE"
RANDOM_STATE = 42
MODEL_PATH = Path("solar_cell_model.joblib")
DATA_CANDIDATES = [
    Path("Solar Cell data set (1).csv"),
    Path("Solar Cell data set (1)(2).csv"),
    Path("solar_cell_dataset.csv"),
    Path("data.csv"),
]

@st.cache_data(show_spinner=False)
def load_and_prepare_data(path_string: str, mtime: float):
    """Mirror the cleaning and feature selection used in the supplied Colab notebook."""
    df = pd.read_csv(path_string, low_memory=False)
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain the target column '{TARGET}'.")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()

    if "JV_measured" in df.columns:
        measured = df["JV_measured"].astype(str).str.strip().str.lower()
        df = df[measured.isin(["true", "1", "yes"])].copy()

    df = df[df[TARGET] > 0].copy()
    leakage_cols = [TARGET, "JV_default_Voc", "JV_default_Jsc", "JV_default_FF", "JV_measured"]
    X = df.drop(columns=[c for c in leakage_cols if c in df.columns]).copy()
    y = df[TARGET].astype(float).copy()

    # Convert object columns to numeric if at least 85% of non-missing values parse.
    for col in X.select_dtypes(include=["object"]).columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        nonmissing = X[col].notna().sum()
        if nonmissing > 0 and converted.notna().sum() / nonmissing >= 0.85:
            X[col] = converted

    X = X.drop(columns=[c for c in X.columns if X[c].isna().all()])
    id_like = [c for c in X.columns if c.lower() in {"id", "record_id", "sample_id", "unnamed: 0"}]
    X = X.drop(columns=id_like)

    if len(X) < 20:
        raise ValueError("Fewer than 20 usable records remain after data cleaning.")

    # Booleans are numeric, matching the supplied notebook.
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    return df, X, y, numeric_features, categorical_features

@st.cache_resource(show_spinner="Preparing the XGBoost model…")
def load_or_train_model(path_string: str, data_mtime: float, model_mtime: float):
    """Load a compatible fitted pipeline, otherwise train the notebook's XGBoost setup."""
    df, X, y, numeric_features, categorical_features = load_and_prepare_data(path_string, data_mtime)

    # Existing model is used only if it can accept the exact feature columns.
    if MODEL_PATH.exists():
        try:
            loaded = joblib.load(MODEL_PATH)
            test_frame = X.iloc[:2].copy()
            _ = loaded.predict(test_frame)
            return loaded, X, y, numeric_features, categorical_features, "Loaded solar_cell_model.joblib"
        except Exception:
            pass

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=15))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ], remainder="drop")

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    regressor = XGBRegressor(
        n_estimators=350,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=4,
        random_state=RANDOM_STATE
    )
    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])
    model.fit(X_train, y_train)
    try:
        joblib.dump(model, MODEL_PATH)
    except Exception:
        pass
    return model, X, y, numeric_features, categorical_features, "Trained a new XGBoost model from the dataset"

def resolve_dataset():
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    csvs = sorted(Path(".").glob("*.csv"))
    for path in csvs:
        try:
            if TARGET in pd.read_csv(path, nrows=2).columns:
                return path
        except Exception:
            continue
    return None

def category_options(series):
    vals = series.dropna().astype(str).value_counts().index.tolist()
    # Keep the widget manageable for columns with thousands of distinct recipes.
    return vals[:300] if vals else ["Unknown"]

def make_user_input(X, numeric_features):
    """Render input controls and return a one-row DataFrame in training column order."""
    st.subheader("1. Enter your current solar-cell configuration")
    st.caption(
        "Numeric defaults use the dataset median; categorical defaults use the most frequent observed value. "
        "For categorical columns with many values, the dropdown shows the 300 most common observed values."
    )
    values = {}
    columns = st.columns(2)
    for i, feature in enumerate(X.columns):
        container = columns[i % 2]
        series = X[feature]
        with container:
            if feature in numeric_features:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if numeric.empty:
                    values[feature] = 0.0
                    st.number_input(feature, value=0.0, key=f"num_{feature}", disabled=True)
                else:
                    low = float(numeric.min())
                    high = float(numeric.max())
                    default = float(numeric.median())
                    if low == high:
                        step = 1.0
                    else:
                        step = float((high - low) / 1000)
                        if not np.isfinite(step) or step <= 0:
                            step = 0.1
                    value = st.number_input(
                        feature, min_value=low, max_value=high, value=min(max(default, low), high),
                        step=step, key=f"num_{feature}"
                    )
                    values[feature] = value
            else:
                options = category_options(series)
                counts = series.dropna().astype(str).value_counts()
                default = str(counts.index[0]) if len(counts) else options[0]
                if default not in options:
                    options.insert(0, default)
                selected = st.selectbox(feature, options=options, index=options.index(default), key=f"cat_{feature}")
                values[feature] = selected

    # Cast categorical values to their original type, while preserving numeric/bool columns.
    row = {}
    for feature in X.columns:
        if feature in numeric_features:
            if pd.api.types.is_bool_dtype(X[feature].dtype):
                row[feature] = bool(values[feature])
            else:
                row[feature] = float(values[feature])
        else:
            row[feature] = str(values[feature])
    return pd.DataFrame([row], columns=X.columns)

st.title("☀️ Solar Cell AI Optimizer")
st.write(
    "Predict power conversion efficiency (PCE) and search for a potentially better configuration. "
    "The XGBoost model predicts PCE; the optimization step searches alternatives."
)

dataset_path = resolve_dataset()
if dataset_path is None:
    st.error(
        "Dataset CSV not found. Put `Solar Cell data set (1).csv` in the same repository folder as `app.py` "
        "and `requirements.txt`."
    )
    st.stop()

try:
    data_mtime = dataset_path.stat().st_mtime
    model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0
    df_clean, X_data, y_data, numeric_features, categorical_features = load_and_prepare_data(
        str(dataset_path), data_mtime
    )
    model, X_data, y_data, numeric_features, categorical_features, model_status = load_or_train_model(
        str(dataset_path), data_mtime, model_mtime
    )
except Exception as exc:
    st.error(f"Could not load the dataset or model: {exc}")
    st.stop()

with st.sidebar:
    st.header("Project information")
    st.write(f"**Dataset:** `{dataset_path.name}`")
    st.write(f"**Usable records:** {len(X_data):,}")
    st.write(f"**Input features:** {len(X_data.columns)}")
    st.write("**Target:** `JV_default_PCE`")
    st.caption(model_status)

tab_predict, tab_optimize, tab_about = st.tabs(["PCE Prediction", "Configuration Optimization", "How it works"])

with tab_predict:
    user_row = make_user_input(X_data, numeric_features)
    if st.button("Estimate Cell Efficiency", type="primary", key="predict_btn"):
        try:
            predicted = float(np.asarray(model.predict(user_row)).reshape(-1)[0])
            st.session_state["current_configuration"] = user_row.copy()
            st.session_state["current_pce"] = predicted
            st.success("Prediction complete.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if "current_pce" in st.session_state:
        st.metric("Predicted Power Conversion Efficiency (PCE)", f"{st.session_state['current_pce']:.3f}%")
        st.caption("This is a model estimate, not an experimentally measured PCE.")

with tab_optimize:
    st.subheader("2. Search for a better configuration")
    st.write(
        "The optimizer keeps unselected inputs fixed. It samples complete observed recipes from the dataset, "
        "then substitutes only the parameters you allow it to change."
    )
    if "current_configuration" in st.session_state:
        baseline = st.session_state["current_configuration"].copy()
        baseline_pce = float(np.asarray(model.predict(baseline)).reshape(-1)[0])
        st.success("Using the configuration last submitted in the PCE Prediction tab.")
    else:
        st.info("Submit a configuration in the PCE Prediction tab first. Until then, the dataset median/mode baseline is used.")
        default_row = {}
        for feature in X_data.columns:
            if feature in numeric_features:
                vals = pd.to_numeric(X_data[feature], errors="coerce").dropna()
                default_row[feature] = float(vals.median()) if len(vals) else 0.0
            elif pd.api.types.is_bool_dtype(X_data[feature].dtype):
                default_row[feature] = bool(X_data[feature].mode().iloc[0]) if not X_data[feature].mode().empty else False
            else:
                mode = X_data[feature].dropna().astype(str).mode()
                default_row[feature] = str(mode.iloc[0]) if not mode.empty else "Unknown"
        baseline = pd.DataFrame([default_row], columns=X_data.columns)
        baseline_pce = float(np.asarray(model.predict(baseline)).reshape(-1)[0])

    st.metric("Starting configuration — predicted PCE", f"{baseline_pce:.3f}%")
    adjustable_options = [f for f in X_data.columns if X_data[f].nunique(dropna=True) > 1]
    default_adjustable = [f for f in numeric_features if f in adjustable_options]
    adjustable = st.multiselect(
        "Parameters the optimizer may change",
        options=adjustable_options,
        default=default_adjustable,
        help="Unselected parameters remain exactly as entered. Select material or process columns if you want those changed too."
    )
    n_candidates = st.slider("Number of candidate configurations", 100, 2000, 500, 100)

    st.warning(
        "The optimizer uses random search over observed dataset recipes. It is not a genetic algorithm or Bayesian optimization. "
        "Predicted improvements are not experimental proof, and combinations with fixed inputs may be unfamiliar to the model."
    )

    if st.button("Find a better configuration", type="primary", key="optimize_btn"):
        if not adjustable:
            st.error("Select at least one parameter to optimize.")
        else:
            try:
                rng = np.random.default_rng(RANDOM_STATE)
                # Candidate zero is the baseline, ensuring the selected result cannot
                # score lower than the baseline according to the same model.
                candidates = pd.concat([baseline.copy()] * (n_candidates + 1), ignore_index=True)
                source_indices = rng.integers(0, len(X_data), size=n_candidates)
                sampled_recipes = X_data.iloc[source_indices].reset_index(drop=True)
                for feature in adjustable:
                    candidates.loc[1:, feature] = sampled_recipes[feature].to_numpy()

                # Restore exact training dtypes to keep the fitted pipeline happy.
                for feature in X_data.columns:
                    if feature in numeric_features:
                        if pd.api.types.is_bool_dtype(X_data[feature].dtype):
                            candidates[feature] = candidates[feature].astype(bool)
                        else:
                            candidates[feature] = pd.to_numeric(candidates[feature], errors="coerce")
                    else:
                        candidates[feature] = candidates[feature].astype("object")

                predictions = []
                for start in range(0, len(candidates), 250):
                    predictions.extend(np.asarray(model.predict(candidates.iloc[start:start + 250]), dtype=float).reshape(-1))
                predictions = np.asarray(predictions)
                best_idx = int(np.nanargmax(predictions))
                best_config = candidates.iloc[[best_idx]].copy()
                best_pce = float(predictions[best_idx])
                improvement = best_pce - baseline_pce

                st.subheader("Recommended configuration")
                c1, c2, c3 = st.columns(3)
                c1.metric("Starting predicted PCE", f"{baseline_pce:.3f}%")
                c2.metric("Best predicted PCE", f"{best_pce:.3f}%", delta=f"{improvement:+.3f} PCE points")
                relative = improvement / abs(baseline_pce) * 100 if baseline_pce else np.nan
                c3.metric("Relative predicted change", f"{relative:+.2f}%" if np.isfinite(relative) else "N/A")

                if improvement > 1e-9:
                    st.success("A candidate with higher predicted PCE was found.")
                else:
                    st.info("No tested candidate improved on the baseline. The baseline is the best result found in this search.")

                comparison = pd.DataFrame({
                    "Parameter": X_data.columns,
                    "Starting value": baseline.iloc[0].reindex(X_data.columns).to_numpy(),
                    "Suggested value": best_config.iloc[0].reindex(X_data.columns).to_numpy(),
                })
                changed_mask = []
                for old, new in zip(comparison["Starting value"], comparison["Suggested value"]):
                    old_missing, new_missing = pd.isna(old), pd.isna(new)
                    changed_mask.append(not (old_missing and new_missing) and (old_missing != new_missing or old != new))
                comparison["Changed"] = changed_mask
                changed = comparison[comparison["Changed"]].drop(columns="Changed")
                if changed.empty:
                    st.write("The selected configuration matches the starting configuration.")
                else:
                    st.markdown("**Changed parameters**")
                    st.dataframe(changed, use_container_width=True, hide_index=True)

                with st.expander("View all recommended parameter values"):
                    st.dataframe(best_config.T.rename(columns={best_config.index[0]: "Suggested value"}), use_container_width=True)

                export = best_config.copy()
                export["Starting_Predicted_PCE"] = baseline_pce
                export["Suggested_Predicted_PCE"] = best_pce
                export["Predicted_Improvement_PCE_Points"] = improvement
                st.download_button(
                    "Download suggested configuration as CSV",
                    data=export.to_csv(index=False).encode("utf-8"),
                    file_name="solar_cell_suggested_configuration.csv",
                    mime="text/csv"
                )
                st.caption(
                    f"Compared {n_candidates:,} sampled candidates plus the baseline. "
                    "The result is the highest predicted PCE found in this finite search, not a proven global optimum."
                )
            except Exception as exc:
                st.error(f"Optimization failed: {exc}")

with tab_about:
    st.subheader("What the application does")
    st.markdown(
        """
        1. **Data preparation:** follows the supplied notebook's target cleaning and leakage-column exclusions.
        2. **Prediction:** an XGBoost regression pipeline estimates `JV_default_PCE`.
        3. **Optimization:** random search samples observed recipes and changes only the parameters selected by the user.
        4. **Selection:** the candidate with the highest predicted PCE is shown as the recommendation.

        **Limitations:** Random search is not guaranteed to find the global optimum. The dataset is observational; a high predicted PCE is not proof that a configuration is physically feasible or will perform better in a laboratory. Validate suggested configurations with domain constraints and experiments.
        """
    )
