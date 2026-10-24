import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------
st.set_page_config(
    page_title="SolarCell AI Optimizer",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# DESIGN
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F8F7F2;
        color: #102B2A;
    }

    [data-testid="stHeader"] {
        background-color: #F8F7F2;
    }

    h1, h2, h3 {
        color: #102B2A !important;
    }

    .hero {
        background: linear-gradient(120deg, #102B2A, #24554B);
        padding: 30px;
        border-radius: 18px;
        color: #F8F7F2;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #F8F7F2 !important;
        margin-bottom: 8px;
    }

    .hero p {
        color: #E8F0E9;
        font-size: 16px;
    }

    .result-card {
        background: #E8F0E9;
        border-left: 6px solid #D9A441;
        padding: 22px;
        border-radius: 12px;
        margin-top: 15px;
    }

    .small-note {
        color: #53645F;
        font-size: 14px;
    }

    div.stButton > button[kind="primary"] {
        background-color: #D9A441;
        color: #102B2A;
        border: none;
        font-weight: 700;
        border-radius: 9px;
        min-height: 45px;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #C38D2D;
        color: #102B2A;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #E0E6DF;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD THE GOOGLE COLAB MODEL
# Keep this file in the same GitHub folder as app.py.
# --------------------------------------------------
MODEL_PATH = Path(__file__).parent / "solar_cell_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def get_model_feature_names(loaded_model):
    """Get the feature names the saved model expects."""

    if hasattr(loaded_model, "feature_names_in_"):
        return list(loaded_model.feature_names_in_)

    # A saved sklearn Pipeline may expose feature names
    # through its preprocessing or model steps.
    if hasattr(loaded_model, "named_steps"):
        for step in loaded_model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


def get_categorical_columns(loaded_model):
    """
    Try to identify categorical columns from a
    scikit-learn Pipeline containing a ColumnTransformer.
    """
    categorical = set()

    if not hasattr(loaded_model, "named_steps"):
        return categorical

    for step in loaded_model.named_steps.values():
        transformers = getattr(step, "transformers_", None)

        if transformers is None:
            continue

        for name, transformer, columns in transformers:
            if name == "remainder" or transformer == "drop":
                continue

            transformer_name = str(type(transformer)).lower()

            if (
                "categorical" in name.lower()
                or "onehot" in transformer_name
                or "ordinal" in transformer_name
            ):
                if isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
                    categorical.update(str(col) for col in columns)

    return categorical


def is_probably_categorical(feature_name, categorical_columns):
    """Fallback when the saved model does not expose column types."""

    if feature_name in categorical_columns:
        return True

    name = feature_name.lower()

    categorical_words = [
        "material",
        "composition",
        "substrate",
        "architecture",
        "structure",
        "type",
        "method",
        "technique",
        "process",
        "layer",
        "solvent",
        "electrode",
        "additive",
        "category"
    ]

    return any(word in name for word in categorical_words)


def display_feature_name(name):
    """Make column names easier to read on the website."""

    return str(name).replace("_", " ").replace("-", " ").strip().title()


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>☀️ SolarCell AI Optimizer</h1>
        <p>
            Explore solar-cell material and process inputs with
            an XGBoost model trained in Google Colab.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Enter the properties of your proposed solar-cell design. "
    "The saved machine-learning model will estimate its output."
)

st.info(
    "The fields below are blank intentionally. The placeholders are "
    "guidance only, not recommended values. Use values and units "
    "consistent with the dataset used to train your model."
)

# --------------------------------------------------
# CHECK AND LOAD MODEL
# --------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        "The trained model file was not found. Upload "
        "'solar_cell_model.joblib' to the same GitHub repository "
        "folder as app.py."
    )
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error("The saved model could not be loaded.")
    st.code(str(error))
    st.caption(
        "Check the Python and package versions used in Google Colab "
        "against the versions in requirements.txt."
    )
    st.stop()


feature_names = get_model_feature_names(model)

if not feature_names:
    st.error(
        "The app could not automatically read the model's input column "
        "names. The model may not expose feature_names_in_. To make "
        "the form match your model exactly, the original training "
        "feature names are needed."
    )
    st.stop()

categorical_columns = get_categorical_columns(model)

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
st.subheader("1. Solar-cell design inputs")
st.caption(
    "Complete each field. For numeric fields, enter a number. "
    "For material or process fields, enter the category used in training."
)

with st.form("solar_cell_prediction_form"):

    user_values = {}

    # Two columns keep the form easy to read.
    left_col, right_col = st.columns(2)

    for index, feature in enumerate(feature_names):
        feature = str(feature)
        label = display_feature_name(feature)

        target_col = left_col if index % 2 == 0 else right_col

        with target_col:
            if is_probably_categorical(feature, categorical_columns):
                user_values[feature] = st.text_input(
                    label,
                    value="",
                    placeholder=f"Enter {label.lower()}",
                    help=(
                        "Use the same category names or spelling "
                        "as the data used to train the model."
                    ),
                    key=f"input_{index}"
                )
            else:
                user_values[feature] = st.text_input(
                    label,
                    value="",
                    placeholder=f"Enter {label.lower()}",
                    help=(
                        "Enter a numeric value in the same units "
                        "used during model training."
                    ),
                    key=f"input_{index}"
                )

    submitted = st.form_submit_button(
        "☀️ Predict Solar-Cell Efficiency",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if submitted:

    missing_fields = []
    invalid_fields = []
    prediction_values = {}

    for feature in feature_names:
        raw_value = str(user_values[feature]).strip()

        if not raw_value:
            missing_fields.append(display_feature_name(feature))
            continue

        if is_probably_categorical(feature, categorical_columns):
            prediction_values[feature] = raw_value
        else:
            try:
                number = float(raw_value)

                if not np.isfinite(number):
                    raise ValueError("Value must be finite.")

                prediction_values[feature] = number

            except (ValueError, TypeError):
                invalid_fields.append(display_feature_name(feature))

    if missing_fields:
        st.error(
            "Please complete these required fields: "
            + ", ".join(missing_fields)
        )

    elif invalid_fields:
        st.error(
            "Please enter numeric values for: "
            + ", ".join(invalid_fields)
        )

    else:
        input_df = pd.DataFrame(
            [prediction_values],
            columns=feature_names
        )

        try:
            prediction = model.predict(input_df)
            predicted_value = float(np.asarray(prediction).reshape(-1)[0])

            if not np.isfinite(predicted_value):
                raise ValueError("The model returned a non-finite prediction.")

            st.markdown("---")
            st.subheader("2. Prediction result")

            st.markdown(
                f"""
                <div class="result-card">
                    <p style="margin:0; color:#53645F;">
                        Model-predicted output
                    </p>
                    <h2 style="color:#102B2A; margin-top:8px;">
                        {predicted_value:.4f}
                    </h2>
                    <p style="margin-bottom:0; color:#53645F;">
                        This is the raw prediction from your saved model.
                        Confirm the target variable and units from your
                        Google Colab training notebook before interpreting it.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "Prediction completed using the model saved from Google Colab."
            )

            st.download_button(
                "Download prediction as CSV",
                data=input_df.assign(
                    model_prediction=predicted_value
                ).to_csv(index=False).encode("utf-8"),
                file_name="solar_cell_prediction.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(
                "The model could not make a prediction with these inputs. "
                "This can happen if a value has the wrong category, a "
                "required feature is missing, or the saved model expects "
                "a different input format."
            )
            st.code(str(error))

# --------------------------------------------------
# OPTIONAL DATASET
# The CSV is not required for predictions.
# --------------------------------------------------
st.markdown("---")
st.subheader("Optional: Explore historical data")
st.write(
    "Upload your dataset only if you want to inspect historical records. "
    "It is not needed to run the prediction form."
)

uploaded_csv = st.file_uploader(
    "Optional CSV dataset",
    type=["csv"],
    help="Upload the dataset used in your project, if you want to view it."
)

if uploaded_csv is not None:
    try:
        historical_data = pd.read_csv(uploaded_csv)

        st.write(f"Rows: {historical_data.shape[0]:,}")
        st.write(f"Columns: {historical_data.shape[1]:,}")

        st.dataframe(
            historical_data.head(10),
            use_container_width=True
        )

    except Exception as error:
        st.error("The uploaded CSV could not be read.")
        st.code(str(error))

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <p class="small-note">
        <b>About this project</b><br>
        This application uses a pretrained XGBoost-based model exported
        from Google Colab with joblib. Predictions are estimates based
        on patterns learned from the training data; they are not
        guarantees of real-world experimental performance.
    </p>
    """,
    unsafe_allow_html=True
)
