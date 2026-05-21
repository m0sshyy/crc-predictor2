from pathlib import Path
import pickle
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CRC Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crc_xgboost_model.pkl"
IMAGE_CANDIDATES = [
    BASE_DIR / "assets" / "digestive_system.png",
    BASE_DIR / "Untitled design.png",  # fallback if you keep the uploaded filename
]

# ─────────────────────────────────────────────
#  MODEL SETTINGS
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "Age",
    "Gender",
    "Family_History",
    "Smoking_History",
    "Alcohol_Consumption",
    "Diabetes",
    "Inflammatory_Bowel_Disease",
    "Genetic_Mutation",
    "Obesity_Risk_Level",
    "Diet_Risk_Level",
    "Physical_Inactivity_Risk",
    "Screening_History_Irregular",
    "Screening_History_Never",
    "Screening_History_Regular",
    "Genetic_Age_Interaction",
    "Medical_Comorbidity_Score",
    "Lifestyle_Index",
]

FEATURE_LABELS = {
    "Age": "Age",
    "Gender": "Gender",
    "Family_History": "Family history",
    "Smoking_History": "Smoking history",
    "Alcohol_Consumption": "Alcohol consumption",
    "Diabetes": "Diabetes",
    "Inflammatory_Bowel_Disease": "Inflammatory bowel disease",
    "Genetic_Mutation": "Genetic mutation",
    "Obesity_Risk_Level": "Obesity level",
    "Diet_Risk_Level": "Diet risk",
    "Physical_Inactivity_Risk": "Physical inactivity",
    "Screening_History_Irregular": "Irregular screening",
    "Screening_History_Never": "Never screened",
    "Screening_History_Regular": "Regular screening",
    "Genetic_Age_Interaction": "Genetic × age",
    "Medical_Comorbidity_Score": "Medical comorbidity score",
    "Lifestyle_Index": "Lifestyle index",
}

CLASS_NAMES = {0: "Low", 1: "Moderate", 2: "High"}
CLASS_EMOJI = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}
CLASS_ORDER = ["Low", "Moderate", "High"]

# ─────────────────────────────────────────────
#  STYLE
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    #MainMenu, footer {visibility: hidden;}

    .hero-card {
        padding: 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #eef8ff 0%, #fff5f5 55%, #ffffff 100%);
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.3rem;
    }

    .eyebrow {
        font-size: 0.78rem;
        font-weight: 800;
        color: #0f766e;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.7rem;
    }

    .hero-title {
        font-size: 2.05rem;
        line-height: 1.18;
        font-weight: 800;
        color: #172554;
        margin-bottom: 0.75rem;
    }

    .hero-text {
        font-size: 0.98rem;
        line-height: 1.7;
        color: #475569;
        max-width: 680px;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #172554;
        margin-top: 0.8rem;
        margin-bottom: 0.7rem;
    }

    .soft-card {
        padding: 1.1rem 1.15rem;
        border-radius: 18px;
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        margin-bottom: 0.8rem;
    }

    .metric-card {
        padding: 1.35rem;
        border-radius: 22px;
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        text-align: center;
    }

    .risk-low {
        border-top: 8px solid #16a34a;
    }

    .risk-moderate {
        border-top: 8px solid #f59e0b;
    }

    .risk-high {
        border-top: 8px solid #dc2626;
    }

    .risk-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #64748b;
        font-weight: 800;
    }

    .risk-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #0f172a;
        margin-top: 0.3rem;
    }

    .small-note {
        font-size: 0.82rem;
        color: #64748b;
        line-height: 1.55;
    }

    .warning-box {
        padding: 1rem 1.15rem;
        border-radius: 18px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #7c2d12;
        font-size: 0.88rem;
        line-height: 1.55;
    }

    .success-box {
        padding: 1rem 1.15rem;
        border-radius: 18px;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #14532d;
        font-size: 0.88rem;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  CACHED RESOURCES
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    with open(model_path, "rb") as file:
        return pickle.load(file)


@st.cache_resource(show_spinner=False)
def load_explainer(_model):
    return shap.TreeExplainer(_model)


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def find_image_path() -> Path | None:
    for path in IMAGE_CANDIDATES:
        if path.exists():
            return path
    return None


def yes_no_to_int(value: str) -> int:
    return 1 if value == "Yes" else 0


def build_input_dataframe(values: dict) -> pd.DataFrame:
    gender_val = 1 if values["gender"] == "Male" else 0
    family_val = yes_no_to_int(values["family_history"])
    smoking_val = yes_no_to_int(values["smoking"])
    alcohol_val = yes_no_to_int(values["alcohol"])
    diabetes_val = yes_no_to_int(values["diabetes"])
    ibd_val = yes_no_to_int(values["ibd"])
    genetic_val = yes_no_to_int(values["genetic_mutation"])

    obesity_map = {"Normal": 0, "Overweight": 1, "Obese": 2}
    diet_map = {"Low": 0, "Moderate": 1, "High": 2}
    activity_map = {"High": 0, "Moderate": 1, "Low": 2}

    obesity_score = obesity_map[values["obesity"]]
    diet_score = diet_map[values["diet_risk"]]
    inactivity_score = activity_map[values["activity"]]

    row = {
        "Age": values["age"],
        "Gender": gender_val,
        "Family_History": family_val,
        "Smoking_History": smoking_val,
        "Alcohol_Consumption": alcohol_val,
        "Diabetes": diabetes_val,
        "Inflammatory_Bowel_Disease": ibd_val,
        "Genetic_Mutation": genetic_val,
        "Obesity_Risk_Level": obesity_score,
        "Diet_Risk_Level": diet_score,
        "Physical_Inactivity_Risk": inactivity_score,
        "Screening_History_Irregular": 1 if values["screening"] == "Irregular" else 0,
        "Screening_History_Never": 1 if values["screening"] == "Never" else 0,
        "Screening_History_Regular": 1 if values["screening"] == "Regular" else 0,
        "Genetic_Age_Interaction": genetic_val * values["age"],
        "Medical_Comorbidity_Score": diabetes_val + ibd_val,
        "Lifestyle_Index": smoking_val + alcohol_val + obesity_score + diet_score + inactivity_score,
    }

    return pd.DataFrame([row])[FEATURE_COLS]


def predict_risk(model, input_df: pd.DataFrame) -> tuple[str, dict[str, float], float]:
    prediction = int(model.predict(input_df)[0])
    predicted_label = CLASS_NAMES.get(prediction, str(prediction))

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        model_classes = list(getattr(model, "classes_", [0, 1, 2]))
        probability_dict = {CLASS_NAMES.get(int(cls), str(cls)): float(prob) for cls, prob in zip(model_classes, probabilities)}
    else:
        probability_dict = {"Low": 0.0, "Moderate": 0.0, "High": 0.0}
        probability_dict[predicted_label] = 1.0

    confidence = probability_dict.get(predicted_label, max(probability_dict.values()))
    return predicted_label, probability_dict, confidence


def extract_shap_values(model, input_df: pd.DataFrame, predicted_label: str) -> pd.Series:
    explainer = load_explainer(model)
    shap_values = explainer.shap_values(input_df)
    predicted_class = {v: k for k, v in CLASS_NAMES.items()}.get(predicted_label, 0)

    if isinstance(shap_values, list):
        selected_values = np.array(shap_values[predicted_class]).flatten()
    else:
        shap_array = np.array(shap_values)
        if shap_array.ndim == 3:
            selected_values = shap_array[0, :, predicted_class]
        elif shap_array.ndim == 2:
            selected_values = shap_array[0]
        else:
            selected_values = shap_array.flatten()

    if len(selected_values) != len(FEATURE_COLS):
        raise ValueError(
            f"SHAP output has {len(selected_values)} values, but {len(FEATURE_COLS)} features were expected."
        )

    labels = [FEATURE_LABELS.get(col, col) for col in FEATURE_COLS]
    return pd.Series(selected_values, index=labels)


def plot_shap_bar(shap_series: pd.Series, predicted_label: str):
    shap_top = shap_series.reindex(shap_series.abs().sort_values(ascending=True).index).tail(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#dc2626" if value > 0 else "#16a34a" for value in shap_top.values]
    ax.barh(shap_top.index, shap_top.values, color=colors)
    ax.axvline(0, color="#94a3b8", linewidth=1, linestyle="--")
    ax.set_xlabel("SHAP value impact on predicted class")
    ax.set_title(f"Top feature contributions for {predicted_label} risk")

    red_patch = mpatches.Patch(color="#dc2626", label="Pushes prediction higher")
    green_patch = mpatches.Patch(color="#16a34a", label="Pushes prediction lower")
    ax.legend(handles=[red_patch, green_patch], loc="lower right")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    return fig


def interpretation_text(predicted_label: str) -> list[str]:
    interpretations = {
        "Low": [
            "The current profile is predicted as low risk by the model.",
            "The user may continue healthy lifestyle habits and age-appropriate CRC screening awareness.",
            "A medical consultation is still important if symptoms, family history, or new risk factors appear.",
        ],
        "Moderate": [
            "The current profile is predicted as moderate risk by the model.",
            "The user should consider discussing CRC screening eligibility with a healthcare professional.",
            "Lifestyle-related factors such as smoking, alcohol use, diet, physical inactivity, and weight should be reviewed.",
        ],
        "High": [
            "The current profile is predicted as high risk by the model.",
            "The user should be encouraged to consult a qualified healthcare professional for proper assessment.",
            "The result is not a diagnosis, but it may indicate that screening advice should not be delayed.",
        ],
    }
    return interpretations.get(predicted_label, ["No interpretation is available for this prediction."])


def create_summary(values: dict, predicted_label: str, probability_dict: dict[str, float], shap_series: pd.Series | None) -> str:
    profile_lines = "\n".join([f"- {key.replace('_', ' ').title()}: {value}" for key, value in values.items()])
    probability_lines = "\n".join([f"- {label}: {probability_dict.get(label, 0) * 100:.1f}%" for label in CLASS_ORDER])

    shap_lines = "SHAP explanation was not generated."
    if shap_series is not None:
        top_factors = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index).head(8)
        shap_lines = "\n".join([f"- {feature}: {value:+.4f}" for feature, value in top_factors.items()])

    return f"""CRC Risk Prediction Summary

Predicted risk level: {predicted_label}

Input profile:
{profile_lines}

Class probabilities:
{probability_lines}

Top SHAP contributions:
{shap_lines}

Disclaimer:
This output is generated for academic and educational purposes only. It is not a medical diagnosis and must not replace professional medical advice, screening, or treatment.
"""


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please place `crc_xgboost_model.pkl` in the same folder as `app.py` before running the app."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error(f"The model could not be loaded: {error}")
    st.stop()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("About this prototype")
    st.write(
        "This web app demonstrates an XGBoost-based colorectal cancer risk prediction prototype with SHAP explainability."
    )
    st.info(
        "Academic tool only. This app does not diagnose colorectal cancer and does not replace advice from a qualified healthcare professional."
    )
    st.markdown("**Model input groups**")
    st.markdown("- Demographic factors\n- Clinical and genetic factors\n- Lifestyle and screening factors")

# ─────────────────────────────────────────────
#  HERO SECTION WITH IMAGE
# ─────────────────────────────────────────────
hero_left, hero_right = st.columns([1.55, 0.85], vertical_alignment="center")

with hero_left:
    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">AI-Based CRC Risk Prediction</div>
            <div class="hero-title">Early Detection of Colorectal Cancer Risk Factors Using XGBoost</div>
            <div class="hero-text">
                This prototype estimates colorectal cancer risk level using structured risk-factor inputs.
                It also provides SHAP-based explanations so users can understand which factors influenced the prediction.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    image_path = find_image_path()
    if image_path is not None:
        st.image(str(image_path), use_container_width=True)
    else:
        st.markdown(
            """
            <div class="soft-card">
                <b>Image not found.</b><br>
                <span class="small-note">To display the illustration, add it as <code>assets/digestive_system.png</code>.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="warning-box">
    <b>Important:</b> This system is for academic demonstration and risk-awareness support only. It should not be used as a final diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  INPUT FORM
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">1. Enter User Risk-Factor Profile</div>', unsafe_allow_html=True)

with st.form("crc_prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographic")
        age = st.number_input("Age", min_value=20, max_value=100, value=50, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        screening = st.selectbox("Screening history", ["Regular", "Irregular", "Never"])

    with col2:
        st.subheader("Clinical / Genetic")
        family_history = st.selectbox("Family history of CRC", ["No", "Yes"])
        genetic_mutation = st.selectbox("Known genetic mutation", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        ibd = st.selectbox("Inflammatory bowel disease", ["No", "Yes"])

    with col3:
        st.subheader("Lifestyle")
        smoking = st.selectbox("Smoking history", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol consumption", ["No", "Yes"])
        obesity = st.selectbox("Obesity / BMI category", ["Normal", "Overweight", "Obese"])
        diet_risk = st.selectbox("Diet risk level", ["Low", "Moderate", "High"])
        activity = st.selectbox("Physical activity level", ["High", "Moderate", "Low"])

    submitted = st.form_submit_button("Analyse CRC Risk Profile", use_container_width=True)

values = {
    "age": age,
    "gender": gender,
    "screening": screening,
    "family_history": family_history,
    "genetic_mutation": genetic_mutation,
    "diabetes": diabetes,
    "ibd": ibd,
    "smoking": smoking,
    "alcohol": alcohol,
    "obesity": obesity,
    "diet_risk": diet_risk,
    "activity": activity,
}

# ─────────────────────────────────────────────
#  PREDICTION OUTPUT
# ─────────────────────────────────────────────
if submitted:
    input_df = build_input_dataframe(values)
    predicted_label, probability_dict, confidence = predict_risk(model, input_df)

    st.markdown('<div class="section-title">2. Prediction Result</div>', unsafe_allow_html=True)

    risk_class = predicted_label.lower()
    result_col, prob_col = st.columns([0.9, 1.1])

    with result_col:
        st.markdown(
            f"""
            <div class="metric-card risk-{risk_class}">
                <div class="risk-label">Predicted Risk Level</div>
                <div class="risk-value">{CLASS_EMOJI.get(predicted_label, '')} {predicted_label}</div>
                <div class="small-note">Model confidence: <b>{confidence * 100:.1f}%</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with prob_col:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.subheader("Class probability")
        for label in CLASS_ORDER:
            st.progress(probability_dict.get(label, 0.0), text=f"{label}: {probability_dict.get(label, 0.0) * 100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("View processed model input"):
        labelled_df = input_df.rename(columns=FEATURE_LABELS)
        st.dataframe(labelled_df, use_container_width=True)

    # SHAP section
    st.markdown('<div class="section-title">3. Explainable AI Output Using SHAP</div>', unsafe_allow_html=True)
    shap_series = None
    try:
        shap_series = extract_shap_values(model, input_df, predicted_label)
        fig = plot_shap_bar(shap_series, predicted_label)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        risk_increasing = shap_series[shap_series > 0].sort_values(ascending=False).head(5)
        risk_decreasing = shap_series[shap_series < 0].sort_values(ascending=True).head(5)

        shap_col1, shap_col2 = st.columns(2)
        with shap_col1:
            st.markdown("**Factors pushing the model toward the predicted class**")
            if len(risk_increasing) > 0:
                st.dataframe(risk_increasing.rename("SHAP value"), use_container_width=True)
            else:
                st.write("No positive SHAP contribution detected.")

        with shap_col2:
            st.markdown("**Factors reducing the model score for the predicted class**")
            if len(risk_decreasing) > 0:
                st.dataframe(risk_decreasing.rename("SHAP value"), use_container_width=True)
            else:
                st.write("No negative SHAP contribution detected.")

    except Exception as error:
        st.warning(f"SHAP explanation could not be generated for this prediction. Details: {error}")

    # Interpretation section
    st.markdown('<div class="section-title">4. Interpretation and Next-Step Awareness</div>', unsafe_allow_html=True)
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    for item in interpretation_text(predicted_label):
        st.markdown(f"- {item}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="warning-box">
        <b>Medical disclaimer:</b> The prediction and SHAP values are model-based outputs for academic research.
        They do not prove medical causality and must not replace professional medical diagnosis, screening, or treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_text = create_summary(values, predicted_label, probability_dict, shap_series)
    st.download_button(
        label="Download prediction summary",
        data=summary_text,
        file_name="crc_prediction_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )
else:
    st.markdown(
        """
        <div class="soft-card">
        Fill in the risk-factor profile above and click <b>Analyse CRC Risk Profile</b> to generate the risk prediction and SHAP explanation.
        </div>
        """,
        unsafe_allow_html=True,
    )
