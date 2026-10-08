import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIRS = (PROJECT_ROOT / "results", PROJECT_ROOT / "result")

MODEL_LABELS_BY_FILENAME = {
    "logistic_regression.joblib": "Logistic Regression",
    "svm.joblib": "SVM",
    "decision_tree.joblib": "Decision Tree",
    "knn.joblib": "KNN",
    "xgboost.joblib": "XGBoost",
    "ann_model.keras": "ANN",
}


st.set_page_config(
    page_title="Intelligent Heart Disease Diagnosis",
    page_icon="♥️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #16353b;
        --muted: #61777a;
        --teal: #167d78;
        --teal-soft: #e7f4f1;
        --line: #dce8e5;
        --paper: #f6faf8;
        --white: #ffffff;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3, [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        letter-spacing: 0;
    }

    h1 { font-size: 2.15rem; font-weight: 800; }
    h2 { font-size: 1.35rem; font-weight: 700; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }

    [data-testid="stSidebar"] {
        background: #eef6f3;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }

    .brand-mark {
        display: inline-flex;
        width: 42px;
        height: 42px;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        background: var(--teal);
        color: white;
        font-size: 22px;
        font-weight: 700;
    }

    .eyebrow {
        margin: 0 0 9px;
        color: var(--teal);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .hero {
        padding: 30px 32px;
        border: 1px solid #d5e8e2;
        border-radius: 12px;
        background: linear-gradient(115deg, #e7f4f1 0%, #f1f8f5 60%, #ffffff 100%);
    }

    .hero h1 { margin: 0 0 7px; }
    .hero-subtitle {
        margin: 0 0 12px;
        color: var(--teal);
        font-size: 1.08rem;
        font-weight: 600;
    }

    .hero-copy {
        max-width: 760px;
        margin: 0;
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.65;
    }

    .stat-card {
        min-height: 124px;
        padding: 19px 20px;
        border: 1px solid var(--line);
        border-radius: 10px;
        background: var(--white);
    }

    .stat-icon {
        display: block;
        margin-bottom: 11px;
        color: var(--teal);
        font-size: 1.2rem;
    }

    .stat-title {
        margin: 0 0 4px;
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 1rem;
        font-weight: 700;
    }

    .stat-detail {
        margin: 0;
        color: var(--muted);
        font-size: 0.84rem;
    }

    .section-heading {
        margin: 25px 0 13px;
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
    }

    .notice {
        padding: 14px 17px;
        border-left: 4px solid #d79b46;
        border-radius: 4px;
        background: #fff8eb;
        color: #68563b;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .stButton > button[kind="primaryFormSubmit"],
    .stButton > button[kind="primary"] {
        min-height: 44px;
        border: 0;
        border-radius: 7px;
        background: var(--teal);
        color: white;
        font-weight: 700;
    }

    .stButton > button[kind="primaryFormSubmit"]:hover,
    .stButton > button[kind="primary"]:hover {
        background: #116a66;
        color: white;
    }

    div[data-testid="stForm"] {
        padding: 22px;
        border: 1px solid var(--line);
        border-radius: 10px;
        background: var(--white);
    }

    @media (max-width: 640px) {
        h1 { font-size: 1.75rem; }
        .hero { padding: 23px 20px; }
        div[data-testid="stForm"] { padding: 14px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_json_artifact(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required model artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as artifact_file:
        return json.load(artifact_file)


@st.cache_resource
def load_artifacts():
    feature_names = _read_json_artifact(MODELS_DIR / "feature_names.json")
    imputation_config = _read_json_artifact(MODELS_DIR / "imputation_config.json")
    deployment_config = _read_json_artifact(MODELS_DIR / "deployment_config.json")

    if feature_names != deployment_config.get("FEATURE_NAMES"):
        raise ValueError(
            "feature_names.json and deployment_config.json contain different feature orders."
        )

    scaler_path = MODELS_DIR / "scaler.joblib"
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Required model artifact is missing: {scaler_path}")
    scaler = joblib.load(scaler_path)

    from tensorflow.keras.models import load_model

    models = {}
    for artifact_name in deployment_config.get("MODEL_NAMES", []):
        if artifact_name not in MODEL_LABELS_BY_FILENAME:
            raise ValueError(f"Unsupported model artifact in deployment config: {artifact_name}")

        artifact_path = MODELS_DIR / artifact_name
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Required model artifact is missing: {artifact_path}")

        model_label = MODEL_LABELS_BY_FILENAME[artifact_name]
        if artifact_name.endswith(".keras"):
            models[model_label] = load_model(artifact_path)
        else:
            models[model_label] = joblib.load(artifact_path)

    expected_models = set(MODEL_LABELS_BY_FILENAME.values())
    if set(models) != expected_models:
        missing_models = sorted(expected_models - set(models))
        raise ValueError(
            "deployment_config.json must list all six model artifacts. "
            f"Missing model entries: {', '.join(missing_models)}"
        )

    return {
        "scaler": scaler,
        "feature_names": feature_names,
        "imputation_config": imputation_config,
        "deployment_config": deployment_config,
        "models": models,
    }


def predict_patient(input_data):
    artifacts = load_artifacts()
    feature_names = artifacts["feature_names"]
    patient = pd.DataFrame([input_data], columns=feature_names)

    imputation_config = artifacts["imputation_config"]
    for feature_name in ("ca", "thal"):
        if feature_name in patient.columns and feature_name in imputation_config:
            patient[feature_name] = patient[feature_name].fillna(
                imputation_config[f"{feature_name}_median"]
            )

    if patient.isna().any().any():
        missing_features = patient.columns[patient.isna().any()].tolist()
        raise ValueError(
            "Input is missing values without saved imputation values for: "
            + ", ".join(missing_features)
        )
 
    scaled_patient = artifacts["scaler"].transform(patient[feature_names])
    model_probabilities = {}
    for model_name, model in artifacts["models"].items():
        if model_name == "ANN":
            probability = float(model.predict(scaled_patient, verbose=0)[0][0])
        else:
            probability = float(model.predict_proba(scaled_patient)[0][1])
        model_probabilities[model_name] = probability

    deployment_config = artifacts["deployment_config"]
    hybrid_models = deployment_config["HYBRID_MODELS"]
    hybrid_weights = deployment_config["HYBRID_WEIGHTS"]
    hybrid_probability = sum(
        hybrid_weights[model_name] * model_probabilities[model_name]
        for model_name in hybrid_models
    )
    hybrid_prediction = int(
        hybrid_probability >= deployment_config["HYBRID_THRESHOLD"]
    )

    return {
        "individual_probabilities": model_probabilities,
        "hybrid_probability": float(hybrid_probability),
        "hybrid_prediction": hybrid_prediction,
        "hybrid_models": hybrid_models,
        "hybrid_weights": hybrid_weights,
        "hybrid_threshold": deployment_config["HYBRID_THRESHOLD"],
    }


def _result_file(filename):
    for results_dir in RESULTS_DIRS:
        candidate = results_dir / filename
        if candidate.is_file():
            return candidate
    return None


def _display_prediction(result):
    st.subheader("Hybrid Risk Assessment")
    prediction_label = "Higher Risk" if result["hybrid_prediction"] else "Lower Risk"
    metrics = st.columns(2)
    metrics[0].metric("Predicted Risk", prediction_label)
    metrics[1].metric("Hybrid Probability", f"{result['hybrid_probability']:.1%}")

    st.caption(
        "Academic model estimate based on the configured hybrid threshold; "
        "this is not a medical diagnosis."
    )
    st.markdown("**Individual model probabilities**")
    probabilities = pd.DataFrame(
        [
            {"Model": name, "Probability": f"{probability:.1%}"}
            for name, probability in result["individual_probabilities"].items()
        ]
    )
    st.dataframe(probabilities, hide_index=True, use_container_width=True)

    st.markdown("**Hybrid model weights**")
    weights = pd.DataFrame(
        [
            {"Hybrid model": name, "Validation-F1 weight": weight}
            for name, weight in result["hybrid_weights"].items()
            if name in result["hybrid_models"]
        ]
    )
    st.dataframe(weights, hide_index=True, use_container_width=True)


def show_home():
    st.markdown(
        """
        <section class="hero">
            <p class="eyebrow">Clinical decision-support research</p>
            <h1>Intelligent Heart Disease Diagnosis</h1>
            <p class="hero-subtitle">Hybrid AI-Based Heart Disease Risk Assessment</p>
            <p class="hero-copy">
                Explore an academic application that brings together multiple machine-learning
                models and a validation-weighted hybrid voting approach to estimate heart-disease risk.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-heading">At a glance</p>', unsafe_allow_html=True)
    cards = st.columns(4)
    card_content = [
        ("✦", "6 AI Models", "Classical ML and neural networks"),
        ("⌘", "13 Patient Features", "Structured clinical inputs"),
        ("⟲", "Hybrid AI", "Top-three validation-F1 weighted voting"),
        ("▤", "UCI Heart Disease", "Research dataset"),
    ]
    for column, (icon, title, detail) in zip(cards, card_content):
        with column:
            st.markdown(
                f"""
                <div class="stat-card">
                    <span class="stat-icon">{icon}</span>
                    <p class="stat-title">{title}</p>
                    <p class="stat-detail">{detail}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<p class="section-heading">Get started</p>', unsafe_allow_html=True)
    st.write("Enter the 13 model features to open the risk assessment workflow.")
    if st.button("Start Risk Assessment", type="primary", icon="➡️"):
        st.switch_page(assessment_page)

    st.markdown(
        '<div class="notice"><strong>Medical disclaimer:</strong> This application is an '
        'educational/academic machine-learning project and is not a professional medical diagnosis.</div>',
        unsafe_allow_html=True,
    )


def show_assessment():
    st.markdown('<p class="eyebrow">Patient feature entry</p>', unsafe_allow_html=True)
    st.title("Risk Assessment")
    st.write("Provide the encoded feature values used by the project model.")

    with st.form("risk_assessment_form"):
        st.subheader("Patient profile")
        profile = st.columns(3)
        with profile[0]:
            age = st.number_input("age", min_value=1, max_value=120, value=50, step=1)
        with profile[1]:
            sex = st.selectbox("sex", options=[0, 1], help="Encoded category: 0 or 1.")
        with profile[2]:
            cp = st.selectbox("cp", options=[0, 1, 2, 3], help="Chest-pain category code.")

        st.subheader("Clinical measurements")
        measurements = st.columns(3)
        with measurements[0]:
            trestbps = st.number_input("trestbps", min_value=0, max_value=300, value=120, step=1)
        with measurements[1]:
            chol = st.number_input("chol", min_value=0, max_value=700, value=200, step=1)
        with measurements[2]:
            fbs = st.selectbox("fbs", options=[0, 1], help="Encoded category: 0 or 1.")

        clinical = st.columns(3)
        with clinical[0]:
            restecg = st.selectbox("restecg", options=[0, 1, 2], help="Resting ECG category code.")
        with clinical[1]:
            thalach = st.number_input("thalach", min_value=0, max_value=300, value=150, step=1)
        with clinical[2]:
            exang = st.selectbox("exang", options=[0, 1], help="Encoded category: 0 or 1.")

        remaining = st.columns(3)
        with remaining[0]:
            oldpeak = st.number_input("oldpeak", min_value=0.0, max_value=20.0, value=1.0, step=0.1)
        with remaining[1]:
            slope = st.selectbox("slope", options=[0, 1, 2], help="Slope category code.")
        with remaining[2]:
            ca = st.selectbox("ca", options=[0, 1, 2, 3, 4], help="Number of major vessels, encoded 0-4.")

        thal = st.selectbox("thal", options=[0, 1, 2, 3], help="Thal category code.")
        submitted = st.form_submit_button(
            "Analyze Heart Disease Risk",
            type="primary",
            icon="🔎",
            use_container_width=True,
        )

    if submitted:
        st.session_state.pop("latest_prediction", None)
        patient_data = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
        try:
            prediction = predict_patient(patient_data)
            st.session_state["latest_prediction"] = prediction
            _display_prediction(prediction)
        except Exception as error:
            st.error(f"Unable to calculate the risk estimate: {error}")

    st.markdown(
        '<div class="notice"><strong>Medical disclaimer:</strong> This application is an '
        'educational/academic machine-learning project and is not a professional medical diagnosis.</div>',
        unsafe_allow_html=True,
    )


def show_model_insights():
    st.markdown('<p class="eyebrow">Model transparency</p>', unsafe_allow_html=True)
    st.title("Model Insights")
    latest_prediction = st.session_state.get("latest_prediction")
    if latest_prediction:
        _display_prediction(latest_prediction)
    else:
        st.info("Complete a risk assessment to view its model probabilities and hybrid result here.")
        try:
            deployment_config = load_artifacts()["deployment_config"]
            configured_hybrid = pd.DataFrame(
                [
                    {
                        "Hybrid model": model_name,
                        "Validation-F1 weight": deployment_config["HYBRID_WEIGHTS"][model_name],
                    }
                    for model_name in deployment_config["HYBRID_MODELS"]
                ]
            )
            st.markdown("**Configured hybrid models and weights**")
            st.dataframe(configured_hybrid, hide_index=True, use_container_width=True)
        except Exception as error:
            st.error(f"Unable to load model artifacts: {error}")

    st.markdown('<p class="section-heading">Model comparison</p>', unsafe_allow_html=True)
    comparison_path = _result_file("model_comparison.csv")
    if comparison_path:
        try:
            st.dataframe(pd.read_csv(comparison_path), hide_index=True, use_container_width=True)
        except Exception as error:
            st.error(f"Unable to load model comparison from {comparison_path}: {error}")
    else:
        st.warning(
            "Model comparison data is not available. Expected results/model_comparison.csv "
            "or result/model_comparison.csv."
        )

    st.markdown('<p class="section-heading">XGBoost Feature-Level Explainability</p>', unsafe_allow_html=True)
    st.caption(
        "Global feature importance from the saved SHAP analysis describes model behavior; "
        "it does not establish medical causation."
    )
    shap_path = _result_file("shap_feature_importance.csv")
    if shap_path:
        try:
            importance_data = pd.read_csv(shap_path)
            normalized_columns = {
                "".join(character for character in str(column).lower() if character.isalnum()): column
                for column in importance_data.columns
            }
            feature_column = next(
                (
                    column
                    for normalized, column in normalized_columns.items()
                    if "feature" in normalized
                ),
                None,
            )
            importance_column = next(
                (
                    normalized_columns[name]
                    for name in ("meanabsshapvalue", "meanabsshap", "shapimportance", "importance", "meanabsvalue")
                    if name in normalized_columns
                ),
                None,
            )
            if feature_column and importance_column:
                importance_data[importance_column] = pd.to_numeric(
                    importance_data[importance_column], errors="coerce"
                )
                importance_data = importance_data.dropna(subset=[importance_column])
                importance_data = importance_data.sort_values(importance_column, ascending=False)
                st.bar_chart(importance_data.set_index(feature_column)[importance_column])
                st.dataframe(importance_data, hide_index=True, use_container_width=True)
                st.caption(
                    "Most influential features in the saved global importance table: "
                    + ", ".join(importance_data[feature_column].head(5).astype(str))
                )
            else:
                st.warning(
                    f"The SHAP file at {shap_path} does not expose recognizable feature and importance columns."
                )
                st.dataframe(importance_data, hide_index=True, use_container_width=True)
        except Exception as error:
            st.error(f"Unable to load XGBoost feature importance from {shap_path}: {error}")
    else:
        st.warning(
            "XGBoost feature importance is not available. Expected results/shap_feature_importance.csv "
            "or result/shap_feature_importance.csv."
        )


def show_about():
    st.markdown('<p class="eyebrow">Project overview</p>', unsafe_allow_html=True)
    st.title("About")
    st.write(
        "This academic project explores heart-disease risk assessment using the UCI Heart Disease dataset. "
        "The interface accepts 13 input features and is designed to connect to trained model artifacts in a later step."
    )

    st.subheader("Models")
    st.write(
        "Logistic Regression · SVM · Decision Tree · KNN · XGBoost · ANN"
    )

    st.subheader("Hybrid approach")
    st.write(
        "The hybrid method selects the top three models by validation F1 score and assigns normalized "
        "weights based on their validation F1 scores. This page does not calculate or display model results."
    )

    st.subheader("Limitations")
    st.write(
        "Dataset-based models may not generalize to every population, clinical setting, or measurement process. "
        "This frontend is not a validated clinical tool; model outputs require careful evaluation and must not "
        "replace professional medical judgment."
    )
    st.markdown(
        '<div class="notice"><strong>Medical disclaimer:</strong> This application is an '
        'educational/academic machine-learning project and is not a professional medical diagnosis.</div>',
        unsafe_allow_html=True,
    )


home_page = st.Page(show_home, title="Home", icon="🏠", default=True)
assessment_page = st.Page(show_assessment, title="Risk Assessment", icon="🩺")
insights_page = st.Page(show_model_insights, title="Model Insights", icon="📊")
about_page = st.Page(show_about, title="About", icon="ℹ️")

st.sidebar.markdown(
    '<div style="display:flex; align-items:center; gap:11px; margin: 8px 0 20px;">'
    '<span class="brand-mark">♥</span><div><strong style="color:#16353b;">Heart AI</strong>'
    '<br><span style="font-size:0.78rem; color:#61777a;">Research dashboard</span></div></div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("HEART DISEASE · HYBRID AI")

navigation = st.navigation(
    [home_page, assessment_page, insights_page, about_page],
    position="sidebar",
)
navigation.run()