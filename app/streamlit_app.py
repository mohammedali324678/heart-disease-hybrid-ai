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

MODEL_DESCRIPTIONS = {
    "Logistic Regression": (
        "A linear classification method that estimates class probability from a weighted "
        "combination of input features."
    ),
    "SVM": (
        "A Support Vector Machine that separates classes by learning a decision boundary "
        "in the feature space."
    ),
    "Decision Tree": (
        "A tree-based model that applies a sequence of feature-based splits to produce an output."
    ),
    "KNN": (
        "A K-Nearest Neighbors classifier that predicts from the labels of nearby training examples."
    ),
    "XGBoost": (
        "A gradient-boosted tree model that combines sequentially trained decision trees."
    ),
    "ANN": (
        "An Artificial Neural Network that learns patterns through connected computational layers."
    ),
}

SEX_LABELS = {0: "Female", 1: "Male"}
CHEST_PAIN_LABELS = {
    0: "Typical angina",
    1: "Atypical angina",
    2: "Non-anginal pain",
    3: "Asymptomatic",
}
FASTING_BLOOD_SUGAR_LABELS = {0: "<= 120 mg/dL", 1: "> 120 mg/dL"}
RESTING_ECG_LABELS = {
    0: "Normal",
    1: "ST-T wave abnormality",
    2: "Left ventricular hypertrophy",
}
EXERCISE_ANGINA_LABELS = {0: "No", 1: "Yes"}
SLOPE_LABELS = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
MAJOR_VESSEL_LABELS = {
    0: "0 major vessels",
    1: "1 major vessel",
    2: "2 major vessels",
    3: "3 major vessels",
}
THAL_LABELS = {
    0: "Normal",
    1: "Fixed defect",
    2: "Reversible defect",
    3: "Other/unknown",
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
        font-family: 'Segoe UI', sans-serif;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Segoe UI', sans-serif;
        letter-spacing: 0;
    }

    h1 { font-size: clamp(1.75rem, 3vw, 2.35rem); font-weight: 750; line-height: 1.18; }
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
        margin-bottom: 1.25rem;
        padding: 34px 36px;
        border: 1px solid #d5e8e2;
        border-radius: 14px;
        background: linear-gradient(115deg, #e7f4f1 0%, #f1f8f5 60%, #ffffff 100%);
    }

    .hero h1 { max-width: 900px; margin: 0 0 10px; }
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
        min-height: 132px;
        padding: 20px;
        border: 1px solid var(--line);
        border-radius: 12px;
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
        font-family: 'Segoe UI', sans-serif;
        font-size: 1rem;
        font-weight: 700;
    }

    .stat-detail {
        margin: 0;
        color: var(--muted);
        font-size: 0.84rem;
    }

    .section-heading {
        margin: 30px 0 14px;
        color: var(--ink);
        font-family: 'Segoe UI', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
    }

    .risk-state {
        display: inline-block;
        margin: 5px 0 2px;
        padding: 7px 13px;
        border: 1px solid transparent;
        border-radius: 999px;
        font-size: 1.1rem;
        font-weight: 700;
    }

    .risk-lower {
        border-color: #b9ded0;
        background: #e8f5ef;
        color: #246446;
    }

    .risk-elevated {
        border-color: #f0d0b4;
        background: #fff1e5;
        color: #8c4c20;
    }

    .notice {
        padding: 14px 17px;
        border: 1px solid #f0dfbd;
        border-left: 4px solid #d79b46;
        border-radius: 8px;
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
        padding: 24px;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: var(--white);
    }

    div[data-testid="stForm"] h3 {
        margin-top: 0.8rem;
        padding-bottom: 0.55rem;
        border-bottom: 1px solid var(--line);
    }

    div[data-testid="stForm"] [data-testid="stFormSubmitButton"] {
        padding-top: 0.75rem;
    }

    [data-testid="stSidebar"] [data-testid="stPageLink"] a {
        border-radius: 8px;
        font-weight: 600;
    }

    [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
        background: var(--teal-soft);
        color: var(--teal);
    }

    [data-testid="stMetric"] {
        min-height: 98px;
        padding: 16px 18px;
        border: 1px solid var(--line);
        border-radius: 10px;
        background: var(--white);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 10px;
        overflow: hidden;
    }

    @media (max-width: 800px) {
        [data-testid="stMainBlockContainer"] {
            padding-top: 1.25rem;
        }
        .hero { padding: 24px 20px; }
        .section-heading { margin-top: 24px; }
        div[data-testid="stForm"] { padding: 14px; }
        [data-testid="stMetric"] { min-height: unset; }
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


def _show_medical_disclaimer():
    st.markdown(
        '<div class="notice"><strong>Medical disclaimer:</strong> This application is an '
        'educational/academic machine-learning project and is not a professional medical diagnosis.</div>',
        unsafe_allow_html=True,
    )


def _display_prediction(result, show_hybrid_weights=True):
    st.subheader("Hybrid Risk Assessment")
    st.info("Academic Model Estimate — Not a Medical Diagnosis")
    prediction_label = "Elevated Risk" if result["hybrid_prediction"] else "Lower Risk"
    risk_class = "risk-elevated" if result["hybrid_prediction"] else "risk-lower"
    with st.container(border=True):
        metrics = st.columns(2)
        with metrics[0]:
            st.markdown("**Predicted Risk**")
            st.markdown(
                f'<span class="risk-state {risk_class}">{prediction_label}</span>',
                unsafe_allow_html=True,
            )
        metrics[1].metric("Hybrid Probability", f"{result['hybrid_probability']:.1%}")

    detail_columns = st.columns(2 if show_hybrid_weights else 1)
    with detail_columns[0]:
        st.markdown("### Individual Model Probabilities")
        probabilities = pd.DataFrame(
            [
                {"Model": name, "Probability": f"{probability:.1%}"}
                for name, probability in result["individual_probabilities"].items()
            ]
        )
        st.dataframe(probabilities, hide_index=True, use_container_width=True)
    if show_hybrid_weights:
        with detail_columns[1]:
            st.markdown("### Hybrid Model Weights")
            weights = pd.DataFrame(
                [
                    {"Hybrid model": name, "Validation-F1 Weight": f"{weight:.1%}"}
                    for name, weight in result["hybrid_weights"].items()
                    if name in result["hybrid_models"]
                ]
            )
            st.dataframe(weights, hide_index=True, use_container_width=True)
    st.caption(
        "The hybrid prediction combines the probabilities of the selected top-performing "
        "models using validation-F1-based weights."
    )


def show_home():
    st.markdown(
        """
        <section class="hero">
            <p class="eyebrow">Academic AI research dashboard</p>
            <h1>Intelligent Heart Disease Diagnosis using Hybrid AI Models</h1>
            <p class="hero-subtitle">An academic machine-learning research project</p>
            <p class="hero-copy">
                This project explores how six trained machine-learning models and a
                validation-F1-weighted hybrid approach can be brought together to estimate risk
                from structured patient features.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Start Risk Assessment",
        type="primary",
        icon="➡️",
        use_container_width=True,
    ):
        st.switch_page(assessment_page)
    st.caption("Enter 13 project features to view an academic model estimate.")

    st.markdown('<p class="section-heading">At a glance</p>', unsafe_allow_html=True)
    cards = st.columns(4)
    card_content = [
        ("✦", "6 Models", "Classical machine learning and neural networks"),
        ("⌘", "13 Features", "Structured project input variables"),
        ("⟲", "Hybrid Method", "Validation-F1-weighted probabilities"),
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

    st.markdown('<p class="section-heading">The six individual models</p>', unsafe_allow_html=True)
    model_columns = st.columns(3)
    for index, (model_name, description) in enumerate(MODEL_DESCRIPTIONS.items()):
        with model_columns[index % len(model_columns)]:
            with st.container(border=True):
                st.markdown(f"**{model_name}**")
                st.caption(description)

    st.markdown('<p class="section-heading">Hybrid AI approach</p>', unsafe_allow_html=True)
    with st.container(border=True):
        st.write(
            "The hybrid model combines probabilities from the selected models using "
            "validation-F1-based weights stored in the project deployment configuration. "
            "This brings together distinct modeling approaches; it does not guarantee improved "
            "performance or replace evaluation of each model."
        )

    _show_medical_disclaimer()


def show_assessment():
    st.markdown('<p class="eyebrow">Patient feature entry</p>', unsafe_allow_html=True)
    st.title("Risk Assessment")
    st.write("Enter patient features using clear clinical labels and units.")

    with st.form("risk_assessment_form"):
        st.subheader("Patient Profile")
        profile = st.columns(3)
        with profile[0]:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
        with profile[1]:
            sex = st.selectbox(
                "Sex",
                options=list(SEX_LABELS),
                format_func=lambda value: SEX_LABELS[value],
                help="Select the recorded sex category.",
            )
        with profile[2]:
            cp = st.selectbox(
                "Chest Pain Type",
                options=list(CHEST_PAIN_LABELS),
                format_func=lambda value: CHEST_PAIN_LABELS[value],
                help="Select the chest-pain category recorded for this input.",
            )

        st.subheader("Clinical Measurements")
        measurements = st.columns(3)
        with measurements[0]:
            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=50,
                max_value=250,
                value=120,
                step=1,
                help="Resting blood pressure measurement.",
            )
        with measurements[1]:
            chol = st.number_input(
                "Cholesterol (mg/dL)",
                min_value=50,
                max_value=700,
                value=200,
                step=1,
                help="Serum cholesterol measurement.",
            )
        with measurements[2]:
            fbs = st.selectbox(
                "Fasting Blood Sugar",
                options=list(FASTING_BLOOD_SUGAR_LABELS),
                format_func=lambda value: FASTING_BLOOD_SUGAR_LABELS[value],
                help="Whether fasting blood sugar is above 120 mg/dL.",
            )

        st.subheader("ECG / Exercise Information")
        ecg_exercise = st.columns(3)
        with ecg_exercise[0]:
            restecg = st.selectbox(
                "Resting ECG",
                options=list(RESTING_ECG_LABELS),
                format_func=lambda value: RESTING_ECG_LABELS[value],
                help="Resting electrocardiogram category.",
            )
        with ecg_exercise[1]:
            thalach = st.number_input(
                "Maximum Heart Rate (bpm)",
                min_value=40,
                max_value=250,
                value=150,
                step=1,
                help="Maximum heart rate recorded during exercise.",
            )
        with ecg_exercise[2]:
            exang = st.selectbox(
                "Exercise-Induced Angina",
                options=list(EXERCISE_ANGINA_LABELS),
                format_func=lambda value: EXERCISE_ANGINA_LABELS[value],
                help="Whether exercise induced angina.",
            )

        remaining = st.columns(3)
        with remaining[0]:
            oldpeak = st.number_input(
                "ST Depression (oldpeak)",
                min_value=-3.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="ST depression induced by exercise relative to rest.",
            )
        with remaining[1]:
            slope = st.selectbox(
                "ST Slope",
                options=list(SLOPE_LABELS),
                format_func=lambda value: SLOPE_LABELS[value],
                help="Slope of the peak exercise ST segment.",
            )
        with remaining[2]:
            ca = st.selectbox(
                "Major Vessels",
                options=list(MAJOR_VESSEL_LABELS),
                format_func=lambda value: MAJOR_VESSEL_LABELS[value],
                help="Number of major vessels colored by fluoroscopy.",
            )

        thal_column = st.columns(3)
        with thal_column[0]:
            thal = st.selectbox(
                "Thalassemia",
                options=list(THAL_LABELS),
                format_func=lambda value: THAL_LABELS[value],
                help="Thalassemia category recorded for this input.",
            )
        submitted = st.form_submit_button(
            "Analyze Heart Disease Risk",
            type="primary",
            icon="🔎",
            use_container_width=True,
        )

    if submitted:
        st.session_state.pop("latest_prediction", None)
        patient_data = {
            "age": int(age),
            "sex": int(sex),
            "cp": int(cp),
            "trestbps": int(trestbps),
            "chol": int(chol),
            "fbs": int(fbs),
            "restecg": int(restecg),
            "thalach": int(thalach),
            "exang": int(exang),
            "oldpeak": float(oldpeak),
            "slope": int(slope),
            "ca": int(ca),
            "thal": int(thal),
        }
        try:
            prediction = predict_patient(patient_data)
            st.session_state["latest_prediction"] = prediction
            _display_prediction(prediction)
        except Exception as error:
            st.error(f"Unable to calculate the risk estimate: {error}")

    _show_medical_disclaimer()


def show_model_insights():
    st.markdown('<p class="eyebrow">Model transparency</p>', unsafe_allow_html=True)
    st.title("Model Insights")
    st.write(
        "The project evaluates six individual classifiers. The hybrid uses a configured subset "
        "and combines its model probabilities with stored validation-F1 weights."
    )

    latest_prediction = st.session_state.get("latest_prediction")
    if latest_prediction:
        _display_prediction(latest_prediction, show_hybrid_weights=False)
        hybrid_models = latest_prediction["hybrid_models"]
        hybrid_weights = latest_prediction["hybrid_weights"]
    else:
        st.info("Complete a risk assessment to view its model probabilities and hybrid result here.")
        try:
            deployment_config = load_artifacts()["deployment_config"]
            hybrid_models = deployment_config["HYBRID_MODELS"]
            hybrid_weights = deployment_config["HYBRID_WEIGHTS"]
        except Exception as error:
            hybrid_models = []
            hybrid_weights = {}
            st.error(f"Unable to load model artifacts: {error}")

    st.markdown('<p class="section-heading">How the models contribute</p>', unsafe_allow_html=True)
    model_columns = st.columns(3)
    for index, (model_name, description) in enumerate(MODEL_DESCRIPTIONS.items()):
        role = "Participates in hybrid" if model_name in hybrid_models else "Individual model"
        with model_columns[index % len(model_columns)]:
            with st.container(border=True):
                st.markdown(f"**{model_name}**")
                st.caption(description)
                st.caption(role)

    st.markdown('<p class="section-heading">Hybrid Model Weights</p>', unsafe_allow_html=True)
    configured_hybrid = pd.DataFrame(
        [
            {
                "Hybrid model": model_name,
                "Validation-F1 Weight": f"{hybrid_weights[model_name]:.1%}",
            }
            for model_name in hybrid_models
            if model_name in hybrid_weights
        ]
    )
    if not configured_hybrid.empty:
        st.dataframe(configured_hybrid, hide_index=True, use_container_width=True)
    st.caption(
        "Combining different model approaches can provide a single weighted estimate from "
        "complementary model outputs. This does not establish that the hybrid is more accurate."
    )

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
    _show_medical_disclaimer()


def show_about():
    st.markdown('<p class="eyebrow">Project overview</p>', unsafe_allow_html=True)
    st.title("About")
    st.write(
        "An academic engineering project exploring hybrid machine-learning methods for "
        "structured heart-disease risk estimation."
    )

    about_columns = st.columns(2)
    with about_columns[0]:
        with st.container(border=True):
            st.markdown("**Project objective**")
            st.write(
                "Compare six trained classification approaches and present a hybrid estimate "
                "from patient feature inputs, while keeping individual probabilities visible."
            )
            st.markdown("**Dataset**")
            st.write("UCI Heart Disease Dataset.")
            st.markdown("**Technologies**")
            st.write(
                "Python · Pandas · NumPy · Scikit-learn · XGBoost · TensorFlow/Keras · "
                "Streamlit · SHAP"
            )
        with st.container(border=True):
            st.markdown("**Explainability**")
            st.write(
                "The application can display saved global XGBoost SHAP feature-importance "
                "results when the corresponding results artifact is available. Feature "
                "importance describes model behavior and does not establish medical causation."
            )
    with about_columns[1]:
        with st.container(border=True):
            st.markdown("**Models**")
            st.write(", ".join(MODEL_DESCRIPTIONS))
            st.markdown("**Hybrid voting approach**")
            st.write(
                "The deployed configuration selects hybrid models and stores their normalized "
                "validation-F1-based weights. The application displays these configured values "
                "without recalculating them."
            )
        with st.container(border=True):
            st.markdown("**Scope and limitations**")
            st.write(
                "This is a dataset-based academic project, not a validated clinical tool. "
                "Model outputs may not generalize to all populations or settings and should not "
                "replace professional medical judgment."
            )

    _show_medical_disclaimer()


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