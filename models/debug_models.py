from pathlib import Path
import json
import joblib

from tensorflow.keras.models import load_model

MODELS_DIR = Path("models")

print("\n==============================")
print("HEART DISEASE MODEL DIAGNOSTIC")
print("==============================\n")

# -----------------------------
# 1. JSON FILES
# -----------------------------
for filename in [
    "feature_names.json",
    "imputation_config.json",
    "deployment_config.json",
]:
    path = MODELS_DIR / filename

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"[OK] {filename}")
        print(f"     {data}\n")

    except Exception as e:
        print(f"[FAIL] {filename}")
        print(f"       {type(e).__name__}: {e}\n")


# -----------------------------
# 2. SCALER
# -----------------------------
print("------------------------------")
print("SCALER")
print("------------------------------")

try:
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    print("[OK] scaler.joblib")
    print(f"     Type: {type(scaler)}\n")
except Exception as e:
    print("[FAIL] scaler.joblib")
    print(f"       {type(e).__name__}: {e}\n")


# -----------------------------
# 3. JOBLIB MODELS
# -----------------------------
model_files = [
    "logistic_regression.joblib",
    "svm.joblib",
    "decision_tree.joblib",
    "knn.joblib",
    "xgboost.joblib",
]

print("------------------------------")
print("JOBLIB MODELS")
print("------------------------------")

loaded_models = {}

for filename in model_files:
    path = MODELS_DIR / filename

    try:
        model = joblib.load(path)
        loaded_models[filename] = model

        print(f"[OK]   {filename}")
        print(f"       Type: {type(model)}")

    except Exception as e:
        print(f"[FAIL] {filename}")
        print(f"       {type(e).__name__}: {e}")


# -----------------------------
# 4. ANN
# -----------------------------
print("\n------------------------------")
print("ANN MODEL")
print("------------------------------")

ann_path = MODELS_DIR / "ann_model.keras"

try:
    ann_model = load_model(ann_path)

    print("[OK] ann_model.keras")
    print(f"     Type: {type(ann_model)}")
    print(f"     Input shape: {ann_model.input_shape}")
    print(f"     Output shape: {ann_model.output_shape}")

except Exception as e:
    print("[FAIL] ann_model.keras")
    print(f"       {type(e).__name__}: {e}")


# -----------------------------
# 5. FILE SIZES
# -----------------------------
print("\n------------------------------")
print("FILE SIZES")
print("------------------------------")

for path in sorted(MODELS_DIR.iterdir()):
    if path.is_file():
        print(f"{path.name:30} {path.stat().st_size:,} bytes")


print("\n==============================")
print("DIAGNOSTIC COMPLETE")
print("==============================")