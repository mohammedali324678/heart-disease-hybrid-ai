import os
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


print("=" * 50)
print("REBUILDING XGBOOST MODEL")
print("=" * 50)


# --------------------------------------------------
# 1. LOAD UCI HEART DISEASE DATASET
# --------------------------------------------------

heart_disease = fetch_ucirepo(id=45)

X_raw = heart_disease.data.features.copy()
y_raw = heart_disease.data.targets.copy()

y = (y_raw["num"] > 0).astype(int)

print(f"Dataset shape: {X_raw.shape}")


# --------------------------------------------------
# 2. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_raw,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training shape: {X_train.shape}")
print(f"Testing shape:  {X_test.shape}")


# --------------------------------------------------
# 3. TRAIN-SAFE IMPUTATION
# --------------------------------------------------

ca_median = X_train["ca"].median()
thal_median = X_train["thal"].median()

X_train["ca"] = X_train["ca"].fillna(ca_median)
X_train["thal"] = X_train["thal"].fillna(thal_median)

X_test["ca"] = X_test["ca"].fillna(ca_median)
X_test["thal"] = X_test["thal"].fillna(thal_median)

print(f"ca median:   {ca_median}")
print(f"thal median: {thal_median}")


# --------------------------------------------------
# 4. SMOTE ON FULL TRAINING DATA
# --------------------------------------------------

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("After SMOTE:")
print(y_train_smote.value_counts())


# --------------------------------------------------
# 5. LOAD EXISTING SCALER
# --------------------------------------------------

models_dir = Path("models")

scaler_path = models_dir / "scaler.joblib"

if scaler_path.exists():
    print("Loading existing scaler...")
    scaler = joblib.load(scaler_path)

    X_train_scaled = scaler.transform(X_train_smote)

else:
    print("Existing scaler not found. Creating scaler...")
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_smote
    )

    joblib.dump(
        scaler,
        scaler_path
    )


# --------------------------------------------------
# 6. BUILD FRESH XGBOOST MODEL
# --------------------------------------------------

print("\nTraining fresh XGBoost model...")

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(
    X_train_scaled,
    y_train_smote
)

print("XGBoost training completed.")


# --------------------------------------------------
# 7. TEST THE MODEL BEFORE SAVING
# --------------------------------------------------

X_test_scaled = scaler.transform(X_test)

test_predictions = xgb_model.predict(X_test_scaled)

test_probability = xgb_model.predict_proba(
    X_test_scaled
)[:, 1]

print("\nXGBoost test prediction successful.")

print("First 10 predictions:")
print(test_predictions[:10])

print("\nFirst 10 probabilities:")
print(test_probability[:10])


# --------------------------------------------------
# 8. SAVE XGBOOST MODEL
# --------------------------------------------------

output_path = models_dir / "xgboost.joblib"

# Remove old corrupted file
if output_path.exists():
    print("\nRemoving old corrupted xgboost.joblib...")
    output_path.unlink()

# Save fresh model
joblib.dump(
    xgb_model,
    output_path
)

print(f"\nSaved new model to: {output_path}")


# --------------------------------------------------
# 9. IMMEDIATELY RELOAD IT
# --------------------------------------------------

print("\nTesting saved model...")

loaded_model = joblib.load(output_path)

loaded_predictions = loaded_model.predict(
    X_test_scaled
)

loaded_probability = loaded_model.predict_proba(
    X_test_scaled
)[:, 1]

print("Reload successful.")
print("Prediction successful.")
print("Probability successful.")


# --------------------------------------------------
# 10. FINAL CHECK
# --------------------------------------------------

print("\n" + "=" * 50)
print("XGBOOST REPAIR SUCCESSFUL")
print("=" * 50)

print(f"Model file: {output_path}")
print(f"File size: {output_path.stat().st_size:,} bytes")

print("\nXGBoost model is now valid and loadable.")
print("=" * 50)
