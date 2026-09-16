import os
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"
FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")


print("=" * 70)
print("STEP 36 - V3 ISOLATION FOREST TRAINING")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load TRAIN data only
# ---------------------------------------------------------

X_train = np.load(
    os.path.join(FEATURES_DIR, "D2_v3_X_train_scaled.npy")
)

print("\nTraining data:")
print("Shape:", X_train.shape)


# ---------------------------------------------------------
# 2. Train Isolation Forest
# ---------------------------------------------------------

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=300,
    max_samples="auto",
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train)


# ---------------------------------------------------------
# 3. Verify model
# ---------------------------------------------------------

print("\nModel configuration:")
print("Algorithm      : Isolation Forest")
print("Trees          :", model.n_estimators)
print("Max samples    :", model.max_samples)
print("Contamination  :", model.contamination)
print("Random state   :", model.random_state)


# ---------------------------------------------------------
# 4. Check training predictions
# ---------------------------------------------------------

train_predictions = model.predict(X_train)

normal_count = np.sum(train_predictions == 1)
abnormal_count = np.sum(train_predictions == -1)

print("\nTraining predictions:")
print("NORMAL   :", normal_count)
print("ABNORMAL :", abnormal_count)


# ---------------------------------------------------------
# 5. Save model
# ---------------------------------------------------------

model_path = os.path.join(
    MODELS_DIR,
    "D2_v3_isolation_forest.pkl"
)

with open(model_path, "wb") as f:
    pickle.dump(model, f)


print("\nModel saved:")
print(model_path)


print("\n" + "=" * 70)
print("STEP 36 COMPLETE")
print("=" * 70)

print("\nIMPORTANT:")
print("- target was NOT used during training.")
print("- Model was trained ONLY on TRAIN data.")
print("- contamination was NOT set to the abnormal test percentage.")
print("- Final Test was NOT used.")