import os
import pickle
import numpy as np
import pandas as pd


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"
FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


print("=" * 70)
print("STEP 37 - V3 ANOMALY SCORES")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load trained model
# ---------------------------------------------------------

model_path = os.path.join(
    MODELS_DIR,
    "D2_v3_isolation_forest.pkl"
)

with open(model_path, "rb") as f:
    model = pickle.load(f)


# ---------------------------------------------------------
# 2. Load TRAIN and VALIDATION data
# ---------------------------------------------------------

X_train = np.load(
    os.path.join(FEATURES_DIR, "D2_v3_X_train_scaled.npy")
)

X_validation = np.load(
    os.path.join(FEATURES_DIR, "D2_v3_X_validation_scaled.npy")
)

y_train = np.load(
    os.path.join(FEATURES_DIR, "D2_v3_y_train.npy")
)

y_validation = np.load(
    os.path.join(FEATURES_DIR, "D2_v3_y_validation.npy")
)


print("\nLoaded data:")
print("TRAIN:")
print("  X:", X_train.shape)
print("  y:", y_train.shape)

print("\nVALIDATION:")
print("  X:", X_validation.shape)
print("  y:", y_validation.shape)


# ---------------------------------------------------------
# 3. Generate anomaly scores
# ---------------------------------------------------------
# Isolation Forest score_samples:
# higher value = more normal
#
# We multiply by -1 so that:
# higher anomaly_score = more abnormal
# ---------------------------------------------------------

train_scores = -model.score_samples(X_train)

validation_scores = -model.score_samples(X_validation)


print("\nAnomaly score ranges:")

print(
    "TRAIN      :",
    round(train_scores.min(), 6),
    "to",
    round(train_scores.max(), 6)
)

print(
    "VALIDATION :",
    round(validation_scores.min(), 6),
    "to",
    round(validation_scores.max(), 6)
)


# ---------------------------------------------------------
# 4. Create result tables
# ---------------------------------------------------------

train_results = pd.DataFrame({
    "anomaly_score": train_scores,
    "target": y_train
})

validation_results = pd.DataFrame({
    "anomaly_score": validation_scores,
    "target": y_validation
})


# ---------------------------------------------------------
# 5. Save results
# ---------------------------------------------------------

train_output = os.path.join(
    RESULTS_DIR,
    "D2_v3_train_anomaly_scores.csv"
)

validation_output = os.path.join(
    RESULTS_DIR,
    "D2_v3_validation_anomaly_scores.csv"
)


train_results.to_csv(
    train_output,
    index=False
)

validation_results.to_csv(
    validation_output,
    index=False
)


# ---------------------------------------------------------
# 6. Basic validation information
# ---------------------------------------------------------

print("\nValidation target distribution:")
print(
    validation_results["target"].value_counts().sort_index()
)

print("\nSaved:")
print(train_output)
print(validation_output)


print("\n" + "=" * 70)
print("STEP 37 COMPLETE")
print("=" * 70)

print("\nIMPORTANT:")
print("- Final Test was NOT used.")
print("- target was NOT used to generate anomaly scores.")
print("- target is present only for later validation/evaluation.")
print("- Higher anomaly_score means more abnormal.")