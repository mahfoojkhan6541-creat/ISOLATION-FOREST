import os
import pickle
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"
MODELS_DIR = os.path.join(BASE_DIR, "models")


print("=" * 70)
print("STEP 35 - V3 TRAIN-ONLY PREPROCESSING")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load prepared V3 data
# ---------------------------------------------------------

X_train = np.load(
    os.path.join(BASE_DIR, "features", "D2_v3_X_train.npy"))

X_validation = np.load(
    os.path.join(BASE_DIR, "features", "D2_v3_X_validation.npy")
)

X_final_test = np.load(
    os.path.join(BASE_DIR, "features", "D2_v3_X_final_test.npy")
)


print("\nOriginal shapes:")
print("TRAIN      :", X_train.shape)
print("VALIDATION :", X_validation.shape)
print("FINAL TEST :", X_final_test.shape)


# ---------------------------------------------------------
# 2. Check for invalid values
# ---------------------------------------------------------

for name, X in [
    ("TRAIN", X_train),
    ("VALIDATION", X_validation),
    ("FINAL TEST", X_final_test)
]:
    print(
        f"{name} - NaN: {np.isnan(X).sum()}, "
        f"Inf: {np.isinf(X).sum()}"
    )


# ---------------------------------------------------------
# 3. Fit imputer ONLY on TRAIN
# ---------------------------------------------------------

imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)

X_validation_imputed = imputer.transform(X_validation)

X_final_test_imputed = imputer.transform(X_final_test)


print("\nMedian imputer:")
print("Fitted on TRAIN only.")


# ---------------------------------------------------------
# 4. Fit scaler ONLY on TRAIN
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)

X_validation_scaled = scaler.transform(X_validation_imputed)

X_final_test_scaled = scaler.transform(X_final_test_imputed)


print("\nStandardScaler:")
print("Fitted on TRAIN only.")


# ---------------------------------------------------------
# 5. Verify output
# ---------------------------------------------------------

print("\nProcessed shapes:")
print("TRAIN      :", X_train_scaled.shape)
print("VALIDATION :", X_validation_scaled.shape)
print("FINAL TEST :", X_final_test_scaled.shape)


print("\nTRAIN mean (first 5 features):")
print(np.mean(X_train_scaled[:, :5], axis=0))

print("\nTRAIN std (first 5 features):")
print(np.std(X_train_scaled[:, :5], axis=0))


# ---------------------------------------------------------
# 6. Save processed data
# ---------------------------------------------------------

np.save(
    os.path.join(BASE_DIR, "features", "D2_v3_X_train_scaled.npy"),
    X_train_scaled
)

np.save(
    os.path.join(BASE_DIR, "features", "D2_v3_X_validation_scaled.npy"),
    X_validation_scaled
)

np.save(
    os.path.join(BASE_DIR, "features", "D2_v3_X_final_test_scaled.npy"),
    X_final_test_scaled
)


# ---------------------------------------------------------
# 7. Save preprocessing objects
# ---------------------------------------------------------

preprocessing = {
    "imputer": imputer,
    "scaler": scaler,
    "feature_count": X_train_scaled.shape[1],
    "dataset": "D2",
    "version": "V3",
    "fit_rule": "TRAIN ONLY"
}


preprocessing_path = os.path.join(
    MODELS_DIR,
    "D2_v3_preprocessing.pkl"
)

with open(preprocessing_path, "wb") as f:
    pickle.dump(preprocessing, f)


print("\nSaved:")
print("  D2_v3_X_train_scaled.npy")
print("  D2_v3_X_validation_scaled.npy")
print("  D2_v3_X_final_test_scaled.npy")
print("  D2_v3_preprocessing.pkl")


print("\n" + "=" * 70)
print("STEP 35 COMPLETE")
print("=" * 70)

print("\nIMPORTANT:")
print("Preprocessing was fitted ONLY on TRAIN.")
print("Validation was transformed using TRAIN preprocessing.")
print("Final Test was transformed using TRAIN preprocessing.")