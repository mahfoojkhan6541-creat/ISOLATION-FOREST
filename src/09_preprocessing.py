import os
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


DATASETS = ["D1", "D2"]

METADATA_COLUMNS = [
    "MaterialID",
    "StepID",
    "duration_ms",
    "is_test",
    "target"
]


def get_feature_columns(df):
    return [
        col for col in df.columns
        if col not in METADATA_COLUMNS
    ]


def process_dataset(dataset_name):

    print("\n" + "=" * 60)
    print(f"PREPROCESSING {dataset_name}")
    print("=" * 60)

    train_file = os.path.join(
        FEATURES_DIR,
        f"{dataset_name}_train_engineered.csv"
    )

    validation_file = os.path.join(
        FEATURES_DIR,
        f"{dataset_name}_validation_engineered.csv"
    )

    final_test_file = os.path.join(
        FEATURES_DIR,
        f"{dataset_name}_final_test_engineered.csv"
    )

    train_df = pd.read_csv(train_file)
    validation_df = pd.read_csv(validation_file)
    final_test_df = pd.read_csv(final_test_file)

    feature_columns = get_feature_columns(train_df)

    print("Total model features before quality filtering:",
          len(feature_columns))

    # Remove constant features based ONLY on TRAIN
    constant_features = [
        col
        for col in feature_columns
        if train_df[col].nunique(dropna=True) <= 1
    ]

    feature_columns = [
        col
        for col in feature_columns
        if col not in constant_features
    ]

    print("Constant features removed:", len(constant_features))

    if constant_features:
        print("Removed:")
        for col in constant_features:
            print(" ", col)

    # Keep only numeric model features
    for col in feature_columns:
        train_df[col] = pd.to_numeric(
            train_df[col],
            errors="coerce"
        )
        validation_df[col] = pd.to_numeric(
            validation_df[col],
            errors="coerce"
        )
        final_test_df[col] = pd.to_numeric(
            final_test_df[col],
            errors="coerce"
        )

    X_train = train_df[feature_columns]
    X_validation = validation_df[feature_columns]
    X_final_test = final_test_df[feature_columns]

    # Replace infinite values with NaN
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_validation = X_validation.replace([np.inf, -np.inf], np.nan)
    X_final_test = X_final_test.replace([np.inf, -np.inf], np.nan)

    # ---------------------------------------------------------
    # IMPUTER — FIT ONLY ON TRAIN
    # ---------------------------------------------------------

    imputer = SimpleImputer(strategy="median")

    X_train_imputed = imputer.fit_transform(X_train)

    X_validation_imputed = imputer.transform(X_validation)

    X_final_test_imputed = imputer.transform(X_final_test)

    # ---------------------------------------------------------
    # SCALER — FIT ONLY ON TRAIN
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_imputed)

    X_validation_scaled = scaler.transform(
        X_validation_imputed
    )

    X_final_test_scaled = scaler.transform(
        X_final_test_imputed
    )

    # Save processed arrays
    np.save(
        os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_X_train.npy"
        ),
        X_train_scaled
    )

    np.save(
        os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_X_validation.npy"
        ),
        X_validation_scaled
    )

    np.save(
        os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_X_final_test.npy"
        ),
        X_final_test_scaled
    )

    # Save preprocessing configuration
    preprocessing = {
        "feature_columns": feature_columns,
        "removed_constant_features": constant_features,
        "imputer": imputer,
        "scaler": scaler
    }

    joblib.dump(
        preprocessing,
        os.path.join(
            MODELS_DIR,
            f"{dataset_name}_preprocessing.pkl"
        )
    )

    print("\nFinal model features:", len(feature_columns))

    print("TRAIN shape:", X_train_scaled.shape)
    print("VALIDATION shape:", X_validation_scaled.shape)
    print("FINAL TEST shape:", X_final_test_scaled.shape)

    print("\nTrain preprocessing fitted successfully.")

    print(
        f"Saved: models/{dataset_name}_preprocessing.pkl"
    )


if __name__ == "__main__":

    for dataset in DATASETS:
        process_dataset(dataset)

    print("\n" + "=" * 60)
    print("STEP 9 COMPLETE")
    print("=" * 60)