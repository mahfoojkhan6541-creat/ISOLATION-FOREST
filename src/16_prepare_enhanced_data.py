import os
import numpy as np
import pandas as pd
import joblib


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


METADATA_COLUMNS = [
    "MaterialID",
    "StepID",
    "duration_ms",
    "is_test",
    "target"
]


def prepare_dataset(dataset_name):

    print("\n" + "=" * 70)
    print(f"PREPARING ENHANCED DATA - {dataset_name}")
    print("=" * 70)

    train_file = os.path.join(
        FEATURES_DIR,
        f"{dataset_name}_train_enhanced.csv"
    )

    validation_file = os.path.join(
        FEATURES_DIR,
        f"{dataset_name}_validation_enhanced.csv"
    )

    test_file = os.path.join(
        FEATURES_DIR,
        f"{dataset_name}_final_test_enhanced.csv"
    )

    train_df = pd.read_csv(train_file)
    validation_df = pd.read_csv(validation_file)
    test_df = pd.read_csv(test_file)

    # ---------------------------------------------------------
    # Identify candidate model features
    # ---------------------------------------------------------

    candidate_features = [
        col for col in train_df.columns
        if col not in METADATA_COLUMNS
    ]

    # ---------------------------------------------------------
    # Remove constant features using TRAIN only
    # ---------------------------------------------------------

    constant_features = []

    for feature in candidate_features:

        if train_df[feature].nunique(
            dropna=True
        ) <= 1:

            constant_features.append(feature)

    model_features = [
        feature
        for feature in candidate_features
        if feature not in constant_features
    ]

    print("\nCandidate features:", len(candidate_features))

    print(
        "Constant features removed:",
        len(constant_features)
    )

    if constant_features:
        print("\nRemoved:")
        for feature in constant_features:
            print(" -", feature)

    print(
        "\nFinal model features:",
        len(model_features)
    )

    # ---------------------------------------------------------
    # Extract X
    # ---------------------------------------------------------

    X_train = train_df[model_features].copy()
    X_validation = validation_df[model_features].copy()
    X_final_test = test_df[model_features].copy()

    # ---------------------------------------------------------
    # Verify target is NOT inside X
    # ---------------------------------------------------------

    assert "target" not in X_train.columns
    assert "target" not in X_validation.columns
    assert "target" not in X_final_test.columns

    # ---------------------------------------------------------
    # Verify metadata is NOT inside X
    # ---------------------------------------------------------

    for column in METADATA_COLUMNS:

        assert column not in X_train.columns
        assert column not in X_validation.columns
        assert column not in X_final_test.columns

    # ---------------------------------------------------------
    # Convert to NumPy
    # ---------------------------------------------------------

    X_train = X_train.to_numpy(dtype=np.float64)
    X_validation = X_validation.to_numpy(dtype=np.float64)
    X_final_test = X_final_test.to_numpy(dtype=np.float64)

    # ---------------------------------------------------------
    # Save arrays
    # ---------------------------------------------------------

    np.save(
        os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_enhanced_X_train.npy"
        ),
        X_train
    )

    np.save(
        os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_enhanced_X_validation.npy"
        ),
        X_validation
    )

    np.save(
        os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_enhanced_X_final_test.npy"
        ),
        X_final_test
    )

    # ---------------------------------------------------------
    # Save feature configuration
    # ---------------------------------------------------------

    config = {
        "dataset": dataset_name,
        "candidate_features": candidate_features,
        "removed_constant_features": constant_features,
        "model_features": model_features,
        "metadata_columns": METADATA_COLUMNS
    }

    config_file = os.path.join(
        MODELS_DIR,
        f"{dataset_name}_enhanced_feature_config.pkl"
    )

    joblib.dump(
        config,
        config_file
    )

    # ---------------------------------------------------------
    # Final verification
    # ---------------------------------------------------------

    print("\nTRAIN shape:", X_train.shape)
    print(
        "VALIDATION shape:",
        X_validation.shape
    )
    print(
        "FINAL TEST shape:",
        X_final_test.shape
    )

    print(
        "\nTarget remains available only in source CSVs "
        "for evaluation."
    )

    print(
        "Target used as model feature: NO"
    )

    print(
        "Saved feature configuration:",
        config_file
    )


if __name__ == "__main__":

    prepare_dataset("D1")
    prepare_dataset("D2")

    print("\n" + "=" * 70)
    print("STEP 16 COMPLETE")
    print("=" * 70)