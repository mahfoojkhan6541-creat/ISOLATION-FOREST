import os
import numpy as np
import pandas as pd
import joblib


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

FEATURE_DIR = os.path.join(
    BASE_DIR,
    "features"
)

INPUT_FILE = os.path.join(
    FEATURE_DIR,
    "D2_v3_engineered.csv"
)

TRAIN_FILE = os.path.join(
    FEATURE_DIR,
    "D2_train_engineered.csv"
)

VALIDATION_FILE = os.path.join(
    FEATURE_DIR,
    "D2_validation_engineered.csv"
)

FINAL_TEST_FILE = os.path.join(
    FEATURE_DIR,
    "D2_final_test_engineered.csv"
)


METADATA_COLUMNS = [
    "MaterialID",
    "StepID",
    "duration_ms",
    "is_test",
    "target"
]


def main():

    print("\n" + "=" * 70)
    print("STEP 34 - PREPARE V3 DATA")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"V3 feature file not found:\n{INPUT_FILE}"
        )

    # ---------------------------------------------------------
    # Load V3 features
    # ---------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    train_reference = pd.read_csv(TRAIN_FILE)
    validation_reference = pd.read_csv(VALIDATION_FILE)
    final_reference = pd.read_csv(FINAL_TEST_FILE)

    # ---------------------------------------------------------
    # Existing MaterialID split
    # ---------------------------------------------------------

    train_groups = set(
        train_reference["MaterialID"].unique()
    )

    validation_groups = set(
        validation_reference["MaterialID"].unique()
    )

    final_groups = set(
        final_reference["MaterialID"].unique()
    )

    # ---------------------------------------------------------
    # Check group overlap
    # ---------------------------------------------------------

    overlap_train_validation = (
        train_groups & validation_groups
    )

    overlap_train_final = (
        train_groups & final_groups
    )

    overlap_validation_final = (
        validation_groups & final_groups
    )

    if (
        overlap_train_validation
        or overlap_train_final
        or overlap_validation_final
    ):
        raise ValueError(
            "MaterialID leakage detected between splits."
        )

    # ---------------------------------------------------------
    # Assign rows using MaterialID
    # ---------------------------------------------------------

    train_df = df[
        df["MaterialID"].isin(train_groups)
    ].copy()

    validation_df = df[
        df["MaterialID"].isin(validation_groups)
    ].copy()

    final_df = df[
        df["MaterialID"].isin(final_groups)
    ].copy()

    # ---------------------------------------------------------
    # Feature columns
    # ---------------------------------------------------------

    feature_columns = [
        c for c in df.columns
        if c not in METADATA_COLUMNS
    ]

    # ---------------------------------------------------------
    # Remove constant features using TRAIN ONLY
    # ---------------------------------------------------------

    train_features = train_df[
        feature_columns
    ]

    constant_features = [
        col
        for col in feature_columns
        if train_features[col].nunique(
            dropna=False
        ) <= 1
    ]

    feature_columns = [
        col
        for col in feature_columns
        if col not in constant_features
    ]

    print(
        f"\nInitial V3 features: "
        f"{len(feature_columns) + len(constant_features)}"
    )

    print(
        f"Constant features removed: "
        f"{len(constant_features)}"
    )

    if constant_features:

        print("\nRemoved features:")

        for col in constant_features:
            print(
                f"  {col}"
            )

    print(
        f"\nFinal V3 model features: "
        f"{len(feature_columns)}"
    )

    # ---------------------------------------------------------
    # Prepare X and y
    # ---------------------------------------------------------

    X_train = train_df[
        feature_columns
    ].to_numpy(dtype=float)

    X_validation = validation_df[
        feature_columns
    ].to_numpy(dtype=float)

    X_final = final_df[
        feature_columns
    ].to_numpy(dtype=float)

    y_train = train_df[
        "target"
    ].to_numpy(dtype=int)

    y_validation = validation_df[
        "target"
    ].to_numpy(dtype=int)

    y_final = final_df[
        "target"
    ].to_numpy(dtype=int)

    # ---------------------------------------------------------
    # Save arrays
    # ---------------------------------------------------------

    np.save(
        os.path.join(
            FEATURE_DIR,
            "D2_v3_X_train.npy"
        ),
        X_train
    )

    np.save(
        os.path.join(
            FEATURE_DIR,
            "D2_v3_X_validation.npy"
        ),
        X_validation
    )

    np.save(
        os.path.join(
            FEATURE_DIR,
            "D2_v3_X_final_test.npy"
        ),
        X_final
    )

    np.save(
        os.path.join(
            FEATURE_DIR,
            "D2_v3_y_train.npy"
        ),
        y_train
    )

    np.save(
        os.path.join(
            FEATURE_DIR,
            "D2_v3_y_validation.npy"
        ),
        y_validation
    )

    np.save(
        os.path.join(
            FEATURE_DIR,
            "D2_v3_y_final_test.npy"
        ),
        y_final
    )

    # ---------------------------------------------------------
    # Save feature configuration
    # ---------------------------------------------------------

    config = {
        "feature_columns": feature_columns,
        "removed_constant_features": constant_features,
        "metadata_columns": METADATA_COLUMNS,
        "group_column": "MaterialID",
        "target_column": "target"
    }

    config_file = os.path.join(
        BASE_DIR,
        "models",
        "D2_v3_feature_config.pkl"
    )

    joblib.dump(
        config,
        config_file
    )

    # ---------------------------------------------------------
    # Print shapes
    # ---------------------------------------------------------

    print("\nTRAIN:")
    print(
        f"X: {X_train.shape}"
    )
    print(
        f"Groups: {len(train_groups)}"
    )

    print("\nVALIDATION:")
    print(
        f"X: {X_validation.shape}"
    )
    print(
        f"Groups: {len(validation_groups)}"
    )

    print("\nFINAL TEST:")
    print(
        f"X: {X_final.shape}"
    )
    print(
        f"Groups: {len(final_groups)}"
    )

    print("\nTarget is NOT included in X.")

    print(
        "\nFeature configuration saved:"
    )

    print(config_file)

    print("\n" + "=" * 70)
    print("STEP 34 COMPLETE")
    print("=" * 70)

    print(
        "\nFinal Test has only been prepared."
    )

    print(
        "It has NOT been used for model selection."
    )


if __name__ == "__main__":
    main()