import os
import numpy as np
import pandas as pd


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

FEATURE_DIR = os.path.join(
    BASE_DIR,
    "features"
)


INPUT_FILE = os.path.join(
    FEATURE_DIR,
    "D2_v3_engineered.csv"
)


METADATA_COLUMNS = [
    "MaterialID",
    "StepID",
    "duration_ms",
    "is_test",
    "target"
]


def main():

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"V3 feature file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print("\n" + "=" * 70)
    print("D2 V3 FEATURE QUALITY CHECK")
    print("=" * 70)

    print(
        f"\nDataset shape: {df.shape}"
    )

    print(
        f"MaterialID groups: "
        f"{df['MaterialID'].nunique()}"
    )

    feature_columns = [
        c for c in df.columns
        if c not in METADATA_COLUMNS
    ]

    X = df[feature_columns]

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    missing_total = X.isna().sum().sum()

    missing_features = (
        X.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    # ---------------------------------------------------------
    # Infinite values
    # ---------------------------------------------------------

    infinite_total = np.isinf(
        X.to_numpy(
            dtype=float
        )
    ).sum()

    # ---------------------------------------------------------
    # Constant features
    # ---------------------------------------------------------

    constant_features = []

    for col in feature_columns:

        if X[col].nunique(
            dropna=False
        ) <= 1:

            constant_features.append(
                col
            )

    # ---------------------------------------------------------
    # More than 50% missing
    # ---------------------------------------------------------

    high_missing = []

    for col in feature_columns:

        missing_ratio = (
            X[col].isna().mean()
        )

        if missing_ratio > 0.50:

            high_missing.append(
                (
                    col,
                    missing_ratio
                )
            )

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print("\nModel feature count:")
    print(
        len(feature_columns)
    )

    print("\nTotal missing feature values:")
    print(
        missing_total
    )

    print("\nTotal infinite values:")
    print(
        infinite_total
    )

    print("\nConstant features:")
    if constant_features:
        for col in constant_features:
            print(
                f"  {col}"
            )
    else:
        print(
            "  None"
        )

    print(
        "\nFeatures with >50% missing:"
    )

    if high_missing:

        for col, ratio in high_missing:

            print(
                f"  {col}: "
                f"{ratio * 100:.2f}%"
            )

    else:

        print(
            "  None"
        )

    # ---------------------------------------------------------
    # Missing values by feature family
    # ---------------------------------------------------------

    print(
        "\nTop 15 features by missing values:"
    )

    print(
        missing_features.head(15)
        .to_string()
    )

    # ---------------------------------------------------------
    # Check target
    # ---------------------------------------------------------

    print(
        "\nTarget distribution:"
    )

    print(
        df["target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ---------------------------------------------------------
    # Check target consistency
    # ---------------------------------------------------------

    target_per_material = (
        df.groupby("MaterialID")["target"]
        .nunique()
    )

    inconsistent_groups = (
        target_per_material > 1
    ).sum()

    print(
        "\nMaterialID groups with "
        "multiple target values:"
    )

    print(
        inconsistent_groups
    )

    # ---------------------------------------------------------
    # Final status
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("QUALITY STATUS")
    print("-" * 70)

    if infinite_total > 0:

        print(
            "FAIL: Infinite values detected."
        )

    elif len(high_missing) > 0:

        print(
            "WARNING: Features with >50% missing detected."
        )

    elif inconsistent_groups > 0:

        print(
            "FAIL: Target is inconsistent within MaterialID."
        )

    else:

        print(
            "PASS: V3 feature quality is acceptable "
            "for the next development step."
        )

    print("\n" + "=" * 70)
    print("STEP 33 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()