import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FEATURE_DIR = BASE_DIR / "features"

FEATURE_DIR.mkdir(exist_ok=True)


def create_features(file_path, dataset_name):

    print("\n" + "=" * 70)
    print(f"{dataset_name} - FEATURE ENGINEERING")
    print("=" * 70)

    df = pd.read_csv(file_path)

    # Preserve original row order as provenance
    df["_source_order"] = np.arange(len(df))

    feature_columns = [
        col for col in df.columns
        if col.startswith("feature_")
    ]

    print(f"\nRaw measurement features: {len(feature_columns)}")

    # ---------------------------------------------------------
    # Order each MaterialID trajectory by verified time
    # ---------------------------------------------------------
    df = df.sort_values(
        ["MaterialID", "duration_ms", "_source_order"]
    ).copy()

    grouped = df.groupby("MaterialID", sort=False)

    engineered = df[
        [
            "MaterialID",
            "StepID",
            "duration_ms",
            "is_test",
            "target"
        ]
    ].copy()

    # ---------------------------------------------------------
    # Create behaviour features
    # ---------------------------------------------------------
    for feature in feature_columns:

        # 1. Raw level
        engineered[f"{feature}_level"] = df[feature]

        # Previous measurement
        previous = grouped[feature].shift(1)

        # Previous time
        previous_time = grouped["duration_ms"].shift(1)

        # Time difference
        dt = df["duration_ms"] - previous_time

        # -----------------------------------------------------
        # 2. Change / Delta
        # -----------------------------------------------------
        delta = df[feature] - previous

        engineered[f"{feature}_delta"] = delta

        # -----------------------------------------------------
        # 3. Slope / Trend
        # -----------------------------------------------------
        slope = delta / dt.replace(0, np.nan)

        engineered[f"{feature}_slope"] = slope

        # -----------------------------------------------------
        # 4. Rolling variability
        # -----------------------------------------------------
        rolling_std = (
            grouped[feature]
            .rolling(window=3, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )

        engineered[f"{feature}_rolling_std"] = rolling_std

        # -----------------------------------------------------
        # 5. Spike magnitude
        # -----------------------------------------------------
        engineered[f"{feature}_spike"] = delta.abs()

    # ---------------------------------------------------------
    # Replace invalid generated values
    # ---------------------------------------------------------
    generated_feature_columns = [
        col for col in engineered.columns
        if col.startswith("feature_")
    ]

    engineered[generated_feature_columns] = (
        engineered[generated_feature_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    # ---------------------------------------------------------
    # Save engineered dataset
    # ---------------------------------------------------------
    output_path = (
        FEATURE_DIR /
        f"{dataset_name}_engineered.csv"
    )

    engineered.to_csv(
        output_path,
        index=False
    )

    # ---------------------------------------------------------
    # Information
    # ---------------------------------------------------------
    print(f"\nOriginal features: {len(feature_columns)}")

    print(
        f"Engineered feature columns: "
        f"{len(generated_feature_columns)}"
    )

    print(
        f"Rows: {len(engineered):,}"
    )

    print(
        f"Generated NaN values: "
        f"{engineered[generated_feature_columns].isna().sum().sum():,}"
    )

    print(f"\nSaved:")
    print(output_path)

    return engineered


# D1
d1_features = create_features(
    DATA_DIR / "D1.csv",
    "D1"
)

# D2
d2_features = create_features(
    DATA_DIR / "D2.csv",
    "D2"
)

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)