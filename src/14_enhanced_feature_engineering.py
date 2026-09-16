import os
import numpy as np
import pandas as pd


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

RESULTS_DIR = os.path.join(BASE_DIR, "results")
FEATURES_DIR = os.path.join(BASE_DIR, "features")

os.makedirs(FEATURES_DIR, exist_ok=True)


METADATA_COLUMNS = [
    "MaterialID",
    "StepID",
    "duration_ms",
    "is_test",
    "target"
]


def engineer_enhanced_features(df):

    df = df.copy()

    # Preserve original source order
    df["_source_order"] = np.arange(len(df))

    # Verified trajectory ordering
    df = df.sort_values(
        ["MaterialID", "duration_ms", "_source_order"]
    ).reset_index(drop=True)

    raw_features = [
        col for col in df.columns
        if col.startswith("feature_")
        and not any(
            col.endswith(suffix)
            for suffix in [
                "_delta",
                "_slope",
                "_rolling_std",
                "_spike"
            ]
        )
    ]

    output = df[
        [
            col for col in [
                "MaterialID",
                "StepID",
                "duration_ms",
                "is_test",
                "target"
            ]
            if col in df.columns
        ]
    ].copy()

    # =========================================================
    # 1. LEVEL
    # =========================================================

    for feature in raw_features:
        output[feature] = df[feature]

    # =========================================================
    # 2. DELTA
    # =========================================================

    deltas = {}

    for feature in raw_features:

        previous = df.groupby("MaterialID")[feature].shift(1)

        delta = df[feature] - previous

        deltas[feature] = delta

        output[f"{feature}_delta"] = delta

    # =========================================================
    # 3. SLOPE / DRIFT
    # =========================================================

    slopes = {}

    previous_time = (
        df.groupby("MaterialID")["duration_ms"]
        .shift(1)
    )

    dt = df["duration_ms"] - previous_time

    # Invalid/non-positive elapsed time is not used
    dt_safe = dt.where(dt > 0, np.nan)

    for feature in raw_features:

        slope = deltas[feature] / dt_safe

        slopes[feature] = slope

        output[f"{feature}_slope"] = slope

    # =========================================================
    # 4. ROLLING VARIABILITY
    # =========================================================

    for feature in raw_features:

        rolling_std = (
            df.groupby("MaterialID")[feature]
            .transform(
                lambda x:
                x.rolling(
                    window=3,
                    min_periods=2
                ).std()
            )
        )

        output[
            f"{feature}_rolling_std"
        ] = rolling_std

    # =========================================================
    # 5. SPIKE / DROP MAGNITUDE
    # =========================================================

    for feature in raw_features:

        output[
            f"{feature}_spike"
        ] = deltas[feature].abs()

    # =========================================================
    # 6. ACCELERATION / CURVATURE
    # =========================================================

    for feature in raw_features:

        previous_slope = (
            pd.Series(slopes[feature])
            .groupby(df["MaterialID"])
            .shift(1)
        )

        acceleration = slopes[feature] - previous_slope

        output[
            f"{feature}_acceleration"
        ] = acceleration

    # =========================================================
    # 7. STABILIZATION
    # =========================================================

    for feature in raw_features:

        abs_delta = deltas[feature].abs()

        stabilization = (
            abs_delta
            .groupby(df["MaterialID"])
            .rolling(
                window=3,
                min_periods=2
            )
            .mean()
            .reset_index(
                level=0,
                drop=True
            )
        )

        # Reset index to match dataframe rows
        stabilization = (
            stabilization
            .reset_index(drop=True)
        )

        output[
            f"{feature}_stabilization"
        ] = stabilization

    # =========================================================
    # 8. TRAJECTORY SHAPE
    # =========================================================

    # Local cumulative change from the beginning
    for feature in raw_features:

        first_value = (
            df.groupby("MaterialID")[feature]
            .transform("first")
        )

        trajectory_change = (
            df[feature] - first_value
        )

        output[
            f"{feature}_trajectory_change"
        ] = trajectory_change

    # =========================================================
    # 9. CROSS-PARAMETER RELATIONSHIPS
    # =========================================================

    # Row-wise mean of measurement parameters
    output["measurement_mean"] = (
        df[raw_features].mean(axis=1)
    )

    # Row-wise standard deviation
    output["measurement_std"] = (
        df[raw_features].std(axis=1)
    )

    # Distance from the row's parameter mean
    for feature in raw_features:

        output[
            f"{feature}_peer_deviation"
        ] = (
            df[feature]
            - df[raw_features].mean(axis=1)
        )

    return output


def process_dataset(dataset_name):

    splits = [
        "train",
        "validation",
        "final_test"
    ]

    print("\n" + "=" * 70)
    print(f"ENHANCED FEATURE ENGINEERING - {dataset_name}")
    print("=" * 70)

    for split in splits:

        input_file = os.path.join(
            RESULTS_DIR,
            f"{dataset_name}_{split}.csv"
        )

        output_file = os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_{split}_enhanced.csv"
        )

        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f"File not found: {input_file}"
            )

        df = pd.read_csv(input_file)

        print(f"\n{split.upper()}")
        print("Raw shape:", df.shape)
        print(
            "MaterialID groups:",
            df["MaterialID"].nunique()
        )

        result = engineer_enhanced_features(df)

        # Check numerical problems
        numeric_features = result.select_dtypes(
            include=np.number
        )

        infinite_count = np.isinf(
            numeric_features
        ).sum().sum()

        print("Enhanced shape:", result.shape)
        print(
            "Missing values:",
            result.isna().sum().sum()
        )
        print(
            "Infinite values:",
            infinite_count
        )

        result.to_csv(
            output_file,
            index=False
        )

        print("Saved:", output_file)


if __name__ == "__main__":

    process_dataset("D1")
    process_dataset("D2")

    print("\n" + "=" * 70)
    print("STEP 14 COMPLETE")
    print("=" * 70)