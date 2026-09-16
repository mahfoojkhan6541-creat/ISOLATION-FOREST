import os
import numpy as np
import pandas as pd


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FEATURES_DIR = os.path.join(BASE_DIR, "features")

os.makedirs(FEATURES_DIR, exist_ok=True)


def engineer_features(df):
    df = df.copy()

    # Preserve original row order for provenance
    df["_source_order"] = np.arange(len(df))

    # Order each MaterialID trajectory using verified time field.
    # Source order is used only to break equal-time ties.
    df = df.sort_values(
        ["MaterialID", "duration_ms", "_source_order"]
    ).reset_index(drop=True)

    raw_features = [
        col for col in df.columns
        if col.startswith("feature_")
        and not col.endswith("_delta")
        and not col.endswith("_slope")
        and not col.endswith("_rolling_std")
        and not col.endswith("_spike")
    ]

    engineered = []

    for feature in raw_features:

        # 1. Level
        engineered.append(
            df[feature].rename(feature)
        )

        # Previous value within the same MaterialID
        previous = df.groupby("MaterialID")[feature].shift(1)

        # 2. Delta / change
        delta = df[feature] - previous
        engineered.append(
            delta.rename(f"{feature}_delta")
        )

        # Actual elapsed time
        previous_time = df.groupby("MaterialID")["duration_ms"].shift(1)
        dt = df["duration_ms"] - previous_time

        # Avoid division by zero
        dt_safe = dt.where(dt > 0, np.nan)

        # 3. Slope / rate of change
        slope = delta / dt_safe
        engineered.append(
            slope.rename(f"{feature}_slope")
        )

        # 4. Rolling variability
        rolling_std = (
            df.groupby("MaterialID")[feature]
            .transform(
                lambda x: x.rolling(window=3, min_periods=2).std()
            )
        )

        engineered.append(
            rolling_std.rename(f"{feature}_rolling_std")
        )

        # 5. Spike / drop magnitude
        spike = delta.abs()

        engineered.append(
            spike.rename(f"{feature}_spike")
        )

    engineered_df = pd.concat(engineered, axis=1)

    # Keep metadata and evaluation label.
    metadata_columns = [
        col for col in [
            "MaterialID",
            "StepID",
            "duration_ms",
            "is_test",
            "target"
        ]
        if col in df.columns
    ]

    result = pd.concat(
        [
            df[metadata_columns].reset_index(drop=True),
            engineered_df.reset_index(drop=True)
        ],
        axis=1
    )

    return result


def process_dataset(dataset_name):
    split_names = [
        "train",
        "validation",
        "final_test"
    ]

    print("\n" + "=" * 60)
    print(f"PROCESSING {dataset_name}")
    print("=" * 60)

    for split in split_names:

        input_file = os.path.join(
            RESULTS_DIR,
            f"{dataset_name}_{split}.csv"
        )

        output_file = os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_{split}_engineered.csv"
        )

        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f"Missing file: {input_file}"
            )

        print(f"\n[{split.upper()}]")
        print(f"Reading: {input_file}")

        df = pd.read_csv(input_file)

        print(f"Raw shape: {df.shape}")
        print(f"MaterialID groups: {df['MaterialID'].nunique()}")

        result = engineer_features(df)

        # Check that every MaterialID remains inside this split
        print(
            f"Engineered shape: {result.shape}"
        )

        print(
            f"Missing values: "
            f"{result.isna().sum().sum()}"
        )

        result.to_csv(
            output_file,
            index=False
        )

        print(f"Saved: {output_file}")


if __name__ == "__main__":

    process_dataset("D1")
    process_dataset("D2")

    print("\n" + "=" * 60)
    print("STEP 8A COMPLETE")
    print("=" * 60)