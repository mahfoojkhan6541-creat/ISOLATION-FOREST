import os
import numpy as np
import pandas as pd


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

DATA_DIR = os.path.join(BASE_DIR, "data")
FEATURE_DIR = os.path.join(BASE_DIR, "features")


METADATA_COLUMNS = [
    "MaterialID",
    "StepID",
    "duration_ms",
    "is_test",
    "target"
]


def engineer_features(df):

    df = df.copy()

    # Preserve original source order
    df["_source_order"] = np.arange(len(df))

    # Sort only within MaterialID using available time field
    df = df.sort_values(
        ["MaterialID", "duration_ms", "_source_order"]
    )

    measurement_columns = [
        c for c in df.columns
        if c.startswith("feature_")
    ]

    feature_frames = []

    # ---------------------------------------------------------
    # Original measurement levels
    # ---------------------------------------------------------

    for col in measurement_columns:

        feature_frames.append(
            df[col].rename(col)
        )

    # ---------------------------------------------------------
    # Trajectory features
    # ---------------------------------------------------------

    grouped = df.groupby(
        "MaterialID",
        sort=False
    )

    for col in measurement_columns:

        previous = grouped[col].shift(1)

        delta = (
            df[col] - previous
        )

        dt = (
            df.groupby("MaterialID")["duration_ms"]
            .diff()
        )

        # Only use positive elapsed time
        valid_dt = dt.where(dt > 0)

        slope = (
            delta / valid_dt
        )

        rolling_std = (
            grouped[col]
            .rolling(
                window=3,
                min_periods=2
            )
            .std()
            .reset_index(
                level=0,
                drop=True
            )
        )

        spike = delta.abs()

        slope_change = (
            slope -
            grouped[col]
            .apply(
                lambda x: (
                    x.diff() /
                    df.loc[x.index, "duration_ms"].diff()
                )
            ).reset_index(
                level=0,
                drop=True
            )
        )

        abs_delta = delta.abs()

        stabilization = (
            abs_delta
            .groupby(df["MaterialID"])
            .rolling(
                window=3,
                min_periods=1
            )
            .mean()
            .reset_index(
                level=0,
                drop=True
            )
        )

        first_value = (
            grouped[col]
            .transform("first")
        )

        trajectory_change = (
            df[col] - first_value
        )

        feature_frames.append(
            delta.rename(
                f"{col}_delta"
            )
        )

        feature_frames.append(
            slope.rename(
                f"{col}_slope"
            )
        )

        feature_frames.append(
            rolling_std.rename(
                f"{col}_rolling_std"
            )
        )

        feature_frames.append(
            spike.rename(
                f"{col}_spike"
            )
        )

        feature_frames.append(
            slope_change.rename(
                f"{col}_slope_change"
            )
        )

        feature_frames.append(
            stabilization.rename(
                f"{col}_stabilization"
            )
        )

        feature_frames.append(
            trajectory_change.rename(
                f"{col}_trajectory_change"
            )
        )

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    engineered = pd.concat(
        [
            df[METADATA_COLUMNS],
            *feature_frames
        ],
        axis=1
    )

    engineered = engineered.reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # Restore source order
    # ---------------------------------------------------------

    source_order = (
        df["_source_order"]
        .reset_index(drop=True)
    )

    engineered["_source_order"] = source_order

    engineered = engineered.sort_values(
        "_source_order"
    )

    engineered = engineered.drop(
        columns="_source_order"
    )

    engineered = engineered.reset_index(
        drop=True
    )

    return engineered


def process_dataset(input_file, output_file):

    print("\n" + "=" * 70)
    print(f"PROCESSING {input_file}")
    print("=" * 70)

    df = pd.read_csv(
        input_file
    )

    print(
        f"Input shape: {df.shape}"
    )

    result = engineer_features(
        df
    )

    print(
        f"Output shape: {result.shape}"
    )

    feature_columns = [
        c for c in result.columns
        if c not in METADATA_COLUMNS
    ]

    print(
        f"Generated model features: "
        f"{len(feature_columns)}"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved:\n{output_file}"
    )


def main():

    process_dataset(
        os.path.join(
            DATA_DIR,
            "D2.csv"
        ),
        os.path.join(
            FEATURE_DIR,
            "D2_v3_engineered.csv"
        )
    )

    print("\n" + "=" * 70)
    print("STEP 32 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()