import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def inspect_trajectories(file_path, dataset_name):

    print("\n" + "=" * 70)
    print(f"{dataset_name} - TRAJECTORY INSPECTION")
    print("=" * 70)

    df = pd.read_csv(file_path)

    # Preserve original source order
    df["_source_order"] = range(len(df))

    # Sort only for inspection using duration_ms
    ordered = df.sort_values(
        ["MaterialID", "duration_ms", "_source_order"]
    ).copy()

    # ---------------------------------------------------------
    # 1. Rows per MaterialID
    # ---------------------------------------------------------
    rows_per_material = (
        ordered.groupby("MaterialID")
        .size()
    )

    print("\n[1] ROWS PER MATERIALID")
    print(
        rows_per_material.describe()
    )

    # ---------------------------------------------------------
    # 2. StepID count
    # ---------------------------------------------------------
    print("\n[2] UNIQUE STEPIDs PER MATERIALID")

    steps_per_material = (
        ordered.groupby("MaterialID")["StepID"]
        .nunique()
    )

    print(
        steps_per_material.value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # 3. Duration range
    # ---------------------------------------------------------
    print("\n[3] DURATION RANGE")

    duration_range = (
        ordered.groupby("MaterialID")["duration_ms"]
        .agg(["min", "max", "nunique"])
    )

    print(duration_range.describe())

    # ---------------------------------------------------------
    # 4. Duplicate timestamps
    # ---------------------------------------------------------
    print("\n[4] DUPLICATE TIMESTAMPS")

    duplicate_time = ordered.duplicated(
        subset=["MaterialID", "duration_ms"],
        keep=False
    )

    print(
        f"Rows involved in repeated timestamps: "
        f"{duplicate_time.sum():,}"
    )

    # ---------------------------------------------------------
    # 5. Source-order time reversals
    # ---------------------------------------------------------
    print("\n[5] SOURCE-ORDER TIME REVERSALS")

    ordered_by_source = df.sort_values(
        ["MaterialID", "_source_order"]
    ).copy()

    ordered_by_source["time_diff"] = (
        ordered_by_source
        .groupby("MaterialID")["duration_ms"]
        .diff()
    )

    negative_source_steps = (
        ordered_by_source["time_diff"] < 0
    ).sum()

    print(
        f"Negative duration differences in source order: "
        f"{negative_source_steps:,}"
    )

    # ---------------------------------------------------------
    # 6. Save summary
    # ---------------------------------------------------------
    summary = pd.DataFrame({
        "metric": [
            "total_rows",
            "unique_materials",
            "measurement_features",
            "unique_stepids",
            "duplicate_timestamp_rows",
            "negative_source_time_differences"
        ],
        "value": [
            len(df),
            df["MaterialID"].nunique(),
            len([
                c for c in df.columns
                if c.startswith("feature_")
            ]),
            df["StepID"].nunique(),
            int(duplicate_time.sum()),
            int(negative_source_steps)
        ]
    })

    summary.to_csv(
        RESULTS_DIR / f"{dataset_name}_trajectory_summary.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 7. Example trajectory plot
    # ---------------------------------------------------------
    example_material = (
        ordered["MaterialID"]
        .value_counts()
        .index[0]
    )

    example = ordered[
        ordered["MaterialID"] == example_material
    ]

    feature_columns = [
        c for c in df.columns
        if c.startswith("feature_")
    ]

    # Plot first feature only
    plt.figure(figsize=(10, 5))

    plt.plot(
        example["duration_ms"],
        example[feature_columns[0]]
    )

    plt.xlabel("duration_ms")
    plt.ylabel(feature_columns[0])
    plt.title(
        f"{dataset_name} Example MaterialID "
        f"{example_material}"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        f"{dataset_name}_example_trajectory.png",
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    df.drop(columns=["_source_order"], inplace=True)

    print("\nSaved:")
    print(
        RESULTS_DIR /
        f"{dataset_name}_trajectory_summary.csv"
    )

    print(
        RESULTS_DIR /
        f"{dataset_name}_example_trajectory.png"
    )


# D1
inspect_trajectories(
    DATA_DIR / "D1.csv",
    "D1"
)

# D2
inspect_trajectories(
    DATA_DIR / "D2.csv",
    "D2"
)

print("\n" + "=" * 70)
print("TRAJECTORY INSPECTION COMPLETED")
print("=" * 70)