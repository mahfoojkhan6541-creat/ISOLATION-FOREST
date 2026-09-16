import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def data_quality_check(file_path, dataset_name):

    print("\n" + "=" * 70)
    print(f"{dataset_name} - DATA QUALITY GATE")
    print("=" * 70)

    df = pd.read_csv(file_path)

    # ---------------------------------------------------------
    # 1. Basic information
    # ---------------------------------------------------------
    print("\n[1] BASIC DATA CHECK")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"MaterialID groups: {df['MaterialID'].nunique():,}")

    # ---------------------------------------------------------
    # 2. Missing values
    # ---------------------------------------------------------
    print("\n[2] MISSING VALUE CHECK")

    missing_total = df.isnull().sum().sum()

    print(f"Total missing values: {missing_total:,}")

    # ---------------------------------------------------------
    # 3. Duplicate rows
    # ---------------------------------------------------------
    print("\n[3] DUPLICATE CHECK")

    duplicate_rows = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_rows:,}")

    # ---------------------------------------------------------
    # 4. Infinite values
    # ---------------------------------------------------------
    print("\n[4] INFINITE VALUE CHECK")

    numeric_columns = df.select_dtypes(include=np.number).columns

    infinite_count = np.isinf(df[numeric_columns]).sum().sum()

    print(f"Infinite values: {infinite_count:,}")

    # ---------------------------------------------------------
    # 5. MaterialID validity
    # ---------------------------------------------------------
    print("\n[5] MATERIAL ID CHECK")

    invalid_material_ids = df["MaterialID"].isnull().sum()

    print(f"Missing MaterialID: {invalid_material_ids:,}")
    print(f"Unique MaterialID: {df['MaterialID'].nunique():,}")

    # ---------------------------------------------------------
    # 6. StepID validity
    # ---------------------------------------------------------
    print("\n[6] STEP ID CHECK")

    print("Unique StepID values:")
    print(sorted(df["StepID"].unique().tolist()))

    # ---------------------------------------------------------
    # 7. Time field check
    # ---------------------------------------------------------
    print("\n[7] TIME / DURATION CHECK")

    print(f"Missing duration_ms: {df['duration_ms'].isnull().sum():,}")
    print(
        f"Negative duration_ms: "
        f"{(df['duration_ms'] < 0).sum():,}"
    )

    # ---------------------------------------------------------
    # 8. Time ordering check
    # ---------------------------------------------------------
    print("\n[8] TRAJECTORY ORDER CHECK")

    # Preserve original file order
    df["_original_row"] = np.arange(len(df))

    # Compare consecutive rows within MaterialID
    df["time_difference"] = (
        df.groupby("MaterialID")["duration_ms"]
        .diff()
    )

    negative_time_steps = (
        df["time_difference"] < 0
    ).sum()

    print(
        f"Negative time differences in source row order: "
        f"{negative_time_steps:,}"
    )

    # ---------------------------------------------------------
    # 9. Duplicate timestamps within MaterialID
    # ---------------------------------------------------------
    print("\n[9] DUPLICATE TIMESTAMP CHECK")

    duplicate_timestamps = df.duplicated(
        subset=["MaterialID", "duration_ms"]
    ).sum()

    print(
        f"Repeated MaterialID + duration_ms rows: "
        f"{duplicate_timestamps:,}"
    )

    # ---------------------------------------------------------
    # 10. Target consistency within MaterialID
    # ---------------------------------------------------------
    print("\n[10] TARGET CONSISTENCY CHECK")

    target_per_material = (
        df.groupby("MaterialID")["target"]
        .nunique()
    )

    inconsistent_materials = (
        target_per_material > 1
    ).sum()

    print(
        f"MaterialID groups with multiple target values: "
        f"{inconsistent_materials:,}"
    )

    # ---------------------------------------------------------
    # 11. Final summary
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print(f"{dataset_name} DATA QUALITY SUMMARY")
    print("-" * 70)

    print(f"Missing values              : {missing_total:,}")
    print(f"Duplicate rows              : {duplicate_rows:,}")
    print(f"Infinite values             : {infinite_count:,}")
    print(f"Negative duration values    : {(df['duration_ms'] < 0).sum():,}")
    print(
        f"Negative time differences   : "
        f"{negative_time_steps:,}"
    )
    print(
        f"Duplicate timestamps       : "
        f"{duplicate_timestamps:,}"
    )
    print(
        f"Target-inconsistent groups : "
        f"{inconsistent_materials:,}"
    )

    # Remove temporary columns
    df.drop(
        columns=["_original_row", "time_difference"],
        inplace=True
    )

    return df


# D1
d1 = data_quality_check(
    DATA_DIR / "D1.csv",
    "D1"
)

# D2
d2 = data_quality_check(
    DATA_DIR / "D2.csv",
    "D2"
)

print("\n" + "=" * 70)
print("DATA QUALITY GATE CHECK COMPLETED")
print("=" * 70)