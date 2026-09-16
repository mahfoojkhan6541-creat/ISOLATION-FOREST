import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

D1_PATH = DATA_DIR / "D1.csv"
D2_PATH = DATA_DIR / "D2.csv"


def profile_dataset(file_path, dataset_name):
    print("\n" + "=" * 60)
    print(f"{dataset_name} DATASET PROFILING")
    print("=" * 60)

    df = pd.read_csv(file_path)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nInfinite Values:")
    numeric_df = df.select_dtypes(include="number")
    print(numeric_df.isin([float("inf"), float("-inf")]).sum().sum())

    print("\nUnique MaterialID:")
    print(df["MaterialID"].nunique())

    print("\nUnique StepID:")
    print(sorted(df["StepID"].unique()))

    print("\nTarget Distribution:")
    print(df["target"].value_counts().sort_index())

    print("\nTarget Percentage:")
    print(
        (df["target"].value_counts(normalize=True).sort_index() * 100)
        .round(4)
    )

    print("\nis_test Distribution:")
    print(df["is_test"].value_counts().sort_index())

    print("\nTarget vs is_test:")
    print(pd.crosstab(df["is_test"], df["target"]))

    feature_columns = [
        col for col in df.columns
        if col.startswith("feature_")
    ]

    print("\nMeasurement Features:")
    print(f"Number of features: {len(feature_columns)}")
    print(feature_columns)

    print("\nConstant Features:")
    constant_features = [
        col for col in feature_columns
        if df[col].nunique() <= 1
    ]

    if constant_features:
        print(constant_features)
    else:
        print("None")

    print("\nMaterialID Target Consistency:")
    target_per_material = df.groupby("MaterialID")["target"].nunique()

    inconsistent = (target_per_material > 1).sum()

    print(f"MaterialID groups with multiple target values: {inconsistent}")

    return df


# Profile D1
d1 = profile_dataset(D1_PATH, "D1")

# Profile D2
d2 = profile_dataset(D2_PATH, "D2")

print("\n" + "=" * 60)
print("DATASET PROFILING COMPLETED")
print("=" * 60)