import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_DIR = BASE_DIR / "features"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def check_feature_quality(file_path, dataset_name):

    print("\n" + "=" * 70)
    print(f"{dataset_name} - FEATURE QUALITY CHECK")
    print("=" * 70)

    df = pd.read_csv(file_path)

    feature_columns = [
        col for col in df.columns
        if col.startswith("feature_")
    ]

    feature_df = df[feature_columns].copy()

    # ---------------------------------------------------------
    # Replace infinite values temporarily for analysis
    # ---------------------------------------------------------
    feature_df = feature_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ---------------------------------------------------------
    # Feature statistics
    # ---------------------------------------------------------
    quality = pd.DataFrame(index=feature_columns)

    quality["missing_count"] = feature_df.isna().sum()

    quality["missing_percent"] = (
        feature_df.isna().mean() * 100
    )

    quality["valid_count"] = (
        feature_df.notna().sum()
    )

    quality["unique_values"] = (
        feature_df.nunique(dropna=True)
    )

    quality["mean"] = feature_df.mean()

    quality["std"] = feature_df.std()

    quality["min"] = feature_df.min()

    quality["max"] = feature_df.max()

    quality["infinite_count"] = (
        df[feature_columns]
        .isin([np.inf, -np.inf])
        .sum()
    )

    # ---------------------------------------------------------
    # Constant features
    # ---------------------------------------------------------
    constant_features = quality[
        quality["unique_values"] <= 1
    ].index.tolist()

    # ---------------------------------------------------------
    # High missing features
    # ---------------------------------------------------------
    high_missing_features = quality[
        quality["missing_percent"] > 50
    ].index.tolist()

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------
    print(f"\nRows: {len(df):,}")
    print(f"Engineered features: {len(feature_columns)}")

    print(
        f"\nTotal missing feature values: "
        f"{quality['missing_count'].sum():,}"
    )

    print(
        f"Total infinite feature values: "
        f"{quality['infinite_count'].sum():,}"
    )

    print(
        f"\nConstant features: "
        f"{len(constant_features)}"
    )

    if constant_features:
        print(constant_features)

    print(
        f"\nFeatures with >50% missing values: "
        f"{len(high_missing_features)}"
    )

    if high_missing_features:
        print(high_missing_features)

    # ---------------------------------------------------------
    # Display feature quality table
    # ---------------------------------------------------------
    print("\nFEATURE QUALITY TABLE")
    print("-" * 70)

    display_columns = [
        "missing_count",
        "missing_percent",
        "unique_values",
        "std",
        "min",
        "max"
    ]

    print(
        quality[display_columns]
        .round(4)
        .to_string()
    )

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------
    output_path = (
        RESULTS_DIR /
        f"{dataset_name}_feature_quality.csv"
    )

    quality.to_csv(output_path)

    print(f"\nSaved:")
    print(output_path)

    return quality


# D1
d1_quality = check_feature_quality(
    FEATURE_DIR / "D1_engineered.csv",
    "D1"
)

# D2
d2_quality = check_feature_quality(
    FEATURE_DIR / "D2_engineered.csv",
    "D2"
)

print("\n" + "=" * 70)
print("FEATURE QUALITY CHECK COMPLETED")
print("=" * 70)