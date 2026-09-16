import os
import numpy as np
import pandas as pd


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"
FEATURES_DIR = os.path.join(BASE_DIR, "features")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


METADATA_COLUMNS = [
    "MaterialID",
    "StepID",
    "duration_ms",
    "is_test",
    "target"
]


def check_features(dataset_name, split):

    file_path = os.path.join(
        FEATURES_DIR,
        f"{dataset_name}_{split}_enhanced.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path)

    feature_columns = [
        col for col in df.columns
        if col not in METADATA_COLUMNS
    ]

    X = df[feature_columns]

    missing = X.isna().sum()
    infinite = np.isinf(X).sum()
    constant = X.nunique(dropna=True) <= 1

    summary = pd.DataFrame({
        "feature": feature_columns,
        "missing_count": missing.values,
        "missing_percent": (
            missing.values / len(df) * 100
        ),
        "infinite_count": infinite.values,
        "unique_values": X.nunique(
            dropna=True
        ).values,
        "constant": constant.values
    })

    summary_file = os.path.join(
        RESULTS_DIR,
        f"{dataset_name}_{split}_enhanced_quality.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\n" + "=" * 70)
    print(f"{dataset_name} - {split.upper()}")
    print("=" * 70)

    print("Rows:", len(df))
    print("Total features:", len(feature_columns))

    print(
        "Total missing values:",
        int(missing.sum())
    )

    print(
        "Total infinite values:",
        int(infinite.sum())
    )

    print(
        "Constant features:",
        int(constant.sum())
    )

    high_missing = summary[
        summary["missing_percent"] > 50
    ]

    print(
        "Features with >50% missing:",
        len(high_missing)
    )

    if constant.sum() > 0:
        print("\nConstant features:")
        print(
            summary.loc[
                summary["constant"],
                "feature"
            ].tolist()
        )

    if infinite.sum() > 0:
        print("\nFeatures containing infinity:")
        print(
            summary.loc[
                summary["infinite_count"] > 0,
                [
                    "feature",
                    "infinite_count"
                ]
            ].to_string(index=False)
        )

    print(
        "\nSaved:",
        summary_file
    )

    return summary


if __name__ == "__main__":

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_summaries = []

    for dataset in ["D1", "D2"]:

        for split in [
            "train",
            "validation",
            "final_test"
        ]:

            summary = check_features(
                dataset,
                split
            )

            summary["dataset"] = dataset
            summary["split"] = split

            all_summaries.append(summary)

    combined = pd.concat(
        all_summaries,
        ignore_index=True
    )

    combined_file = os.path.join(
        RESULTS_DIR,
        "enhanced_feature_quality_summary.csv"
    )

    combined.to_csv(
        combined_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("STEP 15 COMPLETE")
    print("=" * 70)

    print(
        "Combined report:",
        combined_file
    )