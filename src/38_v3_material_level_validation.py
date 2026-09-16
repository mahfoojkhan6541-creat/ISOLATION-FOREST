import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix
)


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"
FEATURES_DIR = os.path.join(BASE_DIR, "features")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


print("=" * 70)
print("STEP 38 - V3 MATERIALID-LEVEL VALIDATION")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load V3 validation anomaly scores
# ---------------------------------------------------------

score_file = os.path.join(
    RESULTS_DIR,
    "D2_v3_validation_anomaly_scores.csv"
)

scores = pd.read_csv(score_file)


# ---------------------------------------------------------
# 2. Load original D2 dataset
# ---------------------------------------------------------

d2_file = os.path.join(
    BASE_DIR,
    "data",
    "D2.csv"
)

d2 = pd.read_csv(d2_file)


# ---------------------------------------------------------
# 3. Load EXACT existing V2 validation groups
# ---------------------------------------------------------
# This file contains the MaterialIDs used by the
# previous validated Train/Validation split.
# We reuse them exactly.
# ---------------------------------------------------------

existing_validation_file = os.path.join(
    RESULTS_DIR,
    "D2_material_level_validation_scores.csv"
)

existing_validation = pd.read_csv(
    existing_validation_file
)


if "MaterialID" not in existing_validation.columns:

    raise ValueError(
        "MaterialID column not found in existing validation file."
    )


validation_ids = set(
    existing_validation["MaterialID"]
)


print("\nExisting validation groups found:", len(validation_ids))


# ---------------------------------------------------------
# 4. Recover exact validation rows from D2
# ---------------------------------------------------------

validation_data = d2[
    d2["MaterialID"].isin(validation_ids)
].copy()


print("Validation rows from D2:", len(validation_data))
print("V3 validation score rows:", len(scores))


# ---------------------------------------------------------
# 5. Verify row count
# ---------------------------------------------------------

if len(validation_data) != len(scores):

    raise ValueError(
        "\nValidation row mismatch!\n"
        f"D2 validation rows : {len(validation_data)}\n"
        f"V3 score rows       : {len(scores)}\n"
        "\nThe V3 scores do not match the expected validation split."
    )


# ---------------------------------------------------------
# 6. Preserve exact row order
# ---------------------------------------------------------

validation_data = validation_data.reset_index(drop=True)
scores = scores.reset_index(drop=True)


scores["MaterialID"] = validation_data["MaterialID"].values
scores["target"] = validation_data["target"].values


# ---------------------------------------------------------
# 7. Verify target is constant within MaterialID
# ---------------------------------------------------------

target_counts = (
    scores
    .groupby("MaterialID")["target"]
    .nunique()
)


if not (target_counts == 1).all():

    raise ValueError(
        "\nTarget is not constant within every MaterialID."
    )


print(
    "\nTarget is constant within every MaterialID: YES"
)


# ---------------------------------------------------------
# 8. Aggregate anomaly scores by MaterialID
# ---------------------------------------------------------

grouped = (
    scores
    .groupby("MaterialID")
    .agg(
        mean_score=("anomaly_score", "mean"),

        median_score=("anomaly_score", "median"),

        max_score=("anomaly_score", "max"),

        p95_score=(
            "anomaly_score",
            lambda x: np.percentile(x, 95)
        ),

        top10_mean_score=(
            "anomaly_score",
            lambda x: x.nlargest(
                max(1, int(np.ceil(len(x) * 0.10)))
            ).mean()
        ),

        target=("target", "first")
    )
    .reset_index()
)


print("\nMaterialID-level validation groups:", len(grouped))

print("\nTarget distribution:")
print(
    grouped["target"]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------
# 9. Evaluate each aggregation
# ---------------------------------------------------------

aggregations = [
    "mean_score",
    "median_score",
    "max_score",
    "p95_score",
    "top10_mean_score"
]


all_results = []


for aggregation in aggregations:

    y_true = grouped["target"].values
    y_score = grouped[aggregation].values

    ap = average_precision_score(
        y_true,
        y_score
    )

    thresholds = np.unique(y_score)

    for threshold in thresholds:

        y_pred = (
            y_score >= threshold
        ).astype(int)

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                average="binary",
                zero_division=0
            )
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        )

        fnr = (
            fn / (fn + tp)
            if (fn + tp) > 0
            else 0
        )

        all_results.append({
            "aggregation": aggregation,
            "threshold": threshold,
            "AP": ap,
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "FPR": fpr,
            "FNR": fnr,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })


results = pd.DataFrame(all_results)


# ---------------------------------------------------------
# 10. Best F1 for every aggregation
# ---------------------------------------------------------

best_results = (
    results
    .sort_values(
        ["aggregation", "F1"],
        ascending=[True, False]
    )
    .groupby("aggregation")
    .head(1)
    .sort_values(
        "F1",
        ascending=False
    )
)


print("\n" + "=" * 70)
print("BEST VALIDATION RESULT PER AGGREGATION")
print("=" * 70)

print(
    best_results[
        [
            "aggregation",
            "threshold",
            "AP",
            "precision",
            "recall",
            "F1",
            "FPR",
            "FNR",
            "TN",
            "FP",
            "FN",
            "TP"
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# 11. Save complete results
# ---------------------------------------------------------

output_file = os.path.join(
    RESULTS_DIR,
    "D2_v3_material_level_validation.csv"
)

best_file = os.path.join(
    RESULTS_DIR,
    "D2_v3_material_level_validation_best.csv"
)

grouped_file = os.path.join(
    RESULTS_DIR,
    "D2_v3_material_level_validation_scores.csv"
)


results.to_csv(
    output_file,
    index=False
)

best_results.to_csv(
    best_file,
    index=False
)

grouped.to_csv(
    grouped_file,
    index=False
)


print("\nSaved:")
print(output_file)
print(best_file)
print(grouped_file)


print("\n" + "=" * 70)
print("STEP 38 COMPLETE")
print("=" * 70)

print("\nIMPORTANT:")
print("- Exact existing validation groups were reused.")
print("- Validation groups:", len(validation_ids))
print("- Final Test was NOT used.")
print("- No threshold has been frozen.")
print("- No contamination was selected from Final Test.")