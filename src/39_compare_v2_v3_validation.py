import pandas as pd
from sklearn.metrics import average_precision_score

# ============================================================
# STEP 39: V2 vs V3 MATERIALID VALIDATION COMPARISON
# ============================================================

V2_FILE = "results/D2_material_level_validation_scores.csv"
V3_FILE = "results/D2_v3_material_level_validation_scores.csv"

print("=" * 70)
print("V2 vs V3 MATERIALID VALIDATION COMPARISON")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load existing MaterialID-level validation results
# ------------------------------------------------------------

v2 = pd.read_csv(V2_FILE)
v3 = pd.read_csv(V3_FILE)

print("\nV2 validation groups:", len(v2))
print("V3 validation groups:", len(v3))

# ------------------------------------------------------------
# 2. Check MaterialID sets
# ------------------------------------------------------------

v2_ids = set(v2["MaterialID"])
v3_ids = set(v3["MaterialID"])

if v2_ids != v3_ids:
    raise ValueError("V2 and V3 validation MaterialID groups do not match.")

print("Same validation groups: YES")

# ------------------------------------------------------------
# 3. Merge
# ------------------------------------------------------------

comparison = v2[
    ["MaterialID", "target", "mean_score", "median_score",
     "max_score", "p95_score", "top10_mean_score"]
].copy()

comparison = comparison.rename(
    columns={
        "mean_score": "v2_mean_score",
        "median_score": "v2_median_score",
        "max_score": "v2_max_score",
        "p95_score": "v2_p95_score",
        "top10_mean_score": "v2_top10_mean_score"
    }
)

comparison = comparison.merge(
    v3[
        ["MaterialID", "target", "mean_score", "median_score",
         "max_score", "p95_score", "top10_mean_score"]
    ],
    on="MaterialID",
    suffixes=("_v2", "_v3")
)

# ------------------------------------------------------------
# 4. Verify target labels
# ------------------------------------------------------------

if not (comparison["target_v2"] == comparison["target_v3"]).all():
    raise ValueError("V2 and V3 target labels do not match.")

comparison["target"] = comparison["target_v2"].astype(int)

# ------------------------------------------------------------
# 5. Average Precision for every aggregation
# ------------------------------------------------------------

aggregations = {
    "mean_score": ("v2_mean_score", "mean_score"),
    "median_score": ("v2_median_score", "median_score"),
    "max_score": ("v2_max_score", "max_score"),
    "p95_score": ("v2_p95_score", "p95_score"),
    "top10_mean_score": ("v2_top10_mean_score", "top10_mean_score")
}

print("\n" + "=" * 70)
print("AVERAGE PRECISION COMPARISON")
print("=" * 70)

for name, (v2_col, v3_col) in aggregations.items():

    ap_v2 = average_precision_score(
        comparison["target"],
        comparison[v2_col]
    )

    ap_v3 = average_precision_score(
        comparison["target"],
        comparison[v3_col]
    )

    print(f"\n{name}")
    print(f"V2 AP : {ap_v2:.6f}")
    print(f"V3 AP : {ap_v3:.6f}")

# ------------------------------------------------------------
# 6. Compare the important mean_score aggregation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MEAN SCORE — CLASS SEPARATION")
print("=" * 70)

for version, column in [
    ("V2", "v2_mean_score"),
    ("V3", "mean_score")
]:

    normal = comparison.loc[
        comparison["target"] == 0, column
    ]

    abnormal = comparison.loc[
        comparison["target"] == 1, column
    ]

    print(f"\n{version}")
    print(f"Normal mean     : {normal.mean():.6f}")
    print(f"Abnormal mean   : {abnormal.mean():.6f}")
    print(f"Normal median   : {normal.median():.6f}")
    print(f"Abnormal median : {abnormal.median():.6f}")
    print(
        f"Separation      : "
        f"{abnormal.mean() - normal.mean():.6f}"
    )

# ------------------------------------------------------------
# 7. Rank comparison using mean_score
# ------------------------------------------------------------

comparison["v2_rank"] = comparison["v2_mean_score"].rank(
    ascending=False,
    method="min"
)

comparison["v3_rank"] = comparison["mean_score"].rank(
    ascending=False,
    method="min"
)

top_n = min(20, len(comparison))

v2_top = set(
    comparison.nlargest(
        top_n, "v2_mean_score"
    )["MaterialID"]
)

v3_top = set(
    comparison.nlargest(
        top_n, "mean_score"
    )["MaterialID"]
)

overlap = len(v2_top.intersection(v3_top))

print("\n" + "=" * 70)
print("RANKING COMPARISON")
print("=" * 70)

print(f"\nTop {top_n} MaterialID overlap: {overlap}/{top_n}")

# ------------------------------------------------------------
# 8. Save comparison
# ------------------------------------------------------------

OUTPUT = "results/D2_v2_v3_material_level_comparison.csv"

comparison.to_csv(OUTPUT, index=False)

print("\nSaved:")
print(OUTPUT)

print("\n" + "=" * 70)
print("STEP 39 COMPLETED")
print("=" * 70)