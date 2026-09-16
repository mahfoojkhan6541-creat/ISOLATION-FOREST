import os
import joblib
import pandas as pd


CONFIG_FILE = "models/D2_enhanced_feature_config.pkl"
OUTPUT_FILE = "results/D2_v2_feature_audit.csv"


print("=" * 70)
print("STEP 43 - EXACT V2 FEATURE FAMILY AUDIT")
print("=" * 70)


# ============================================================
# LOAD EXACT V2 CONFIGURATION
# ============================================================

if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(
        f"Configuration file not found: {CONFIG_FILE}"
    )

config = joblib.load(CONFIG_FILE)

print("\nConfiguration loaded:")
print(CONFIG_FILE)


# ============================================================
# READ MODEL FEATURES
# ============================================================

model_features = config["model_features"]
removed_features = config["removed_constant_features"]

print("\nCandidate features :", len(config["candidate_features"]))
print("Removed constants  :", len(removed_features))
print("Model features     :", len(model_features))


# ============================================================
# FEATURE FAMILY CLASSIFICATION
# ============================================================

def classify_feature(name):

    if name.startswith("feature_"):

        if "_trajectory_change" in name:
            return "trajectory_change"

        if "_stabilization" in name:
            return "stabilization"

        if "_acceleration" in name:
            return "acceleration"

        if "_rolling_std" in name:
            return "variability"

        if "_peer_deviation" in name:
            return "peer_deviation"

        if "_delta" in name:
            return "delta"

        if "_slope" in name:
            return "slope"

        if "_spike" in name:
            return "spike"

        return "level"

    if name == "measurement_mean":
        return "measurement_mean"

    if name == "measurement_std":
        return "measurement_std"

    return "other"


# ============================================================
# CREATE AUDIT DATAFRAME
# ============================================================

audit = pd.DataFrame({
    "feature_name": model_features
})

audit["family"] = audit["feature_name"].apply(
    classify_feature
)


# ============================================================
# FAMILY COUNTS
# ============================================================

family_order = [
    "level",
    "delta",
    "slope",
    "variability",
    "spike",
    "acceleration",
    "stabilization",
    "trajectory_change",
    "measurement_mean",
    "measurement_std",
    "peer_deviation",
    "other"
]


family_counts = (
    audit["family"]
    .value_counts()
    .reindex(family_order, fill_value=0)
    .rename_axis("family")
    .reset_index(name="feature_count")
)


print("\n" + "=" * 70)
print("EXACT V2 FEATURE FAMILY COUNTS")
print("=" * 70)

print(
    family_counts.to_string(index=False)
)


# ============================================================
# REMOVED CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("REMOVED CONSTANT FEATURES")
print("=" * 70)

for feature in removed_features:
    print(feature)


# ============================================================
# MODEL FEATURES BY FAMILY
# ============================================================

print("\n" + "=" * 70)
print("MODEL FEATURES BY FAMILY")
print("=" * 70)

for family in family_order:

    names = audit.loc[
        audit["family"] == family,
        "feature_name"
    ].tolist()

    if not names:
        continue

    print(f"\n--- {family.upper()} ({len(names)}) ---")

    for name in names:
        print(name)


# ============================================================
# CHECK
# ============================================================

total_from_families = family_counts["feature_count"].sum()

print("\n" + "=" * 70)
print("FEATURE COUNT CHECK")
print("=" * 70)

print("Model features              :", len(model_features))
print("Family-count total          :", total_from_families)

if total_from_families == len(model_features):
    print("CHECK: PASS")
else:
    print("CHECK: FAILED")


# ============================================================
# SAVE
# ============================================================

os.makedirs("results", exist_ok=True)

audit.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)


print("\n" + "=" * 70)
print("STEP 43 COMPLETED")
print("=" * 70)