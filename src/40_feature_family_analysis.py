import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

# ============================================================
# STEP 40: FEATURE FAMILY ANALYSIS
# ============================================================

DATA_FILE = "data/D2.csv"

print("=" * 70)
print("STEP 40 - FEATURE FAMILY ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load D2
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

measurement_cols = [
    c for c in df.columns
    if c.startswith("feature_")
]

print("\nMeasurement features:", len(measurement_cols))

# ------------------------------------------------------------
# 2. Use the SAME validation MaterialIDs
# ------------------------------------------------------------

v2_groups = pd.read_csv(
    "results/D2_material_level_validation_scores.csv"
)

validation_ids = set(v2_groups["MaterialID"])

df = df[df["MaterialID"].isin(validation_ids)].copy()

print("Validation groups:", df["MaterialID"].nunique())
print("Validation rows:", len(df))

# ------------------------------------------------------------
# 3. Sort trajectories
# ------------------------------------------------------------

df["_source_order"] = np.arange(len(df))

df = df.sort_values(
    ["MaterialID", "duration_ms", "_source_order"]
).reset_index(drop=True)

grouped = df.groupby("MaterialID", sort=False)

# ------------------------------------------------------------
# 4. Build feature families
# ------------------------------------------------------------

families = {}

# ---------- LEVEL ----------

level_features = measurement_cols.copy()

families["level"] = level_features

# ---------- DELTA ----------

delta_features = []

for col in measurement_cols:
    name = f"{col}_delta"

    df[name] = grouped[col].diff()

    delta_features.append(name)

families["delta"] = delta_features

# ---------- SLOPE ----------

slope_features = []

for col in measurement_cols:
    delta = df[col].groupby(df["MaterialID"]).diff()
    dt = df["duration_ms"].groupby(df["MaterialID"]).diff()

    name = f"{col}_slope"

    df[name] = delta / dt.replace(0, np.nan)

    slope_features.append(name)

families["slope"] = slope_features

# ---------- ROLLING VARIABILITY ----------

rolling_features = []

for col in measurement_cols:
    name = f"{col}_rolling_std"

    df[name] = (
        df.groupby("MaterialID")[col]
        .rolling(window=3, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )

    rolling_features.append(name)

families["variability"] = rolling_features

# ---------- SPIKE / DROP ----------

spike_features = []

for col in measurement_cols:
    name = f"{col}_spike"

    df[name] = df[f"{col}_delta"].abs()

    spike_features.append(name)

families["spike"] = spike_features

# ---------- STABILIZATION ----------

stabilization_features = []

for col in measurement_cols:
    name = f"{col}_stabilization"

    df[name] = (
        df.groupby("MaterialID")[f"{col}_delta"]
        .transform(
            lambda x: x.abs().rolling(
                window=3,
                min_periods=1
            ).mean()
        )
    )

    stabilization_features.append(name)

families["stabilization"] = stabilization_features

# ---------- TRAJECTORY CHANGE ----------

trajectory_features = []

for col in measurement_cols:
    first_value = grouped[col].transform("first")

    name = f"{col}_trajectory_change"

    df[name] = df[col] - first_value

    trajectory_features.append(name)

families["trajectory"] = trajectory_features

# ------------------------------------------------------------
# 5. Remove constant features
# ------------------------------------------------------------

for family in families:

    valid_features = []

    for feature in families[family]:

        if feature in df.columns:
            if df[feature].nunique(dropna=True) > 1:
                valid_features.append(feature)

    families[family] = valid_features

# ------------------------------------------------------------
# 6. Prepare target
# ------------------------------------------------------------

y = (
    df.groupby("MaterialID")["target"]
    .first()
)

# ------------------------------------------------------------
# 7. Function for evaluating a feature family
# ------------------------------------------------------------

def evaluate_features(feature_list, family_name):

    X = df[feature_list].copy()

    # Train-like reference:
    # Since this is feature-family analysis, use validation
    # only for ranking diagnostic, not for final model selection.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X)

    scores = -model.score_samples(X)

    temp = pd.DataFrame({
        "MaterialID": df["MaterialID"].values,
        "score": scores
    })

    temp = temp.groupby("MaterialID")["score"].mean()

    temp = temp.to_frame("score")
    temp["target"] = y

    ap = average_precision_score(
        temp["target"],
        temp["score"]
    )

    return ap


# ------------------------------------------------------------
# 8. Evaluate each family
# ------------------------------------------------------------

results = []

print("\n" + "=" * 70)
print("INDIVIDUAL FEATURE FAMILY RESULTS")
print("=" * 70)

for family_name, features in families.items():

    if len(features) == 0:
        print(f"\n{family_name}: NO FEATURES")
        continue

    ap = evaluate_features(
        features,
        family_name
    )

    results.append({
        "feature_family": family_name,
        "feature_count": len(features),
        "average_precision": ap
    })

    print(
        f"\n{family_name:<18}"
        f"features={len(features):3d} "
        f"AP={ap:.6f}"
    )

# ------------------------------------------------------------
# 9. Save results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "average_precision",
    ascending=False
)

OUTPUT = "results/D2_feature_family_analysis.csv"

results_df.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 70)
print("RANKING")
print("=" * 70)

print(results_df.to_string(index=False))

print("\nSaved:")
print(OUTPUT)

print("\n" + "=" * 70)
print("STEP 40 COMPLETED")
print("=" * 70)