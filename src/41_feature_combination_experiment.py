import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

# ============================================================
# STEP 41: CONTROLLED FEATURE COMBINATION EXPERIMENT
# ============================================================

DATA_FILE = "data/D2.csv"
TRAIN_IDS_FILE = "results/D2_train.csv"
VAL_IDS_FILE = "results/D2_validation.csv"

print("=" * 70)
print("STEP 41 - CONTROLLED FEATURE COMBINATION EXPERIMENT")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

measurement_cols = [
    c for c in df.columns
    if c.startswith("feature_")
]

# ------------------------------------------------------------
# 2. Get exact TRAIN and VALIDATION MaterialIDs
# ------------------------------------------------------------

train_file = pd.read_csv(TRAIN_IDS_FILE)
val_file = pd.read_csv(VAL_IDS_FILE)

train_ids = set(train_file["MaterialID"])
val_ids = set(val_file["MaterialID"])

train_df = df[df["MaterialID"].isin(train_ids)].copy()
val_df = df[df["MaterialID"].isin(val_ids)].copy()

print("\nTrain groups:", train_df["MaterialID"].nunique())
print("Validation groups:", val_df["MaterialID"].nunique())

print("Train rows:", len(train_df))
print("Validation rows:", len(val_df))

# ------------------------------------------------------------
# 3. Create feature families
# ------------------------------------------------------------

def create_features(data):

    data = data.copy()

    data["_source_order"] = np.arange(len(data))

    data = data.sort_values(
        ["MaterialID", "duration_ms", "_source_order"]
    ).reset_index(drop=True)

    grouped = data.groupby("MaterialID", sort=False)

    families = {}

    # LEVEL
    families["level"] = measurement_cols.copy()

    # DELTA
    delta_cols = []

    for col in measurement_cols:
        name = f"{col}_delta"
        data[name] = grouped[col].diff()
        delta_cols.append(name)

    families["delta"] = delta_cols

    # SLOPE
    slope_cols = []

    for col in measurement_cols:

        delta = data.groupby("MaterialID")[col].diff()
        dt = data.groupby("MaterialID")["duration_ms"].diff()

        name = f"{col}_slope"

        data[name] = delta / dt.replace(0, np.nan)

        slope_cols.append(name)

    families["slope"] = slope_cols

    # VARIABILITY
    variability_cols = []

    for col in measurement_cols:

        name = f"{col}_rolling_std"

        data[name] = (
            data.groupby("MaterialID")[col]
            .rolling(window=3, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )

        variability_cols.append(name)

    families["variability"] = variability_cols

    # SPIKE
    spike_cols = []

    for col in measurement_cols:

        name = f"{col}_spike"

        data[name] = data[f"{col}_delta"].abs()

        spike_cols.append(name)

    families["spike"] = spike_cols

    # STABILIZATION
    stabilization_cols = []

    for col in measurement_cols:

        name = f"{col}_stabilization"

        data[name] = (
            data.groupby("MaterialID")[f"{col}_delta"]
            .transform(
                lambda x: x.abs().rolling(
                    window=3,
                    min_periods=1
                ).mean()
            )
        )

        stabilization_cols.append(name)

    families["stabilization"] = stabilization_cols

    # TRAJECTORY
    trajectory_cols = []

    for col in measurement_cols:

        name = f"{col}_trajectory_change"

        first_value = data.groupby("MaterialID")[col].transform("first")

        data[name] = data[col] - first_value

        trajectory_cols.append(name)

    families["trajectory"] = trajectory_cols

    return data, families


train_df, families = create_features(train_df)
val_df, _ = create_features(val_df)

# ------------------------------------------------------------
# 4. Remove constant features using TRAIN ONLY
# ------------------------------------------------------------

for family in families:

    families[family] = [
        col for col in families[family]
        if col in train_df.columns
        and train_df[col].nunique(dropna=True) > 1
    ]

print("\nFeature counts:")

for family, cols in families.items():
    print(f"{family:<18}: {len(cols)}")

# ------------------------------------------------------------
# 5. Evaluation function
# ------------------------------------------------------------

def evaluate(feature_columns, name):

    X_train = train_df[feature_columns]
    X_val = val_df[feature_columns]

    y_val = (
        val_df.groupby("MaterialID")["target"]
        .first()
    )

    # --------------------------------------------------------
    # Train-only preprocessing
    # --------------------------------------------------------

    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train)

    # Higher score = more abnormal
    row_scores = -model.score_samples(X_val)

    score_df = pd.DataFrame({
        "MaterialID": val_df["MaterialID"].values,
        "score": row_scores
    })

    group_scores = (
        score_df
        .groupby("MaterialID")["score"]
        .mean()
    )

    result = pd.DataFrame({
        "score": group_scores,
        "target": y_val
    })

    ap = average_precision_score(
        result["target"],
        result["score"]
    )

    return ap


# ------------------------------------------------------------
# 6. Controlled combinations
# ------------------------------------------------------------

experiments = {
    "level_only": ["level"],

    "level_trajectory": [
        "level",
        "trajectory"
    ],

    "level_stabilization": [
        "level",
        "stabilization"
    ],

    "level_variability": [
        "level",
        "variability"
    ],

    "level_delta": [
        "level",
        "delta"
    ],

    "level_slope": [
        "level",
        "slope"
    ],

    "level_spike": [
        "level",
        "spike"
    ],

    "level_trajectory_stabilization": [
        "level",
        "trajectory",
        "stabilization"
    ],

    "level_trajectory_variability": [
        "level",
        "trajectory",
        "variability"
    ],

    "level_trajectory_stabilization_variability": [
        "level",
        "trajectory",
        "stabilization",
        "variability"
    ]
}

# ------------------------------------------------------------
# 7. Run experiments
# ------------------------------------------------------------

results = []

print("\n" + "=" * 70)
print("EXPERIMENT RESULTS")
print("=" * 70)

for experiment_name, selected_families in experiments.items():

    feature_columns = []

    for family in selected_families:
        feature_columns.extend(families[family])

    ap = evaluate(
        feature_columns,
        experiment_name
    )

    results.append({
        "experiment": experiment_name,
        "feature_count": len(feature_columns),
        "average_precision": ap
    })

    print(
        f"\n{experiment_name:<45}"
        f"features={len(feature_columns):3d} "
        f"AP={ap:.6f}"
    )

# ------------------------------------------------------------
# 8. Rank
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "average_precision",
    ascending=False
)

print("\n" + "=" * 70)
print("RANKING")
print("=" * 70)

print(results_df.to_string(index=False))

# ------------------------------------------------------------
# 9. Save
# ------------------------------------------------------------

OUTPUT = "results/D2_controlled_feature_combinations.csv"

results_df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)

print("\n" + "=" * 70)
print("STEP 41 COMPLETED")
print("=" * 70)