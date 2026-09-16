import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/D2.csv"

TRAIN_FILE = "results/D2_train.csv"
VALIDATION_FILE = "results/D2_validation.csv"

OUTPUT_FILE = "results/D2_material_feature_combinations.csv"

RANDOM_STATE = 42
N_ESTIMATORS = 300


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 42 - MATERIALID FEATURE COMBINATION EXPERIMENT")
print("=" * 70)

data = pd.read_csv(DATA_FILE)
train_split = pd.read_csv(TRAIN_FILE)
validation_split = pd.read_csv(VALIDATION_FILE)

train_ids = set(train_split["MaterialID"].unique())
validation_ids = set(validation_split["MaterialID"].unique())

print(f"Train groups      : {len(train_ids)}")
print(f"Validation groups : {len(validation_ids)}")


# ============================================================
# BASIC COLUMNS
# ============================================================

measurement_cols = [
    c for c in data.columns
    if c.startswith("feature_")
]

print(f"Measurement features: {len(measurement_cols)}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    df["_source_order"] = np.arange(len(df))

    df = df.sort_values(
        ["MaterialID", "duration_ms", "_source_order"]
    ).reset_index(drop=True)

    result = pd.DataFrame(index=df.index)

    # --------------------------------------------------------
    # LEVEL
    # --------------------------------------------------------

    for col in measurement_cols:
        result[col] = df[col]

    # --------------------------------------------------------
    # TEMPORAL FEATURES
    # --------------------------------------------------------

    grouped = df.groupby("MaterialID", sort=False)

    for col in measurement_cols:

        previous = grouped[col].shift(1)

        delta = df[col] - previous

        result[f"{col}_delta"] = delta

        dt = grouped["duration_ms"].diff()

        slope = delta / dt.replace(0, np.nan)

        slope = slope.where(dt > 0)

        result[f"{col}_slope"] = slope

        result[f"{col}_variability"] = (
            grouped[col]
            .rolling(window=3, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
            .sort_index()
        )

        result[f"{col}_spike"] = delta.abs()

        result[f"{col}_stabilization"] = (
            delta.abs()
            .groupby(df["MaterialID"])
            .rolling(window=3, min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
            .sort_index()
        )

        first_value = grouped[col].transform("first")

        result[f"{col}_trajectory"] = (
            df[col] - first_value
        )

    return result


# ============================================================
# SPLIT DATA
# ============================================================

train_df = data[data["MaterialID"].isin(train_ids)].copy()
validation_df = data[data["MaterialID"].isin(validation_ids)].copy()

print(f"Train rows        : {len(train_df)}")
print(f"Validation rows   : {len(validation_df)}")


# ============================================================
# CREATE FEATURES
# ============================================================

print("\nCreating train features...")
X_train_all = create_features(train_df)

print("Creating validation features...")
X_validation_all = create_features(validation_df)


# ============================================================
# TARGET
# ============================================================

y_train = train_df["target"].to_numpy()
y_validation = validation_df["target"].to_numpy()


# ============================================================
# FEATURE FAMILIES
# ============================================================

families = {
    "level": measurement_cols,

    "delta": [
        f"{c}_delta"
        for c in measurement_cols
    ],

    "slope": [
        f"{c}_slope"
        for c in measurement_cols
    ],

    "variability": [
        f"{c}_variability"
        for c in measurement_cols
    ],

    "spike": [
        f"{c}_spike"
        for c in measurement_cols
    ],

    "stabilization": [
        f"{c}_stabilization"
        for c in measurement_cols
    ],

    "trajectory": [
        f"{c}_trajectory"
        for c in measurement_cols
    ],
}


# ============================================================
# EXPERIMENTS
# ============================================================

experiments = {
    "level_only": ["level"],

    "level_delta": [
        "level",
        "delta"
    ],

    "level_spike": [
        "level",
        "spike"
    ],

    "level_stabilization": [
        "level",
        "stabilization"
    ],

    "level_slope": [
        "level",
        "slope"
    ],

    "level_variability": [
        "level",
        "variability"
    ],

    "level_trajectory": [
        "level",
        "trajectory"
    ],

    "level_delta_spike": [
        "level",
        "delta",
        "spike"
    ],

    "level_delta_stabilization": [
        "level",
        "delta",
        "stabilization"
    ],

    "level_delta_spike_stabilization": [
        "level",
        "delta",
        "spike",
        "stabilization"
    ],

    "level_delta_spike_variability": [
        "level",
        "delta",
        "spike",
        "variability"
    ],

    "level_delta_slope": [
        "level",
        "delta",
        "slope"
    ],

    "level_delta_trajectory": [
        "level",
        "delta",
        "trajectory"
    ],

    "level_spike_stabilization": [
        "level",
        "spike",
        "stabilization"
    ],

    "level_delta_spike_stabilization_variability": [
        "level",
        "delta",
        "spike",
        "stabilization",
        "variability"
    ],
}


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(feature_names):

    Xtr = X_train_all[feature_names].copy()
    Xva = X_validation_all[feature_names].copy()

    # Train-only preprocessing
    imputer = SimpleImputer(strategy="median")

    Xtr = imputer.fit_transform(Xtr)
    Xva = imputer.transform(Xva)

    scaler = StandardScaler()

    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)

    # Isolation Forest trained ONLY on training data
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples="auto",
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(Xtr)

    train_scores = -model.score_samples(Xtr)
    validation_scores = -model.score_samples(Xva)

    # --------------------------------------------------------
    # MaterialID-level aggregation
    # --------------------------------------------------------

    validation_result = pd.DataFrame({
        "MaterialID": validation_df["MaterialID"].to_numpy(),
        "target": y_validation,
        "score": validation_scores
    })

    material_scores = (
        validation_result
        .groupby("MaterialID")
        .agg(
            target=("target", "first"),
            mean_score=("score", "mean")
        )
        .reset_index()
    )

    ap = average_precision_score(
        material_scores["target"],
        material_scores["mean_score"]
    )

    return ap, len(feature_names)


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []

print("\n" + "=" * 70)
print("MATERIALID-LEVEL EXPERIMENT RESULTS")
print("=" * 70)

for experiment_name, selected_families in experiments.items():

    feature_names = []

    for family in selected_families:
        feature_names.extend(families[family])

    feature_names = list(dict.fromkeys(feature_names))

    ap, feature_count = evaluate(feature_names)

    results.append({
        "experiment": experiment_name,
        "feature_count": feature_count,
        "average_precision": ap
    })

    print(
        f"{experiment_name:45s} "
        f"features={feature_count:3d} "
        f"MaterialID AP={ap:.6f}"
    )


# ============================================================
# RANKING
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "average_precision",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("RANKING")
print("=" * 70)

print(results_df.to_string(index=False))


# ============================================================
# SAVE
# ============================================================

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("STEP 42 COMPLETED")
print("=" * 70)