import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/D2.csv"
FEATURE_CONFIG = "models/D2_enhanced_feature_config.pkl"

TRAIN_FILE = "results/D2_train.csv"
VALIDATION_FILE = "results/D2_validation.csv"

OUTPUT_FILE = "results/D2_v2_feature_ablation.csv"

RANDOM_STATE = 42
N_ESTIMATORS = 300


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 44 - V2 FEATURE ABLATION EXPERIMENT")
print("=" * 70)

print("\nLoading data...")

df = pd.read_csv(DATA_FILE)

train_split = pd.read_csv(TRAIN_FILE)
validation_split = pd.read_csv(VALIDATION_FILE)

config = joblib.load(FEATURE_CONFIG)


# ============================================================
# IDENTIFY TRAIN / VALIDATION MATERIAL IDs
# ============================================================

train_ids = set(train_split["MaterialID"].unique())
validation_ids = set(validation_split["MaterialID"].unique())

train_df = df[df["MaterialID"].isin(train_ids)].copy()
validation_df = df[df["MaterialID"].isin(validation_ids)].copy()

print("\nTrain groups      :", train_df["MaterialID"].nunique())
print("Validation groups :", validation_df["MaterialID"].nunique())

print("Train rows        :", len(train_df))
print("Validation rows   :", len(validation_df))


# ============================================================
# MEASUREMENT COLUMNS
# ============================================================

measurement_cols = [
    f"feature_{i}"
    for i in range(1, 21)
]


# ============================================================
# EXACT V2 FEATURE GENERATION
# ============================================================

def create_v2_features(data):

    data = data.copy()

    # Preserve original source order
    data["_source_order"] = np.arange(len(data))

    # Exact trajectory ordering used by V2
    data = data.sort_values(
        ["MaterialID", "duration_ms", "_source_order"]
    ).copy()

    grouped = data.groupby("MaterialID", sort=False)

    result = pd.DataFrame(index=data.index)

    # --------------------------------------------------------
    # LEVEL
    # --------------------------------------------------------

    for col in measurement_cols:
        result[col] = data[col]

    # --------------------------------------------------------
    # TIME-DERIVED FEATURES
    # --------------------------------------------------------

    for col in measurement_cols:

        delta = grouped[col].diff()

        dt = grouped["duration_ms"].diff()

        # Delta
        result[f"{col}_delta"] = delta

        # Slope
        slope = delta / dt.replace(0, np.nan)
        result[f"{col}_slope"] = slope

        # Rolling standard deviation
        result[f"{col}_rolling_std"] = (
            grouped[col]
            .rolling(window=3, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )

        # Spike magnitude
        result[f"{col}_spike"] = delta.abs()

        # Acceleration / slope change
        previous_slope = slope.groupby(
            data["MaterialID"]
        ).shift(1)

        result[f"{col}_acceleration"] = (
            slope - previous_slope
        )

        # Stabilization
        abs_delta = delta.abs()

        result[f"{col}_stabilization"] = (
            abs_delta.groupby(
                data["MaterialID"]
            )
            .rolling(window=3, min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Trajectory change from first observed value
        first_value = grouped[col].transform("first")

        result[f"{col}_trajectory_change"] = (
            data[col] - first_value
        )

    # --------------------------------------------------------
    # CROSS-PARAMETER FEATURES
    # --------------------------------------------------------

    measurement_mean = data[measurement_cols].mean(axis=1)
    measurement_std = data[measurement_cols].std(axis=1)

    result["measurement_mean"] = measurement_mean
    result["measurement_std"] = measurement_std

    for col in measurement_cols:
        result[f"{col}_peer_deviation"] = (
            data[col] - measurement_mean
        )

    # Restore original row order
    result = result.loc[data.index]

    return result


print("\nGenerating V2 features...")

X_train_all = create_v2_features(train_df)
X_validation_all = create_v2_features(validation_df)

print("Generated train features     :", X_train_all.shape)
print("Generated validation features:", X_validation_all.shape)


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

removed_constants = set(
    config["removed_constant_features"]
)

all_model_features = [
    feature
    for feature in config["model_features"]
    if feature not in removed_constants
]


# ============================================================
# VERIFY FEATURE MATCH
# ============================================================

missing_train = [
    f for f in all_model_features
    if f not in X_train_all.columns
]

missing_validation = [
    f for f in all_model_features
    if f not in X_validation_all.columns
]

if missing_train or missing_validation:
    print("\nMissing features detected!")

    if missing_train:
        print("Missing in train:")
        print(missing_train)

    if missing_validation:
        print("Missing in validation:")
        print(missing_validation)

    raise ValueError(
        "Generated V2 features do not match the saved V2 configuration."
    )


X_train_all = X_train_all[all_model_features]
X_validation_all = X_validation_all[all_model_features]


# ============================================================
# TARGETS
# ============================================================

y_train = train_df["target"].values
y_validation = validation_df["target"].values


# ============================================================
# FEATURE FAMILY MAPPING
# ============================================================

def get_family(feature):

    if feature == "measurement_mean":
        return "measurement_mean"

    if feature == "measurement_std":
        return "measurement_std"

    if feature.endswith("_peer_deviation"):
        return "peer_deviation"

    if feature.endswith("_trajectory_change"):
        return "trajectory_change"

    if feature.endswith("_stabilization"):
        return "stabilization"

    if feature.endswith("_acceleration"):
        return "acceleration"

    if feature.endswith("_rolling_std"):
        return "variability"

    if feature.endswith("_spike"):
        return "spike"

    if feature.endswith("_delta"):
        return "delta"

    if feature.endswith("_slope"):
        return "slope"

    return "level"


feature_families = {}

for feature in all_model_features:
    family = get_family(feature)
    feature_families.setdefault(family, []).append(feature)


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

experiments = {

    "all_features": [
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
    ],

    "without_delta": [
        "level",
        "slope",
        "variability",
        "spike",
        "acceleration",
        "stabilization",
        "trajectory_change",
        "measurement_mean",
        "measurement_std",
        "peer_deviation",
    ],

    "without_slope": [
        "level",
        "delta",
        "variability",
        "spike",
        "acceleration",
        "stabilization",
        "trajectory_change",
        "measurement_mean",
        "measurement_std",
        "peer_deviation",
    ],

    "without_variability": [
        "level",
        "delta",
        "slope",
        "spike",
        "acceleration",
        "stabilization",
        "trajectory_change",
        "measurement_mean",
        "measurement_std",
        "peer_deviation",
    ],

    "without_spike": [
        "level",
        "delta",
        "slope",
        "variability",
        "acceleration",
        "stabilization",
        "trajectory_change",
        "measurement_mean",
        "measurement_std",
        "peer_deviation",
    ],

    "without_acceleration": [
        "level",
        "delta",
        "slope",
        "variability",
        "spike",
        "stabilization",
        "trajectory_change",
        "measurement_mean",
        "measurement_std",
        "peer_deviation",
    ],

    "without_stabilization": [
        "level",
        "delta",
        "slope",
        "variability",
        "spike",
        "acceleration",
        "trajectory_change",
        "measurement_mean",
        "measurement_std",
        "peer_deviation",
    ],

    "without_trajectory": [
        "level",
        "delta",
        "slope",
        "variability",
        "spike",
        "acceleration",
        "stabilization",
        "measurement_mean",
        "measurement_std",
        "peer_deviation",
    ],

    "without_peer_deviation": [
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
    ],

    "level_only": [
        "level"
    ],

    "level_plus_peer": [
        "level",
        "peer_deviation"
    ],

    "level_plus_mean_std": [
        "level",
        "measurement_mean",
        "measurement_std"
    ],

    "level_peer_mean_std": [
        "level",
        "peer_deviation",
        "measurement_mean",
        "measurement_std"
    ],
}


# ============================================================
# MATERIAL LEVEL EVALUATION
# ============================================================

def evaluate_material_level(
    train_features,
    validation_features,
    experiment_name
):

    selected_features = []

    for family in experiments[experiment_name]:

        selected_features.extend(
            feature_families.get(family, [])
        )

    # Remove duplicate names while preserving order
    selected_features = list(
        dict.fromkeys(selected_features)
    )

    Xtr = train_features[selected_features].copy()
    Xva = validation_features[selected_features].copy()

    # --------------------------------------------------------
    # TRAIN-ONLY PREPROCESSING
    # --------------------------------------------------------

    imputer = SimpleImputer(strategy="median")

    Xtr = imputer.fit_transform(Xtr)
    Xva = imputer.transform(Xva)

    scaler = StandardScaler()

    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)

    # --------------------------------------------------------
    # TRAIN ISOLATION FOREST
    # --------------------------------------------------------

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(Xtr)

    # Higher score = more abnormal
    validation_scores = -model.score_samples(Xva)

    # --------------------------------------------------------
    # MATERIALID AGGREGATION
    # --------------------------------------------------------

    validation_result = pd.DataFrame({
        "MaterialID": validation_df["MaterialID"].values,
        "target": y_validation,
        "score": validation_scores
    })

    grouped = (
        validation_result
        .groupby("MaterialID")
        .agg(
            target=("target", "first"),
            mean_score=("score", "mean")
        )
        .reset_index()
    )

    ap = average_precision_score(
        grouped["target"],
        grouped["mean_score"]
    )

    return {
        "experiment": experiment_name,
        "feature_count": len(selected_features),
        "material_level_AP": ap
    }


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []

print("\n" + "=" * 70)
print("RUNNING ABLATION EXPERIMENTS")
print("=" * 70)

for experiment_name in experiments:

    print(
        f"\nRunning: {experiment_name}"
    )

    result = evaluate_material_level(
        X_train_all,
        X_validation_all,
        experiment_name
    )

    results.append(result)

    print(
        f"Features = {result['feature_count']:3d} | "
        f"MaterialID AP = {result['material_level_AP']:.6f}"
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "material_level_AP",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 70)
print("V2 FEATURE ABLATION RESULTS")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# ============================================================
# COMPARE WITH CURRENT V2
# ============================================================

current_v2_ap = 0.817722

best_ap = results_df.iloc[0]["material_level_AP"]
best_experiment = results_df.iloc[0]["experiment"]

print("\n" + "=" * 70)
print("COMPARISON WITH CURRENT V2")
print("=" * 70)

print(f"Current V2 AP : {current_v2_ap:.6f}")
print(f"Best new AP   : {best_ap:.6f}")
print(f"Best experiment: {best_experiment}")

if best_ap > current_v2_ap:
    print("\nPotential improvement found.")
else:
    print("\nNo improvement over current V2.")


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
print("STEP 44 COMPLETED")
print("=" * 70)