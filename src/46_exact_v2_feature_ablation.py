import os
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

FEATURES_DIR = os.path.join(BASE_DIR, "features")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_family(feature):

    if feature.startswith("feature_"):
        if "_delta" in feature:
            return "delta"
        if "_slope" in feature:
            return "slope"
        if "_rolling_std" in feature:
            return "variability"
        if "_spike" in feature:
            return "spike"
        if "_acceleration" in feature:
            return "acceleration"
        if "_stabilization" in feature:
            return "stabilization"
        if "_trajectory_change" in feature:
            return "trajectory"
        if "_peer_deviation" in feature:
            return "peer_deviation"

        return "level"

    if feature == "measurement_mean":
        return "measurement_mean"

    if feature == "measurement_std":
        return "measurement_std"

    return "other"


def evaluate_configuration(
    name,
    feature_indices,
    feature_names,
    X_train,
    X_validation,
    y_validation,
    validation_groups
):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    Xtr = X_train[:, feature_indices]
    Xva = X_validation[:, feature_indices]

    imputer = SimpleImputer(strategy="median")

    Xtr = imputer.fit_transform(Xtr)
    Xva = imputer.transform(Xva)

    scaler = StandardScaler()

    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42,
        n_jobs=-1
    )

    model.fit(Xtr)

    row_scores = -model.score_samples(Xva)

    score_df = pd.DataFrame({
        "MaterialID": validation_groups,
        "target": y_validation,
        "score": row_scores
    })

    material_scores = (
        score_df
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

    print("Features:", len(feature_indices))
    print("MaterialID groups:", len(material_scores))
    print(f"Material-level AP: {ap:.6f}")

    return {
        "configuration": name,
        "feature_count": len(feature_indices),
        "material_groups": len(material_scores),
        "average_precision": ap
    }


def main():

    print("\n" + "=" * 70)
    print("STEP 46 - EXACT V2 FEATURE ABLATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load EXACT saved V2 feature arrays
    # ---------------------------------------------------------

    X_train = np.load(
        os.path.join(
            FEATURES_DIR,
            "D2_enhanced_X_train.npy"
        )
    )

    X_validation = np.load(
        os.path.join(
            FEATURES_DIR,
            "D2_enhanced_X_validation.npy"
        )
    )

    print("\nTRAIN shape:", X_train.shape)
    print("VALIDATION shape:", X_validation.shape)

    # ---------------------------------------------------------
    # Load exact V2 feature configuration
    # ---------------------------------------------------------

    import joblib

    config = joblib.load(
        os.path.join(
            BASE_DIR,
            "models",
            "D2_enhanced_feature_config.pkl"
        )
    )

    feature_names = config["model_features"]

    print("Exact V2 model features:", len(feature_names))

    assert X_train.shape[1] == len(feature_names)
    assert X_validation.shape[1] == len(feature_names)

    # ---------------------------------------------------------
    # Load validation metadata
    # ---------------------------------------------------------

    validation_source = pd.read_csv(
        os.path.join(
            FEATURES_DIR,
            "D2_validation_enhanced.csv"
        )
    )

    y_validation = validation_source["target"].to_numpy()

    validation_groups = validation_source["MaterialID"].to_numpy()

    assert len(y_validation) == len(X_validation)
    assert len(validation_groups) == len(X_validation)

    # ---------------------------------------------------------
    # Identify feature families
    # ---------------------------------------------------------

    families = {}

    for i, feature in enumerate(feature_names):

        family = get_family(feature)

        families.setdefault(
            family,
            []
        ).append(i)

    print("\nFeature family counts:")

    for family, indices in families.items():
        print(
            f"{family:20s}: {len(indices)}"
        )

    all_indices = np.arange(
        len(feature_names)
    )

    results = []

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------

    results.append(
        evaluate_configuration(
            "all_features",
            all_indices,
            feature_names,
            X_train,
            X_validation,
            y_validation,
            validation_groups
        )
    )

    # ---------------------------------------------------------
    # Remove one family at a time
    # ---------------------------------------------------------

    removable_families = [
        "delta",
        "slope",
        "variability",
        "spike",
        "acceleration",
        "stabilization",
        "trajectory",
        "peer_deviation"
    ]

    for family in removable_families:

        remove_indices = set(
            families.get(
                family,
                []
            )
        )

        keep_indices = [
            i
            for i in all_indices
            if i not in remove_indices
        ]

        results.append(
            evaluate_configuration(
                f"without_{family}",
                keep_indices,
                feature_names,
                X_train,
                X_validation,
                y_validation,
                validation_groups
            )
        )

    # ---------------------------------------------------------
    # Useful compact combinations
    # ---------------------------------------------------------

    def indices_for(names):

        selected = []

        for family in names:
            selected.extend(
                families.get(
                    family,
                    []
                )
            )

        return sorted(
            set(selected)
        )

    combinations = {
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
        ]
    }

    for name, selected_families in combinations.items():

        indices = indices_for(
            selected_families
        )

        results.append(
            evaluate_configuration(
                name,
                indices,
                feature_names,
                X_train,
                X_validation,
                y_validation,
                validation_groups
            )
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "average_precision",
        ascending=False
    )

    output_file = os.path.join(
        RESULTS_DIR,
        "D2_exact_v2_feature_ablation.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL RANKING")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        output_file
    )

    print("\n" + "=" * 70)
    print("STEP 46 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()