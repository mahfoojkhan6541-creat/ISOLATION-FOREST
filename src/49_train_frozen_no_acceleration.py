import os
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def main():

    print("\n" + "=" * 70)
    print("STEP 49 - TRAIN FROZEN V2 WITHOUT ACCELERATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load frozen configuration
    # ---------------------------------------------------------

    config_path = os.path.join(
        MODELS_DIR,
        "D2_v2_no_acceleration_config.pkl"
    )

    config = joblib.load(config_path)

    selected_features = config["model_features"]

    print("\nDataset:", config["dataset"])
    print("Selected features:", len(selected_features))
    print("Contamination:", config["candidate_contamination"])
    print("Trees:", config["n_estimators"])

    # ---------------------------------------------------------
    # Load TRAIN features
    # ---------------------------------------------------------

    train_path = os.path.join(
        FEATURES_DIR,
        "D2_enhanced_X_train.npy"
    )

    X_train_full = np.load(train_path)

    print("\nOriginal TRAIN shape:", X_train_full.shape)

    # Load exact original V2 feature configuration
    original_config = joblib.load(
        os.path.join(
            MODELS_DIR,
            "D2_enhanced_feature_config.pkl"
        )
    )

    original_features = original_config["model_features"]

    # ---------------------------------------------------------
    # Select the exact 156 features
    # ---------------------------------------------------------

    selected_indices = [
        original_features.index(feature)
        for feature in selected_features
    ]

    X_train = X_train_full[:, selected_indices]

    print("Frozen TRAIN shape:", X_train.shape)

    # ---------------------------------------------------------
    # PREPROCESSING
    #
    # Fit ONLY on TRAIN
    # ---------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imputed = imputer.fit_transform(
        X_train
    )

    scaler = StandardScaler()

    X_train_processed = scaler.fit_transform(
        X_train_imputed
    )

    print(
        "Processed TRAIN shape:",
        X_train_processed.shape
    )

    # ---------------------------------------------------------
    # TRAIN ISOLATION FOREST
    #
    # TRAIN DATA ONLY
    # No target used.
    # ---------------------------------------------------------

    model = IsolationForest(
        n_estimators=config["n_estimators"],
        contamination=config["candidate_contamination"],
        random_state=config["random_state"],
        n_jobs=config["n_jobs"]
    )

    model.fit(X_train_processed)

    print("\nIsolation Forest training completed.")

    # ---------------------------------------------------------
    # Save preprocessing
    # ---------------------------------------------------------

    preprocessing = {
        "imputer": imputer,
        "scaler": scaler,
        "selected_features": selected_features,
        "selected_indices": selected_indices,
        "feature_count": len(selected_features),
        "dataset": "D2",
        "fit_data": "TRAIN_ONLY"
    }

    preprocessing_path = os.path.join(
        MODELS_DIR,
        "D2_v2_no_acceleration_preprocessing.pkl"
    )

    joblib.dump(
        preprocessing,
        preprocessing_path
    )

    # ---------------------------------------------------------
    # Save frozen model bundle
    # ---------------------------------------------------------

    frozen_model = {
        "model": model,
        "preprocessing": preprocessing,
        "feature_config": config,
        "threshold": config["validation_threshold"],
        "decision_level": config["decision_level"],
        "score_aggregation": config["score_aggregation"],
        "decision_rule": (
            "mean_score >= threshold -> ABNORMAL"
        ),
        "training_data": "D2 TRAIN ONLY",
        "validation_used_for_threshold": True,
        "final_test_used_for_training": False,
        "final_test_used_for_selection": False
    }

    model_path = os.path.join(
        MODELS_DIR,
        "D2_v2_no_acceleration_frozen_model.pkl"
    )

    joblib.dump(
        frozen_model,
        model_path
    )

    print("\nSaved preprocessing:")
    print(preprocessing_path)

    print("\nSaved frozen model:")
    print(model_path)

    print("\nFrozen threshold:")
    print(config["validation_threshold"])

    print("\nFinal test used for training: NO")
    print("Final test used for threshold selection: NO")

    print("\n" + "=" * 70)
    print("STEP 49 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()