import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def main():

    print("\n" + "=" * 70)
    print("STEP 47 - V2 WITHOUT ACCELERATION VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load exact V2 feature data
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

    config = joblib.load(
        os.path.join(
            MODELS_DIR,
            "D2_enhanced_feature_config.pkl"
        )
    )

    feature_names = config["model_features"]

    # ---------------------------------------------------------
    # Identify acceleration features
    # ---------------------------------------------------------

    acceleration_indices = [
        i
        for i, feature in enumerate(feature_names)
        if "_acceleration" in feature
    ]

    keep_indices = [
        i
        for i in range(len(feature_names))
        if i not in acceleration_indices
    ]

    selected_features = [
        feature_names[i]
        for i in keep_indices
    ]

    print("\nOriginal features:", len(feature_names))
    print("Acceleration features:", len(acceleration_indices))
    print("Remaining features:", len(selected_features))

    # ---------------------------------------------------------
    # Select features
    # ---------------------------------------------------------

    X_train = X_train[:, keep_indices]
    X_validation = X_validation[:, keep_indices]

    print("TRAIN shape:", X_train.shape)
    print("VALIDATION shape:", X_validation.shape)

    # ---------------------------------------------------------
    # Load validation metadata
    # ---------------------------------------------------------

    validation_df = pd.read_csv(
        os.path.join(
            FEATURES_DIR,
            "D2_validation_enhanced.csv"
        )
    )

    y_validation = validation_df["target"].to_numpy()
    material_ids = validation_df["MaterialID"].to_numpy()

    # ---------------------------------------------------------
    # Train-only preprocessing
    # ---------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train
    )

    X_validation = imputer.transform(
        X_validation
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_validation = scaler.transform(
        X_validation
    )

    # ---------------------------------------------------------
    # Train Isolation Forest
    # Same configuration as original V2
    # ---------------------------------------------------------

    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train)

    # ---------------------------------------------------------
    # Generate anomaly scores
    # ---------------------------------------------------------

    scores = -model.score_samples(
        X_validation
    )

    # ---------------------------------------------------------
    # MaterialID-level aggregation
    # ---------------------------------------------------------

    score_df = pd.DataFrame({
        "MaterialID": material_ids,
        "target": y_validation,
        "score": scores
    })

    material_df = (
        score_df
        .groupby("MaterialID")
        .agg(
            target=("target", "first"),
            mean_score=("score", "mean")
        )
        .reset_index()
    )

    y_true = material_df["target"].to_numpy()
    material_scores = material_df["mean_score"].to_numpy()

    # ---------------------------------------------------------
    # Average Precision
    # ---------------------------------------------------------

    ap = average_precision_score(
        y_true,
        material_scores
    )

    print(
        f"\nMaterial-level AP: {ap:.6f}"
    )

    # ---------------------------------------------------------
    # Threshold analysis
    # ---------------------------------------------------------

    thresholds = np.unique(
        material_scores
    )

    rows = []

    for threshold in thresholds:

        y_pred = (
            material_scores >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

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

        rows.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        })

    threshold_df = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Best F1
    # ---------------------------------------------------------

    best_f1 = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    print("\n" + "=" * 70)
    print("BEST F1 OPERATING POINT")
    print("=" * 70)

    print(
        best_f1.to_string()
    )

    # ---------------------------------------------------------
    # Best F1 with FPR <= 20%
    # ---------------------------------------------------------

    acceptable_20 = threshold_df[
        threshold_df["fpr"] <= 0.20
    ]

    if not acceptable_20.empty:

        best_20 = acceptable_20.loc[
            acceptable_20["f1"].idxmax()
        ]

        print("\n" + "=" * 70)
        print("BEST F1 WITH FPR <= 20%")
        print("=" * 70)

        print(
            best_20.to_string()
        )

    # ---------------------------------------------------------
    # Best F1 with FPR <= 15%
    # ---------------------------------------------------------

    acceptable_15 = threshold_df[
        threshold_df["fpr"] <= 0.15
    ]

    if not acceptable_15.empty:

        best_15 = acceptable_15.loc[
            acceptable_15["f1"].idxmax()
        ]

        print("\n" + "=" * 70)
        print("BEST F1 WITH FPR <= 15%")
        print("=" * 70)

        print(
            best_15.to_string()
        )

    # ---------------------------------------------------------
    # Save validation results
    # ---------------------------------------------------------

    output_file = os.path.join(
        RESULTS_DIR,
        "D2_v2_no_acceleration_threshold_analysis.csv"
    )

    threshold_df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nSaved:",
        output_file
    )

    print("\n" + "=" * 70)
    print("STEP 47 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()