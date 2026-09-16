import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def main():

    print("\n" + "=" * 70)
    print("STEP 51 - FROZEN FINAL TEST EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load frozen configuration
    # ---------------------------------------------------------

    model_path = os.path.join(
        MODELS_DIR,
        "D2_v2_no_acceleration_frozen_model.pkl"
    )

    frozen = joblib.load(model_path)

    threshold = frozen["threshold"]

    print("\nFrozen threshold:", threshold)
    print("Decision level:", frozen["decision_level"])
    print("Aggregation:", frozen["score_aggregation"])

    # ---------------------------------------------------------
    # Load final-test predictions
    # ---------------------------------------------------------

    prediction_path = os.path.join(
        RESULTS_DIR,
        "D2_v2_no_acceleration_final_test_predictions.csv"
    )

    df = pd.read_csv(prediction_path)

    print("\nFinal-test prediction file loaded.")
    print("Total MaterialIDs:", len(df))

    # ---------------------------------------------------------
    # Convert labels to binary
    # ---------------------------------------------------------

    y_true = (
        df["actual"]
        .map({
            "NORMAL": 0,
            "ABNORMAL": 1
        })
        .to_numpy()
    )

    y_pred = (
        df["prediction"]
        .map({
            "NORMAL": 0,
            "ABNORMAL": 1
        })
        .to_numpy()
    )

    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

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

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"\nAccuracy    : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision   : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall      : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1 Score    : {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"FPR         : {fpr:.4f} ({fpr * 100:.2f}%)")
    print(f"FNR         : {fnr:.4f} ({fnr * 100:.2f}%)")

    print("\nConfusion Matrix:")
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)

    # ---------------------------------------------------------
    # Save evaluation
    # ---------------------------------------------------------

    evaluation = pd.DataFrame([
        {
            "dataset": "D2",
            "decision_level": "MaterialID",
            "score_aggregation": "mean_score",
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "total_materials": len(df)
        }
    ])

    output_path = os.path.join(
        RESULTS_DIR,
        "D2_v2_no_acceleration_final_test_evaluation.csv"
    )

    evaluation.to_csv(
        output_path,
        index=False
    )

    print("\nEvaluation saved:")
    print(output_path)

    print("\n" + "=" * 70)
    print("STEP 51 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()