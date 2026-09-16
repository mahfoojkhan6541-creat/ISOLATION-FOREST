import os
import joblib
import pandas as pd


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def main():

    config_path = os.path.join(
        MODELS_DIR,
        "D2_v2_no_acceleration_config.pkl"
    )

    evaluation_path = os.path.join(
        RESULTS_DIR,
        "D2_v2_no_acceleration_final_test_evaluation.csv"
    )

    prediction_path = os.path.join(
        RESULTS_DIR,
        "D2_v2_no_acceleration_final_test_predictions.csv"
    )

    config = joblib.load(config_path)
    evaluation = pd.read_csv(evaluation_path).iloc[0]
    predictions = pd.read_csv(prediction_path)

    report_path = os.path.join(
        RESULTS_DIR,
        "D2_Isolation_Forest_Final_Report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write("SIH26170 - ISOLATION FOREST FINAL REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. DATASET\n")
        f.write("-" * 70 + "\n")
        f.write(f"Dataset: {config['dataset']}\n\n")

        f.write("2. MODEL\n")
        f.write("-" * 70 + "\n")
        f.write(f"Method: {config['method']}\n")
        f.write(f"Feature representation: {config['feature_representation']}\n")
        f.write(f"Number of estimators: {config['n_estimators']}\n")
        f.write(f"Random state: {config['random_state']}\n")
        f.write(f"Candidate contamination: {config['candidate_contamination']}\n")
        f.write(f"Score definition: {config['score_definition']}\n\n")

        f.write("3. FEATURES\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Original feature count: "
            f"{config['original_feature_count']}\n"
        )
        f.write(
            f"Removed feature family: "
            f"{config['removed_feature_family']}\n"
        )
        f.write(
            f"Removed feature count: "
            f"{config['removed_feature_count']}\n"
        )
        f.write(
            f"Final model feature count: "
            f"{config['model_feature_count']}\n\n"
        )

        f.write("4. DECISION CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Decision level: {config['decision_level']}\n")
        f.write(f"Score aggregation: {config['score_aggregation']}\n")
        f.write(
            f"Validation threshold: "
            f"{config['validation_threshold']}\n"
        )
        f.write(
            f"Threshold selection split: "
            f"{config['threshold_selection_split']}\n"
        )
        f.write(
            f"Threshold frozen: "
            f"{config['threshold_frozen']}\n\n"
        )

        f.write("5. VALIDATION PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Average Precision: "
            f"{config['validation_average_precision']:.6f}\n"
        )
        f.write(
            f"Precision: "
            f"{config['validation_precision']:.6f}\n"
        )
        f.write(
            f"Recall: "
            f"{config['validation_recall']:.6f}\n"
        )
        f.write(
            f"F1: "
            f"{config['validation_f1']:.6f}\n"
        )
        f.write(
            f"FPR: "
            f"{config['validation_fpr']:.6f}\n"
        )
        f.write(
            f"FNR: "
            f"{config['validation_fnr']:.6f}\n\n"
        )

        f.write("6. FINAL TEST PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Accuracy: "
            f"{evaluation['accuracy'] * 100:.2f}%\n"
        )
        f.write(
            f"Precision: "
            f"{evaluation['precision'] * 100:.2f}%\n"
        )
        f.write(
            f"Recall: "
            f"{evaluation['recall'] * 100:.2f}%\n"
        )
        f.write(
            f"F1: "
            f"{evaluation['f1'] * 100:.2f}%\n"
        )
        f.write(
            f"FPR: "
            f"{evaluation['fpr'] * 100:.2f}%\n"
        )
        f.write(
            f"FNR: "
            f"{evaluation['fnr'] * 100:.2f}%\n\n"
        )

        f.write("7. CONFUSION MATRIX\n")
        f.write("-" * 70 + "\n")
        f.write(f"TN: {int(evaluation['TN'])}\n")
        f.write(f"FP: {int(evaluation['FP'])}\n")
        f.write(f"FN: {int(evaluation['FN'])}\n")
        f.write(f"TP: {int(evaluation['TP'])}\n")
        f.write(
            f"Total MaterialIDs: "
            f"{int(evaluation['total_materials'])}\n\n"
        )

        f.write("8. FINAL PREDICTIONS\n")
        f.write("-" * 70 + "\n")

        prediction_counts = predictions["prediction"].value_counts()

        f.write(
            f"NORMAL: "
            f"{int(prediction_counts.get('NORMAL', 0))}\n"
        )
        f.write(
            f"ABNORMAL: "
            f"{int(prediction_counts.get('ABNORMAL', 0))}\n"
        )
        f.write(
            f"Total: "
            f"{len(predictions)}\n\n"
        )

        f.write("9. LEAKAGE CHECK\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Final test used for threshold selection: "
            f"{config['final_test_used_for_selection']}\n"
        )
        f.write(
            "Threshold source: validation\n"
        )
        f.write(
            "Final-test labels used for model training: No\n"
        )
        f.write(
            "Final-test labels used for threshold selection: No\n\n"
        )

        f.write("=" * 70 + "\n")
        f.write("FINAL REPORT COMPLETE\n")
        f.write("=" * 70 + "\n")

    print("\nFinal report created successfully.")
    print(report_path)


if __name__ == "__main__":
    main()