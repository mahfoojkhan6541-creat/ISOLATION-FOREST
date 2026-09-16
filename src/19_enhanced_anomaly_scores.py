import os
import numpy as np
import pandas as pd
import joblib


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_scores(dataset_name):

    print("\n" + "=" * 70)
    print(f"ENHANCED ANOMALY SCORES - {dataset_name}")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load candidate models
    # ---------------------------------------------------------

    model_file = os.path.join(
        MODELS_DIR,
        f"{dataset_name}_enhanced_if_candidates.pkl"
    )

    models = joblib.load(model_file)

    # ---------------------------------------------------------
    # Process each split
    # ---------------------------------------------------------

    for split in [
        "train",
        "validation",
        "final_test"
    ]:

        feature_file = os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_enhanced_X_{split}_processed.npy"
        )

        source_file = os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_{split}_enhanced.csv"
        )

        X = np.load(feature_file)

        source_df = pd.read_csv(source_file)

        result = source_df[
            [
                "MaterialID",
                "StepID",
                "duration_ms",
                "is_test",
                "target"
            ]
        ].copy()

        print(f"\n{split.upper()}")
        print("Samples:", len(X))

        # -----------------------------------------------------
        # Generate continuous anomaly score
        # -----------------------------------------------------

        for contamination, model in models.items():

            score = -model.score_samples(X)

            result[
                f"score_{contamination}"
            ] = score

            print(
                f"contamination={contamination}: "
                f"min={score.min():.6f}, "
                f"max={score.max():.6f}, "
                f"mean={score.mean():.6f}"
            )

        # -----------------------------------------------------
        # Save scores
        # -----------------------------------------------------

        output_file = os.path.join(
            RESULTS_DIR,
            f"{dataset_name}_{split}_enhanced_anomaly_scores.csv"
        )

        result.to_csv(
            output_file,
            index=False
        )

        print(
            "Saved:",
            output_file
        )


if __name__ == "__main__":

    generate_scores("D1")
    generate_scores("D2")

    print("\n" + "=" * 70)
    print("STEP 19 COMPLETE")
    print("=" * 70)