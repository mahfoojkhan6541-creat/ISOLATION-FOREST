import os
import numpy as np
import pandas as pd
import joblib


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


CONTAMINATION_VALUES = [
    0.01,
    0.03,
    0.05,
    0.10,
    0.15,
    0.20
]


def generate_scores(dataset_name):

    print("\n" + "=" * 60)
    print(f"ANOMALY SCORE GENERATION - {dataset_name}")
    print("=" * 60)

    model_file = os.path.join(
        MODELS_DIR,
        f"{dataset_name}_if_candidates.pkl"
    )

    models = joblib.load(model_file)

    for split in ["validation", "final_test"]:

        feature_file = os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_X_{split}.npy"
        )

        metadata_file = os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_{split}_engineered.csv"
        )

        X = np.load(feature_file)
        metadata = pd.read_csv(metadata_file)

        output = metadata[
            ["MaterialID", "StepID", "duration_ms", "is_test", "target"]
        ].copy()

        print(f"\n{split.upper()}")
        print("Input shape:", X.shape)

        for contamination in CONTAMINATION_VALUES:

            model = models[contamination]

            # Larger value = more normal.
            # Convert it so that larger value = more anomalous.
            anomaly_score = -model.score_samples(X)

            prediction = model.predict(X)

            output[
                f"score_{contamination}"
            ] = anomaly_score

            output[
                f"prediction_{contamination}"
            ] = np.where(
                prediction == -1,
                "ABNORMAL",
                "NORMAL"
            )

        output_file = os.path.join(
            RESULTS_DIR,
            f"{dataset_name}_{split}_anomaly_scores.csv"
        )

        output.to_csv(
            output_file,
            index=False
        )

        print("Saved:", output_file)

        # Display score ranges
        print("\nScore ranges:")

        for contamination in CONTAMINATION_VALUES:

            column = f"score_{contamination}"

            print(
                f"{contamination:>5}: "
                f"min={output[column].min():.6f}, "
                f"max={output[column].max():.6f}, "
                f"mean={output[column].mean():.6f}"
            )


if __name__ == "__main__":

    generate_scores("D1")
    generate_scores("D2")

    print("\n" + "=" * 60)
    print("STEP 11 COMPLETE")
    print("=" * 60)