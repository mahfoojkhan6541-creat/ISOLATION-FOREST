import os
import numpy as np
import pandas as pd
import joblib


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def main():

    print("\n" + "=" * 70)
    print("STEP 50A - REBUILD FINAL TEST FEATURE ARRAY")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load existing final-test enhanced CSV
    # ---------------------------------------------------------

    csv_path = os.path.join(
        FEATURES_DIR,
        "D2_final_test_enhanced.csv"
    )

    df = pd.read_csv(csv_path)

    print("\nSource CSV shape:", df.shape)

    # ---------------------------------------------------------
    # Load the EXACT original V2 feature configuration
    # ---------------------------------------------------------

    config_path = os.path.join(
        MODELS_DIR,
        "D2_enhanced_feature_config.pkl"
    )

    config = joblib.load(config_path)

    model_features = config["model_features"]

    print(
        "Original V2 model features:",
        len(model_features)
    )

    # ---------------------------------------------------------
    # Extract exactly the same 175 features
    # ---------------------------------------------------------

    X_final_test = df[
        model_features
    ].to_numpy(
        dtype=np.float64
    )

    print(
        "Rebuilt FINAL TEST shape:",
        X_final_test.shape
    )

    # ---------------------------------------------------------
    # Verify target is NOT inside the feature array
    # ---------------------------------------------------------

    assert "target" not in model_features
    assert "MaterialID" not in model_features
    assert "StepID" not in model_features
    assert "duration_ms" not in model_features
    assert "is_test" not in model_features

    # ---------------------------------------------------------
    # Save correct NumPy array
    # ---------------------------------------------------------

    output_path = os.path.join(
        FEATURES_DIR,
        "D2_enhanced_X_final_test.npy"
    )

    np.save(
        output_path,
        X_final_test
    )

    # ---------------------------------------------------------
    # Immediately verify the saved file
    # ---------------------------------------------------------

    verification = np.load(
        output_path
    )

    print(
        "\nVerified saved array shape:",
        verification.shape
    )

    print(
        "Data type:",
        verification.dtype
    )

    print(
        "Array verification:",
        np.allclose(
            X_final_test,
            verification,
            equal_nan=True
        )
    )

    print(
        "\nSaved:",
        output_path
    )

    print("\n" + "=" * 70)
    print("STEP 50A COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()