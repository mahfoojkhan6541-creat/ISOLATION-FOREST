import os
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


def preprocess_dataset(dataset_name):

    print("\n" + "=" * 70)
    print(f"ENHANCED PREPROCESSING - {dataset_name}")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load prepared data
    # ---------------------------------------------------------

    X_train = np.load(
        os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_enhanced_X_train.npy"
        )
    )

    X_validation = np.load(
        os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_enhanced_X_validation.npy"
        )
    )

    X_final_test = np.load(
        os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_enhanced_X_final_test.npy"
        )
    )

    print("Original TRAIN shape:", X_train.shape)
    print(
        "Original VALIDATION shape:",
        X_validation.shape
    )
    print(
        "Original FINAL TEST shape:",
        X_final_test.shape
    )

    # ---------------------------------------------------------
    # 1. Median Imputation
    # FIT ONLY ON TRAIN
    # ---------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imputed = imputer.fit_transform(
        X_train
    )

    X_validation_imputed = imputer.transform(
        X_validation
    )

    X_final_test_imputed = imputer.transform(
        X_final_test
    )

    # ---------------------------------------------------------
    # Verify no missing values
    # ---------------------------------------------------------

    print(
        "\nMissing values after imputation:"
    )

    print(
        "TRAIN:",
        np.isnan(X_train_imputed).sum()
    )

    print(
        "VALIDATION:",
        np.isnan(X_validation_imputed).sum()
    )

    print(
        "FINAL TEST:",
        np.isnan(X_final_test_imputed).sum()
    )

    # ---------------------------------------------------------
    # 2. Standard Scaling
    # FIT ONLY ON TRAIN
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_imputed
    )

    X_validation_scaled = scaler.transform(
        X_validation_imputed
    )

    X_final_test_scaled = scaler.transform(
        X_final_test_imputed
    )

    # ---------------------------------------------------------
    # Verify finite values
    # ---------------------------------------------------------

    print(
        "\nInfinite values after preprocessing:"
    )

    print(
        "TRAIN:",
        np.isinf(X_train_scaled).sum()
    )

    print(
        "VALIDATION:",
        np.isinf(X_validation_scaled).sum()
    )

    print(
        "FINAL TEST:",
        np.isinf(X_final_test_scaled).sum()
    )

    # ---------------------------------------------------------
    # Save processed arrays
    # ---------------------------------------------------------

    np.save(
        os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_enhanced_X_train_processed.npy"
        ),
        X_train_scaled
    )

    np.save(
        os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_enhanced_X_validation_processed.npy"
        ),
        X_validation_scaled
    )

    np.save(
        os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_enhanced_X_final_test_processed.npy"
        ),
        X_final_test_scaled
    )

    # ---------------------------------------------------------
    # Save preprocessing pipeline
    # ---------------------------------------------------------

    preprocessing = {
        "imputer": imputer,
        "scaler": scaler
    }

    preprocessing_file = os.path.join(
        MODELS_DIR,
        f"{dataset_name}_enhanced_preprocessing.pkl"
    )

    joblib.dump(
        preprocessing,
        preprocessing_file
    )

    print(
        "\nProcessed TRAIN shape:",
        X_train_scaled.shape
    )

    print(
        "Processed VALIDATION shape:",
        X_validation_scaled.shape
    )

    print(
        "Processed FINAL TEST shape:",
        X_final_test_scaled.shape
    )

    print(
        "\nPreprocessing saved:",
        preprocessing_file
    )

    print(
        "\nImputer fitted on TRAIN only: YES"
    )

    print(
        "Scaler fitted on TRAIN only: YES"
    )


if __name__ == "__main__":

    preprocess_dataset("D1")
    preprocess_dataset("D2")

    print("\n" + "=" * 70)
    print("STEP 17 COMPLETE")
    print("=" * 70)