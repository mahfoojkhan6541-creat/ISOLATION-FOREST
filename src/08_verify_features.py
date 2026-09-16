import os
import pandas as pd

BASE_DIR = r"D:\WhiteBox\SIH2026_IF"
FEATURES_DIR = os.path.join(BASE_DIR, "features")

files = [
    "D1_train_engineered.csv",
    "D1_validation_engineered.csv",
    "D1_final_test_engineered.csv",
    "D2_train_engineered.csv",
    "D2_validation_engineered.csv",
    "D2_final_test_engineered.csv",
]

for filename in files:

    path = os.path.join(FEATURES_DIR, filename)

    print("\n" + "=" * 60)
    print(filename)
    print("=" * 60)

    if not os.path.exists(path):
        print("ERROR: File not found")
        continue

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("MaterialID groups:", df["MaterialID"].nunique())

    print("\nTarget distribution:")
    print(df["target"].value_counts().to_dict())

    print("\nMissing values:", df.isna().sum().sum())

    print("Infinite values:", df.select_dtypes(include="number")
          .isin([float("inf"), float("-inf")]).sum().sum())

    print("Columns:", len(df.columns))

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)