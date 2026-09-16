import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def create_group_split(file_path, dataset_name, random_state=42):

    print("\n" + "=" * 70)
    print(f"{dataset_name} - LEAKAGE-SAFE GROUP SPLIT")
    print("=" * 70)

    df = pd.read_csv(file_path)

    # ---------------------------------------------------------
    # Step 1: Separate groups
    # ---------------------------------------------------------
    groups = df["MaterialID"]

    print(f"\nTotal rows: {len(df):,}")
    print(f"Total MaterialID groups: {groups.nunique():,}")

    # ---------------------------------------------------------
    # Step 2: TRAIN = 70%
    # Remaining = 30%
    # ---------------------------------------------------------
    train_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=random_state
    )

    train_idx, remaining_idx = next(
        train_splitter.split(
            df,
            groups=groups
        )
    )

    train_df = df.iloc[train_idx].copy()
    remaining_df = df.iloc[remaining_idx].copy()

    # ---------------------------------------------------------
    # Step 3:
    # Remaining 30% → Validation 15% + Final Test 15%
    # ---------------------------------------------------------
    remaining_groups = remaining_df["MaterialID"]

    validation_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=random_state
    )

    validation_idx, test_idx = next(
        validation_splitter.split(
            remaining_df,
            groups=remaining_groups
        )
    )

    validation_df = remaining_df.iloc[validation_idx].copy()
    test_df = remaining_df.iloc[test_idx].copy()

    # ---------------------------------------------------------
    # Step 4: Check group leakage
    # ---------------------------------------------------------
    train_groups = set(train_df["MaterialID"])
    validation_groups = set(validation_df["MaterialID"])
    test_groups = set(test_df["MaterialID"])

    train_validation_overlap = (
        train_groups & validation_groups
    )

    train_test_overlap = (
        train_groups & test_groups
    )

    validation_test_overlap = (
        validation_groups & test_groups
    )

    # ---------------------------------------------------------
    # Step 5: Print results
    # ---------------------------------------------------------
    print("\nSPLIT RESULTS")
    print("-" * 70)

    print(
        f"TRAIN      : {len(train_df):,} rows | "
        f"{len(train_groups):,} MaterialID groups"
    )

    print(
        f"VALIDATION : {len(validation_df):,} rows | "
        f"{len(validation_groups):,} MaterialID groups"
    )

    print(
        f"FINAL TEST : {len(test_df):,} rows | "
        f"{len(test_groups):,} MaterialID groups"
    )

    print("\nGROUP LEAKAGE CHECK")
    print("-" * 70)

    print(
        f"TRAIN ↔ VALIDATION : "
        f"{len(train_validation_overlap)}"
    )

    print(
        f"TRAIN ↔ FINAL TEST : "
        f"{len(train_test_overlap)}"
    )

    print(
        f"VALIDATION ↔ FINAL TEST : "
        f"{len(validation_test_overlap)}"
    )

    # ---------------------------------------------------------
    # Step 6: Verify
    # ---------------------------------------------------------
    if (
        len(train_validation_overlap) == 0
        and len(train_test_overlap) == 0
        and len(validation_test_overlap) == 0
    ):
        print("\n✓ NO GROUP LEAKAGE DETECTED")
    else:
        raise RuntimeError(
            "GROUP LEAKAGE DETECTED!"
        )

    # ---------------------------------------------------------
    # Step 7: Save splits
    # ---------------------------------------------------------
    train_df.to_csv(
        RESULTS_DIR / f"{dataset_name}_train.csv",
        index=False
    )

    validation_df.to_csv(
        RESULTS_DIR / f"{dataset_name}_validation.csv",
        index=False
    )

    test_df.to_csv(
        RESULTS_DIR / f"{dataset_name}_final_test.csv",
        index=False
    )

    print("\nSaved files:")
    print(
        RESULTS_DIR / f"{dataset_name}_train.csv"
    )
    print(
        RESULTS_DIR / f"{dataset_name}_validation.csv"
    )
    print(
        RESULTS_DIR / f"{dataset_name}_final_test.csv"
    )

    return train_df, validation_df, test_df


# D1
create_group_split(
    DATA_DIR / "D1.csv",
    "D1"
)

# D2
create_group_split(
    DATA_DIR / "D2.csv",
    "D2"
)

print("\n" + "=" * 70)
print("GROUP SPLITTING COMPLETED")
print("=" * 70)