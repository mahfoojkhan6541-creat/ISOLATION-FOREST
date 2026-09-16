import os
import numpy as np
import pandas as pd


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


def main():

    input_file = os.path.join(
        RESULTS_DIR,
        "D2_enhanced_validation_results.csv"
    )

    df = pd.read_csv(input_file)

    # ---------------------------------------------------------
    # Engineering constraint
    # ---------------------------------------------------------
    #
    # We do not simply select maximum F1.
    # We first keep thresholds with acceptable FPR.
    #
    # The candidate below uses FPR <= 0.35.
    # This is a development operating constraint, not a
    # claim that 35% FPR is acceptable for the final system.
    # ---------------------------------------------------------

    MAX_FPR = 0.35

    acceptable = df[
        df["fpr"] <= MAX_FPR
    ].copy()

    if acceptable.empty:

        print(
            "\nNo threshold satisfies the FPR constraint."
        )

        print(
            f"Maximum allowed FPR: {MAX_FPR}"
        )

        return

    # ---------------------------------------------------------
    # Select highest F1 among acceptable thresholds
    # ---------------------------------------------------------

    selected = acceptable.loc[
        acceptable["f1"].idxmax()
    ]

    print("\n" + "=" * 70)
    print("THRESHOLD SELECTION - D2 VALIDATION")
    print("=" * 70)

    print(
        f"\nMaximum allowed validation FPR: "
        f"{MAX_FPR:.2f}"
    )

    print(
        "\nSelected operating configuration:"
    )

    print(
        selected.to_string()
    )

    # ---------------------------------------------------------
    # Show top candidates
    # ---------------------------------------------------------

    top_candidates = (
        acceptable
        .sort_values(
            ["f1", "recall"],
            ascending=False
        )
        .head(10)
    )

    print(
        "\nTop 10 acceptable candidates:"
    )

    print(
        top_candidates[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "fpr",
                "fnr",
                "tp",
                "tn",
                "fp",
                "fn"
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Save selected configuration
    # ---------------------------------------------------------

    configuration = {
        "dataset_used_for_selection": "D2",
        "split_used_for_selection": "validation",
        "method": "Isolation Forest",
        "feature_representation": "enhanced",
        "score_column": "score_0.01",
        "max_validation_fpr": MAX_FPR,
        "selected_threshold": float(
            selected["threshold"]
        ),
        "validation_precision": float(
            selected["precision"]
        ),
        "validation_recall": float(
            selected["recall"]
        ),
        "validation_f1": float(
            selected["f1"]
        ),
        "validation_fpr": float(
            selected["fpr"]
        ),
        "validation_fnr": float(
            selected["fnr"]
        ),
        "validation_tp": int(
            selected["tp"]
        ),
        "validation_tn": int(
            selected["tn"]
        ),
        "validation_fp": int(
            selected["fp"]
        ),
        "validation_fn": int(
            selected["fn"]
        )
    }

    output_file = os.path.join(
        RESULTS_DIR,
        "D2_selected_threshold.csv"
    )

    pd.DataFrame(
        [configuration]
    ).to_csv(
        output_file,
        index=False
    )

    print(
        "\nSelected threshold saved:",
        output_file
    )

    print("\n" + "=" * 70)
    print("STEP 22 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()