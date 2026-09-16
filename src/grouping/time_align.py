import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class TimeAligner:
    """Manages time normalization, sparse checkpoint alignment, and progressive streaming slicing."""

    @staticmethod
    def get_ordered_checkpoints(df: pd.DataFrame) -> List[Any]:
        """Returns sorted unique checkpoints present in the dataset."""
        if "checkpoint" not in df.columns:
            return []
        cps = df["checkpoint"].dropna().unique()
        try:
            return sorted(list(cps))
        except Exception:
            return list(cps)

    @staticmethod
    def slice_as_of_checkpoint(
        df: pd.DataFrame,
        as_of_checkpoint: Any
    ) -> pd.DataFrame:
        """
        Enforces the absolute No-Leakage Rule (Section 45).
        Returns only records up to and including as_of_checkpoint.
        """
        if "checkpoint" not in df.columns:
            return df
        if isinstance(as_of_checkpoint, (int, float)):
            return df[df["checkpoint"] <= as_of_checkpoint].copy()
        return df[df["checkpoint"] == as_of_checkpoint].copy()

    @staticmethod
    def get_component_trajectory(
        df: pd.DataFrame,
        component_id: str,
        param_key: str,
        as_of_checkpoint: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Extracts chronologically sorted trajectory of measurements for a single component.
        """
        comp_df = df[df["component_id"] == str(component_id)]
        if as_of_checkpoint is not None and "checkpoint" in comp_df.columns:
            if isinstance(as_of_checkpoint, (int, float)):
                comp_df = comp_df[comp_df["checkpoint"] <= as_of_checkpoint]

        sort_cols = [c for c in ["checkpoint", "elapsed_time", "raw_row_index"] if c in comp_df.columns]
        if sort_cols:
            comp_df = comp_df.sort_values(by=sort_cols)

        return comp_df[["component_id", "checkpoint", "elapsed_time", param_key]].dropna(subset=[param_key])
