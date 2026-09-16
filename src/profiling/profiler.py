import numpy as np
import pandas as pd
from typing import Dict, Any, List


class DatasetProfiler:
    """Automated dataset profiler inspecting structure, types, missingness, and candidate concepts."""

    def profile(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, Any]:
        records = len(df)
        cols = list(df.columns)

        # Missingness analysis
        missing_by_field = {col: int(df[col].isnull().sum()) for col in cols}
        total_missing = int(df.isnull().sum().sum())

        # Exact duplicate rows
        duplicate_rows = int(df.duplicated().sum())

        # Classify candidate fields
        candidate_ids = []
        candidate_time = []
        candidate_numeric = []
        candidate_categories = []

        for col in cols:
            col_lower = col.lower()
            dtype = str(df[col].dtype)
            nunique = df[col].nunique()

            if any(k in col_lower for k in ["id", "material", "component", "part", "serial"]):
                candidate_ids.append(col)
            if any(k in col_lower for k in ["time", "step", "duration", "date", "checkpoint", "cycle"]):
                candidate_time.append(col)

            if pd.api.types.is_numeric_dtype(df[col]):
                candidate_numeric.append(col)
            elif nunique < 50:
                candidate_categories.append(col)

        # Check temporal order & continuity if time field exists
        temporal_ordered = True
        unique_checkpoints = []
        if candidate_time:
            primary_time = candidate_time[0]
            try:
                unique_checkpoints = sorted([int(x) if isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer()) else x for x in df[primary_time].dropna().unique()])
            except Exception:
                unique_checkpoints = list(df[primary_time].dropna().unique()[:20])

        # Entity counts
        unique_components = 0
        if candidate_ids:
            primary_id = candidate_ids[0]
            unique_components = int(df[primary_id].nunique())

        # Numeric ranges & saturation check
        numeric_summaries = {}
        warnings: List[str] = []

        if duplicate_rows > 0:
            warnings.append(f"Found {duplicate_rows} exact duplicate rows.")

        for num_col in candidate_numeric[:25]:
            vals = df[num_col].dropna()
            if len(vals) > 0:
                min_v = float(vals.min())
                max_v = float(vals.max())
                std_v = float(vals.std()) if len(vals) > 1 else 0.0
                numeric_summaries[num_col] = {
                    "min": min_v,
                    "max": max_v,
                    "mean": float(vals.mean()),
                    "std": std_v
                }
                # Check for constant/zero variance
                if std_v < 1e-12 and len(vals) > 1:
                    warnings.append(f"Column '{num_col}' has near-zero variance ({std_v:.2e}).")

        report = {
            "dataset_name": dataset_name,
            "dimensions": {
                "records": records,
                "columns_count": len(cols)
            },
            "columns": cols,
            "data_types": {col: str(df[col].dtype) for col in cols},
            "missingness": {
                "total_missing_values": total_missing,
                "by_field": missing_by_field
            },
            "duplicates": {
                "exact_rows": duplicate_rows
            },
            "candidates": {
                "candidate_id_fields": candidate_ids,
                "candidate_time_fields": candidate_time,
                "candidate_numeric_fields": candidate_numeric,
                "candidate_categories": candidate_categories
            },
            "entities": {
                "primary_id_field": candidate_ids[0] if candidate_ids else None,
                "unique_components": unique_components
            },
            "temporal": {
                "primary_time_field": candidate_time[0] if candidate_time else None,
                "unique_checkpoints": unique_checkpoints,
                "checkpoint_count": len(unique_checkpoints)
            },
            "numeric_summaries": numeric_summaries,
            "warnings": warnings
        }
        return report


AutoProfiler = DatasetProfiler
