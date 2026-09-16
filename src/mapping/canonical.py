import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from src.mapping.loader import MappingLoader
from src.mapping.units import UnitConverter


class CanonicalTransformer:
    """Transforms heterogeneous raw source datasets into the standardized Canonical Internal Representation."""

    def __init__(self, mapping_loader: MappingLoader = None, unit_converter: UnitConverter = None):
        self.mapping_loader = mapping_loader or MappingLoader()
        self.unit_converter = unit_converter or UnitConverter()

    def transform(
        self,
        df: pd.DataFrame,
        mapping_config: Dict[str, Any],
        dataset_id: str = "UNKNOWN"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Transforms raw input DataFrame into canonical format adhering to Section 20 of the guide."""
        canonical_df = pd.DataFrame(index=df.index)

        # 1. Map Identity Fields
        for id_field in mapping_config["identity_fields"]:
            src = id_field["source_field"]
            concept = id_field["internal_concept"]
            if src in df.columns:
                canonical_df[concept] = df[src].astype(str)
            else:
                if id_field.get("required", True):
                    raise KeyError(f"Required identity field '{src}' missing in input data.")
                canonical_df[concept] = "UNKNOWN"

        # 2. Map Temporal Fields
        for t_field in mapping_config["temporal_fields"]:
            src = t_field["source_field"]
            concept = t_field["internal_concept"]
            if src in df.columns:
                canonical_df[concept] = df[src]
            else:
                if t_field.get("required", True):
                    raise KeyError(f"Required temporal field '{src}' missing in input data.")
                canonical_df[concept] = np.nan

        # 3. Map Parameter Fields & Units
        param_metadata = {}
        for p_field in mapping_config["parameter_fields"]:
            src = p_field["source_field"]
            pkey = p_field["parameter_key"]
            unit = p_field.get("canonical_unit", "arb_norm")
            if src in df.columns:
                series = df[src].astype(float)
                # Apply unit plausibility check
                _, err = self.unit_converter.check_plausible_bounds(series, unit)
                canonical_df[pkey] = series
                param_metadata[pkey] = {
                    "source_field": src,
                    "canonical_unit": unit,
                    "warning": err
                }
            else:
                if p_field.get("required", True):
                    raise KeyError(f"Required parameter field '{src}' missing in input data.")

        # 4. Map Metadata / Evaluation Fields (Kept strictly separate from model inputs)
        for m_field in mapping_config.get("metadata_fields", []):
            src = m_field["source_field"]
            concept = m_field["internal_concept"]
            if src in df.columns:
                canonical_df[concept] = df[src]

        # 5. Attach Canonical Provenance
        canonical_df["raw_row_index"] = df.index
        canonical_df["dataset_id"] = dataset_id
        canonical_df["mapping_version"] = mapping_config.get("mapping_version", "1.0")

        # Metadata bundle
        metadata = {
            "dataset_id": dataset_id,
            "mapping_version": mapping_config.get("mapping_version", "1.0"),
            "parameter_keys": list(param_metadata.keys()),
            "parameter_metadata": param_metadata,
            "total_canonical_records": len(canonical_df),
            "unique_components": int(canonical_df["component_id"].nunique()) if "component_id" in canonical_df else 0,
            "checkpoints": sorted(canonical_df["checkpoint"].dropna().unique().tolist()) if "checkpoint" in canonical_df else []
        }

        return canonical_df, metadata
