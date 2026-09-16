import pandas as pd
from typing import Dict, Any, Optional, Tuple
from src.grouping.population_policy import PopulationPolicy
from src.grouping.population_validator import PopulationValidator


class PopulationSelector:
    """Selects and validates comparable peer populations for a target component according to policy."""

    def __init__(self, policy: PopulationPolicy = None, validator: PopulationValidator = None):
        self.policy = policy or PopulationPolicy()
        self.validator = validator or PopulationValidator()

    def select_reference_population(
        self,
        canonical_df: pd.DataFrame,
        target_component_id: str,
        checkpoint: Any,
        as_of_checkpoint: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Selects reference peer components enforcing target exclusion and no-leakage time filtering.
        """
        # Step 1: Time Filtering - Only use data available at or before analysis time
        cutoff_cp = as_of_checkpoint if as_of_checkpoint is not None else checkpoint
        # Assuming checkpoints can be compared numerically or equality
        if isinstance(cutoff_cp, (int, float)):
            time_filtered = canonical_df[canonical_df["checkpoint"] <= cutoff_cp]
        else:
            time_filtered = canonical_df[canonical_df["checkpoint"] == cutoff_cp]

        # Step 2: Target Exclusion - Ensure target component never contaminates its own reference
        peers_candidates = time_filtered[time_filtered["component_id"] != str(target_component_id)]

        # Step 3: Iterate through hierarchical policy levels
        levels = self.policy.get_levels()
        for level_key, level_cfg in levels.items():
            keys = level_cfg.get("keys", ["checkpoint"])
            min_size = level_cfg.get("min_population_size", 5)

            # Match on grouping keys (e.g. same checkpoint)
            query_mask = pd.Series(True, index=peers_candidates.index)
            if "checkpoint" in keys and "checkpoint" in peers_candidates.columns:
                query_mask &= (peers_candidates["checkpoint"] == checkpoint)

            matched_peers = peers_candidates[query_mask]

            is_valid, reason = self.validator.validate_population(
                peer_df=matched_peers,
                target_component_id=str(target_component_id),
                checkpoint=checkpoint,
                min_size=min_size
            )

            if is_valid:
                return {
                    "status": "VALID",
                    "population_id": f"pop_{level_key}_{checkpoint}",
                    "level": level_key,
                    "level_name": level_cfg.get("name", level_key),
                    "peer_count": matched_peers["component_id"].nunique(),
                    "total_records": len(matched_peers),
                    "members_df": matched_peers,
                    "reason": "Successfully selected valid comparable reference population"
                }

        # Step 4: Fallback / Controlled non-model outcome
        return {
            "status": "UNAVAILABLE",
            "population_id": "none",
            "level": "none",
            "peer_count": 0,
            "total_records": 0,
            "members_df": pd.DataFrame(),
            "reason": "No approved comparable peer population satisfies minimum size and quality requirements"
        }
