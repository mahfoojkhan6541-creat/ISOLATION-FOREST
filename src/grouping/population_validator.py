import pandas as pd
from typing import Dict, Any, Tuple, Optional


class PopulationValidator:
    """Validates whether a selected candidate peer group is usable without bias or leakage."""

    def __init__(self, min_components: int = 5, target_exclusion: bool = True):
        self.min_components = min_components
        self.target_exclusion = target_exclusion

    def validate_population(
        self,
        peer_df: pd.DataFrame,
        target_component_id: str,
        checkpoint: Any,
        min_size: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        required_size = min_size if min_size is not None else self.min_components

        # Check 1: Empty check
        if peer_df.empty:
            return False, "Candidate peer group is completely empty."

        # Check 2: Target exclusion verification
        if self.target_exclusion and target_component_id in peer_df["component_id"].values:
            return False, f"Target component {target_component_id} was not excluded from peer reference group!"

        # Check 3: Minimum distinct component count
        distinct_peers = peer_df["component_id"].nunique()
        if distinct_peers < required_size:
            return False, f"Insufficient distinct peer components ({distinct_peers} < {required_size})."

        # Check 4: Checkpoint alignment
        if "checkpoint" in peer_df.columns:
            mismatches = (peer_df["checkpoint"] != checkpoint).sum()
            if mismatches > 0:
                return False, f"Peer group contains {mismatches} records with mismatched checkpoints."

        return True, None
