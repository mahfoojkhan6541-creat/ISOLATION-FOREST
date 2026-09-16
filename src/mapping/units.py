import os
import yaml
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


class UnitConverter:
    """Manages unit mapping, conversions, and physical plausibility range checks."""

    def __init__(self, unit_policy_path: str = "configs/units/unit_policy.yaml"):
        self.unit_policy_path = unit_policy_path
        self.policy: Dict[str, Any] = {}
        self.load_policy()

    def load_policy(self):
        if not os.path.exists(self.unit_policy_path):
            self.policy = {"units": {}}
            return
        with open(self.unit_policy_path, "r", encoding="utf-8") as f:
            self.policy = yaml.safe_load(f) or {"units": {}}

    def convert_series(
        self,
        series: pd.Series,
        source_unit: str,
        target_unit: str
    ) -> Tuple[pd.Series, Optional[str]]:
        """Converts a numerical series from source unit to target canonical unit."""
        if source_unit == target_unit or source_unit == "arb_norm":
            return series, None

        units_cfg = self.policy.get("units", {})
        if target_unit in units_cfg:
            conv = units_cfg[target_unit].get("conversions", {}).get(source_unit)
            if conv:
                factor = float(conv.get("factor", 1.0))
                offset = float(conv.get("offset", 0.0))
                return (series * factor) + offset, None

        return series, f"No explicit conversion rule from '{source_unit}' to '{target_unit}'"

    def check_plausible_bounds(
        self,
        series: pd.Series,
        canonical_unit: str
    ) -> Tuple[bool, Optional[str]]:
        """Checks if values lie within plausible engineering limits defined in policy."""
        units_cfg = self.policy.get("units", {})
        unit_info = units_cfg.get(canonical_unit, {})
        validation = unit_info.get("validation", {})
        p_min = validation.get("plausible_min")
        p_max = validation.get("plausible_max")

        violations = 0
        if p_min is not None:
            violations += int((series < p_min).sum())
        if p_max is not None:
            violations += int((series > p_max).sum())

        is_plausible = (violations == 0)
        return is_plausible, f"Found {violations} out-of-plausible-range values for unit {canonical_unit}" if violations > 0 else None
