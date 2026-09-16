import os
import yaml
from typing import Dict, Any, List, Optional


class PopulationPolicy:
    """Encapsulates hierarchical reference population rules according to Section 19."""

    def __init__(self, config_path: str = "configs/populations/population_rules.yaml"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            self.config = {
                "hierarchical_levels": {
                    "level_1": {
                        "name": "same_checkpoint",
                        "keys": ["checkpoint"],
                        "min_population_size": 10,
                        "target_exclusion": True
                    }
                }
            }
            return
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

    def get_levels(self) -> Dict[str, Any]:
        return self.config.get("hierarchical_levels", {})

    def get_validation_rules(self) -> Dict[str, Any]:
        return self.config.get("validation", {
            "min_comparable_components": 5,
            "exclude_target_from_reference": True,
            "prevent_future_leakage": True
        })
