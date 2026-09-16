import os
import yaml
from typing import Dict, Any, Optional


class MappingLoader:
    """Loads and validates versioned schema and semantic mapping configurations."""

    def __init__(self, mappings_dir: str = "configs/mappings"):
        self.mappings_dir = mappings_dir

    def load_mapping(self, mapping_version: str) -> Dict[str, Any]:
        filename = f"{mapping_version}.yaml" if not mapping_version.endswith(".yaml") else mapping_version
        filepath = os.path.join(self.mappings_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Mapping configuration not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Validate required keys
        required_keys = ["mapping_version", "identity_fields", "temporal_fields", "parameter_fields"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Malformed mapping configuration: missing key '{key}' in {filepath}")

        return config
