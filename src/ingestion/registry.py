import os
import yaml
import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import pandas as pd
from src.ingestion.adapters import get_adapter, compute_file_sha256


@dataclass
class DatasetMetadata:
    dataset_id: str
    file_path: str
    domain: str = "burn_in"
    source_format: str = "csv"
    sampling_type: str = "progressive_checkpoints"
    intended_task: str = "trajectory_anomaly_and_forecast"
    notes: str = ""


class DatasetRegistry:
    """Manages dataset registration, checksum verification, and intake metadata."""

    def __init__(self, configs_dir: str = "configs/datasets"):
        self.configs_dir = configs_dir
        self.registered_datasets: Dict[str, Dict[str, Any]] = {}
        self.load_all_configs()

    def load_all_configs(self):
        if not os.path.exists(self.configs_dir):
            return
        for file in os.listdir(self.configs_dir):
            if file.endswith(".yaml") or file.endswith(".yml"):
                filepath = os.path.join(self.configs_dir, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    if config and "dataset_id" in config:
                        self.registered_datasets[config["dataset_id"]] = config

    def register_dataset(
        self,
        dataset_id: str,
        source_path: str,
        source_type: str = "csv",
        source_name: Optional[str] = None,
        intended_tasks: Optional[list] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """Registers a new dataset with computed provenance checksum."""
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source data file does not exist: {source_path}")

        checksum = compute_file_sha256(source_path)
        record = {
            "dataset_id": dataset_id,
            "source_type": source_type,
            "source_name": source_name or dataset_id,
            "source_path": source_path,
            "source_version": f"sha256:{checksum[:12]}",
            "sha256": checksum,
            "received_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "intended_tasks": intended_tasks or ["anomaly"],
            "status": "registered",
            "notes": notes
        }
        self.registered_datasets[dataset_id] = record

        # Save to YAML configuration
        os.makedirs(self.configs_dir, exist_ok=True)
        config_path = os.path.join(self.configs_dir, f"{dataset_id.lower()}_dataset.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(record, f, default_flow_style=False)

        return record

    def register(self, meta: DatasetMetadata) -> Dict[str, Any]:
        return self.register_dataset(
            dataset_id=meta.dataset_id,
            source_path=meta.file_path,
            source_type=meta.source_format,
            notes=meta.notes
        )

    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        return self.registered_datasets.get(dataset_id)

    def get(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        info = self.get_dataset_info(dataset_id)
        if info:
            # ensure 'file_path' key is also present
            if "file_path" not in info and "source_path" in info:
                info["file_path"] = info["source_path"]
        return info

    def load_dataset(self, dataset_id: str, **kwargs) -> pd.DataFrame:
        """Loads a registered dataset through its appropriate adapter."""
        info = self.get_dataset_info(dataset_id)
        if not info:
            raise KeyError(f"Dataset '{dataset_id}' is not registered in the system.")

        source_path = info.get("source_path")
        source_type = info.get("source_type", "csv")
        adapter = get_adapter(source_type)
        return adapter.read(source_path, **kwargs)
