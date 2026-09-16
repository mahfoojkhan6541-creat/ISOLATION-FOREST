import os
import json
import datetime
import pandas as pd
from typing import Dict, Any


class QuarantineManager:
    """Manages isolated storage and provenance logging for quarantined and blocked records."""

    def __init__(self, quarantine_dir: str = "data/quarantine"):
        self.quarantine_dir = quarantine_dir
        os.makedirs(self.quarantine_dir, exist_ok=True)

    def isolate_records(
        self,
        quarantined_df: pd.DataFrame,
        dataset_id: str,
        summary: Dict[str, Any]
    ) -> str:
        """Stores quarantined records into a designated audit parquet/csv with metadata."""
        if len(quarantined_df) == 0:
            return "NO_RECORDS_QUARANTINED"

        timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        batch_filename = f"{dataset_id.lower()}_quarantined_{timestamp_str}.csv"
        batch_filepath = os.path.join(self.quarantine_dir, batch_filename)

        quarantined_df.to_csv(batch_filepath, index=False)

        # Save quarantine provenance audit metadata
        meta_filename = f"{dataset_id.lower()}_quarantined_{timestamp_str}_meta.json"
        meta_filepath = os.path.join(self.quarantine_dir, meta_filename)
        with open(meta_filepath, "w", encoding="utf-8") as f:
            json.dump({
                "dataset_id": dataset_id,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "quarantined_count": len(quarantined_df),
                "file_path": batch_filepath,
                "events": summary.get("events_detected", [])
            }, f, indent=2)

        return batch_filepath
