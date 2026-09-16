import os
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseSourceAdapter(ABC):
    """Abstract base adapter for dataset intake."""

    @abstractmethod
    def read(self, source_path: str, **kwargs) -> pd.DataFrame:
        pass

    def load(self, source_path: str, **kwargs) -> pd.DataFrame:
        return self.read(source_path, **kwargs)


class CSVSourceAdapter(BaseSourceAdapter):
    """Adapter for reading CSV source files with error handling and checksum."""

    def read(self, source_path: str, **kwargs) -> pd.DataFrame:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        return pd.read_csv(source_path, **kwargs)


class JSONSourceAdapter(BaseSourceAdapter):
    """Adapter for reading JSON source files."""

    def read(self, source_path: str, **kwargs) -> pd.DataFrame:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        return pd.read_json(source_path, **kwargs)


def get_adapter(source_type: str) -> BaseSourceAdapter:
    source_type = source_type.lower()
    if source_type in ["csv", "text"]:
        return CSVSourceAdapter()
    elif source_type in ["json"]:
        return JSONSourceAdapter()
    else:
        raise ValueError(f"Unsupported source type: {source_type}")


class IntakeAdapterFactory:
    """Factory helper to get adapter from path or format."""
    @staticmethod
    def get_adapter(source_path_or_type: str) -> BaseSourceAdapter:
        if "." in source_path_or_type:
            ext = source_path_or_type.split(".")[-1].lower()
            return get_adapter(ext)
        return get_adapter(source_path_or_type)


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum for provenance tracking."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
