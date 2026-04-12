"""Local dataset loader for offline benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.datasets.gsm8k import GSM8KDataset
from benchmarks.datasets.longbench import LongBenchDataset

class LocalDataset:
    """Loader and Evaluator for local JSON datasets."""

    def __init__(self, dataset_type: str = "gsm8k") -> None:
        self.dataset_type = dataset_type
        # Mapping dataset types to their specific evaluators
        self._evaluators = {
            "gsm8k": GSM8KDataset(),
            "longbench": LongBenchDataset()
        }

    def load(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Load the local JSON dataset."""
        data_path = Path("benchmarks/data")
        
        if self.dataset_type == "gsm8k":
            file_path = data_path / "gsm8k_tiny.json"
        elif self.dataset_type == "longbench":
            file_path = data_path / "longbench_tiny.json"
        else:
            raise ValueError(f"Unknown local dataset type: {self.dataset_type}")
            
        if not file_path.exists():
            raise FileNotFoundError(f"Local benchmark data not found at {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if limit:
            return data[:limit]
        return data

    def evaluate(self, prediction: str, reference: Any) -> float:
        """Evaluate using the appropriate dataset-specific logic."""
        evaluator = self._evaluators.get(self.dataset_type)
        if evaluator:
            return evaluator.evaluate(prediction, reference)
        return 0.0
