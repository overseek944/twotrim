"""GSM8K (Grade School Math 8K) Benchmark Dataset.

Used to evaluate if compression destructs reasoning ability
and multi-step calculation logic.
"""

from __future__ import annotations

import re
from typing import Any

class GSM8KDataset:
    """Loader and Evaluator for GSM8K."""

    def __init__(self, split: str = "test") -> None:
        self.split = split
        self._data: list[dict[str, Any]] = []

    def load(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Load the dataset from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        ds = load_dataset("openai/gsm8k", "main", split=self.split, trust_remote_code=True)
        
        count = 0
        for item in ds:
            if limit and count >= limit:
                break
                
            question = item["question"]
            answer_full = item["answer"]
            
            # GSM8K standard format puts the final answer after "#### "
            final_answer = answer_full.split("#### ")[-1].strip()
            
            self._data.append({
                "id": f"gsm8k_{count}",
                "prompt": question,
                "reference": final_answer,
                "full_reference": answer_full,
            })
            count += 1
            
        return self._data

    def evaluate(self, prediction: str, reference: str) -> float:
        """Evaluate if the prediction contains the exact numerical reference."""
        # Find all numbers in the prediction text
        numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", prediction.replace(",", ""))
        
        if not numbers:
            return 0.0
            
        # Standard GSM8K evaluation: usually the last generated number should match the reference
        # or we check if the exact reference number is clearly marked/present.
        ref_num = reference.replace(",", "")
        
        try:
            ref_float = float(ref_num)
            for n in reversed(numbers):
                if abs(float(n) - ref_float) < 1e-6:
                    return 1.0
        except ValueError:
            pass
            
        return 0.0
