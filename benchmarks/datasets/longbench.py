"""LongBench Benchmark Dataset.

Used to evaluate how much critical context is pruned or distorted
during semantic or rule-based compression in a long prompt scenario.
"""

from __future__ import annotations

import json
from typing import Any

class LongBenchDataset:
    """Loader and Evaluator for LongBench (NarrativeQA, Qasper, etc)."""

    def __init__(self, subset: str = "narrativeqa", split: str = "test") -> None:
        self.subset = subset
        self.split = split
        self._data: list[dict[str, Any]] = []

    def load(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Load the dataset subset from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        ds = load_dataset("THUDM/LongBench", self.subset, split=self.split, trust_remote_code=True)
        
        count = 0
        for item in ds:
            if limit and count >= limit:
                break
                
            input_context = item["context"]
            question = item["input"]
            answers = item["answers"]
            
            # format the prompt similar to a standard RAG input
            prompt = f"Please read the following context and answer the question.\n\nContext:\n{input_context}\n\nQuestion:\n{question}"

            self._data.append({
                "id": f"longbench_{self.subset}_{count}",
                "prompt": prompt,
                "reference": answers, # usually a list of valid references
            })
            count += 1
            
        return self._data

    def evaluate(self, prediction: str, reference: list[str]) -> float:
        """Evaluate using ROUGE-L (offline, using rouge_score directly)."""
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            raise ImportError("Please install rouge_score: pip install rouge_score")

        # Cache the scorer instance for reuse across calls
        if not hasattr(self, "_scorer"):
            self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        
        # LongBench answers are usually lists of valid strings. We'll take the max ROUGE-L over all references.
        max_score = 0.0
        
        for ref in reference:
            scores = self._scorer.score(ref, prediction)
            score = scores["rougeL"].fmeasure
            if score > max_score:
                max_score = score
                    
        return max_score
