"""Specialized evaluators for TwoTrim benchmarks."""

import re
from typing import Any, List

class MCQEvaluator:
    """Evaluator for Multiple Choice Questions (MMLU, SCBench Choice)."""
    def evaluate(self, prediction: str, reference: str) -> float:
        if not prediction:
            return 0.0
        
        # Clean prediction: "Answer is (A)" -> "A"
        # Look for (A) or A. or Option A
        match = re.search(r"\(([A-D])\)|(?:\s|^)([A-D])(?:\s|\.|$|:)", prediction.upper())
        pred_choice = match.group(1) or match.group(2) if match else prediction.strip().upper()[:1]
        
        ref_choice = reference.strip().upper()[:1]
        return 1.0 if pred_choice == ref_choice else 0.0

class NeedleEvaluator:
    """Evaluator for Needle-in-a-Haystack / RULER."""
    def evaluate(self, prediction: str, reference: Any) -> float:
        if not prediction:
            return 0.0
            
        pred_clean = prediction.lower().strip()
        
        # Reference can be a list of needles
        references = reference if isinstance(reference, list) else [reference]
        
        for ref in references:
            ref_clean = str(ref).lower().strip()
            # Fuzzy match: if any significant part of the needle is in the prediction
            if ref_clean in pred_clean or pred_clean in ref_clean:
                return 1.0
            
            # Check for high overlap
            words_ref = set(ref_clean.split())
            words_pred = set(pred_clean.split())
            if words_ref and len(words_ref & words_pred) / len(words_ref) > 0.7:
                return 1.0
                
        return 0.0

class CodeEvaluator:
    """Evaluator for HumanEval (Basic)."""
    def evaluate(self, prediction: str, reference: str) -> float:
        if not prediction:
            return 0.0
            
        # For now, we check if the prediction contains key logic from the reference
        # or if it's a valid-looking function completion.
        # Deep evaluation would require execution in a sandbox.
        pred_clean = prediction.strip()
        ref_clean = reference.strip()
        
        # Simple heuristic: if it's identical or contains the core logic
        if ref_clean in pred_clean:
            return 1.0
            
        return 0.5 if len(set(pred_clean.split()) & set(ref_clean.split())) / len(set(ref_clean.split())) > 0.5 else 0.0
