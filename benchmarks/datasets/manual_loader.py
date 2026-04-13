"""Manual dataset loader for local JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.datasets.gsm8k import GSM8KDataset
from benchmarks.datasets.longbench import LongBenchDataset
from benchmarks.evaluators import MCQEvaluator, NeedleEvaluator, CodeEvaluator

class CustomDataset:
    """Generic evaluator for custom text-based datasets."""
    def evaluate(self, prediction: str, reference: Any) -> float:
        """Evaluate using basic string presence or similarity."""
        if not prediction:
            return 0.0
        
        # If reference is a list (like LongBench), check any match
        if isinstance(reference, list):
            for ref in reference:
                if self._check(prediction, str(ref)):
                    return 1.0
            return 0.0
            
        return 1.0 if self._check(prediction, str(reference)) else 0.0

    def _check(self, prediction: str, reference: str) -> bool:
        """Check if reference exists in prediction (case-insensitive)."""
        pred_clean = prediction.lower().strip()
        ref_clean = reference.lower().strip()
        return ref_clean in pred_clean or pred_clean in ref_clean

class ManualDataset:
    """Loader and Evaluator for local JSONL benchmark files."""

    def __init__(self, dataset_type: str = "gsm8k", data_path: str | Path | None = None) -> None:
        self.dataset_type = dataset_type
        self.data_path = Path(data_path) if data_path else None
        
        # Reuse existing dataset-specific evaluators for consistent scoring
        self._evaluators = {
            "gsm8k": GSM8KDataset(),
            "longbench": LongBenchDataset(),
            "scbench": MCQEvaluator(),
            "mmlu": MCQEvaluator(),
            "ruler": NeedleEvaluator(),
            "needle": NeedleEvaluator(),
            "humaneval": CodeEvaluator(),
            "zeroscrolls": LongBenchDataset(), # ZeroSCROLLS usually evaluates similar to LongBench (ROUGE-L/F1)
            "custom": CustomDataset()
        }

    def load(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Load the local JSONL dataset from a given path."""
        if not self.data_path or not self.data_path.exists():
            raise FileNotFoundError(f"Manual benchmark data not found at {self.data_path}")
            
        data = []
        count = 0
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if limit and count >= limit:
                    break
                
                try:
                    item = json.loads(line)
                    processed_item = self._process_item(item)
                    if processed_item:
                        data.append(processed_item)
                        count += 1
                except json.JSONDecodeError:
                    continue
                    
        return data

    def _process_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Map raw JSONL fields to unified Benchmark format."""
        
        # Determine format based on common keys if not specified
        if "question" in item and "answer" in item and "####" in str(item["answer"]):
            dataset_type = "gsm8k"
        elif "multi_turns" in item or ("context" in item and "input" in item and "answers" in item):
            dataset_type = self.dataset_type # LongBench or SCBench
        else:
            dataset_type = self.dataset_type

        if dataset_type == "gsm8k":
            # GSM8K: {"question": "...", "answer": "... #### 42"}
            question = item.get("question") or item.get("prompt")
            answer_full = item.get("answer") or item.get("reference")
            if not question or not answer_full: return None
            final_answer = answer_full.split("#### ")[-1].strip() if "#### " in str(answer_full) else str(answer_full).strip()
            return {"id": f"gsm8k_manual_{len(item)}", "prompt": question, "reference": final_answer}
            
        elif dataset_type in ["longbench", "zeroscrolls", "mini_longbench", "minilongbench"]:
            # LongBench/ZeroSCROLLS/MiniLongBench
            # Some datasets have "context" and "input", others just "input" (full prompt)
            context = item.get("context")
            question = item.get("input") or item.get("question")
            reference = item.get("answers") or item.get("answer") or item.get("output") or item.get("reference")

            # Handle pre-normalized format (prompt + reference keys) used by sample/smoke-test files
            if not context and not question and item.get("prompt"):
                if not reference:
                    return None
                if not isinstance(reference, list):
                    reference = [reference]
                return {
                    "id": item.get("id", f"{dataset_type}_manual_{len(item)}"),
                    "prompt": item["prompt"],
                    "reference": reference,
                }

            if not reference: return None
            
            # Determine the dataset sub-type for appropriate prompting
            ds_name = item.get("dataset", "").lower()
            
            if not context and question:
                # ZeroSCROLLS format: input contains the full prompt
                prompt = question
            elif context and question:
                # Standard QA format: context + question
                prompt = f"Please read the following context and answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}"
            elif context and not question:
                # Summarization/counting/code tasks: context only, no question
                if ds_name in ("gov_report", "multi_news"):
                    prompt = f"Please write a summary of the following text.\n\n{context}"
                elif ds_name in ("lcc", "repobench-p"):
                    prompt = f"Please complete the following code.\n\n{context}"
                elif ds_name == "passage_count":
                    prompt = f"How many independent passages are in the following text? Answer with just a number.\n\n{context}"
                elif ds_name == "passage_retrieval_en":
                    prompt = f"Which passage contains the answer? Read the following passages and identify the relevant one.\n\n{context}"
                else:
                    prompt = f"Please read the following and provide a response.\n\n{context}"
            else:
                return None
                
            # Normalize reference to list for LongBenchEvaluator
            if not isinstance(reference, list):
                reference = [reference]
                
            return {"id": f"{dataset_type}_manual_{len(item)}", "prompt": prompt, "reference": reference}
            
        elif dataset_type == "scbench":
            # SCBench (Microsoft version): {"context": "...", "multi_turns": [{"input": "...", "answer": "..."}]}
            context = item.get("context")
            turns = item.get("multi_turns", [])
            if not context or not turns: 
                # fallback for other SCBench versions
                input_q = item.get("input") or item.get("question")
                answer = item.get("answer") or item.get("answers")
                if not context or not input_q: return None
                prompt = f"Context:\n{context}\n\nQuestion:\n{input_q}"
                return {"id": f"scbench_manual_{len(item)}", "prompt": prompt, "reference": answer}
                
            # We'll take the first turn for now. Simple benchmark.
            turn = turns[0]
            prompt = f"Context:\n{context}\n\nQuestion:\n{turn['input']}"
            return {"id": f"scbench_manual_{len(item)}", "prompt": prompt, "reference": turn['answer']}
            
        elif dataset_type == "mmlu":
            # MMLU: {"question": "...", "choices": ["A", "B", "C", "D"], "answer": 0}
            question = item.get("question")
            choices = item.get("choices", [])
            answer_idx = item.get("answer")
            if question is None or not choices: return None
            options = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
            prompt = f"The following are multiple choice questions (with answers) about {item.get('subject', 'general knowledge')}.\n\nQuestion: {question}\n\nChoices:\n{options}\n\nAnswer:"
            reference = chr(65+answer_idx) if isinstance(answer_idx, int) else str(answer_idx)
            return {"id": f"mmlu_{len(item)}", "prompt": prompt, "reference": reference}
            
        elif dataset_type == "humaneval":
            # HumanEval: {"prompt": "...", "canonical_solution": "..."}
            prompt = item.get("prompt")
            reference = item.get("canonical_solution")
            if not prompt: return None
            return {"id": f"humaneval_{len(item)}", "prompt": prompt, "reference": reference}
            
        elif dataset_type in ["needle", "ruler"]:
            context = item.get("context")
            input_q = item.get("input")
            reference = item.get("answers") or [item.get("answer")]
            if not context or not input_q: return None
            prompt = f"Read the context and answer the question accurately.\n\nContext:\n{context}\n\nQuestion:\n{input_q}"
            return {"id": f"needle_manual_{len(item)}", "prompt": prompt, "reference": reference}
            
        return None

    def evaluate(self, prediction: str, reference: Any) -> float:
        """Evaluate using the appropriate dataset-specific logic."""
        # Normalize dataset type for evaluation logic mapping
        eval_key = self.dataset_type
        if "longbench" in eval_key or "zeroscrolls" in eval_key:
            eval_key = "longbench"
        elif "needle" in eval_key or "ruler" in eval_key:
            eval_key = "ruler"
            
        evaluator = self._evaluators.get(eval_key, self._evaluators.get("custom"))
        if evaluator:
            score = evaluator.evaluate(prediction, reference)
            return score
        return 0.0
