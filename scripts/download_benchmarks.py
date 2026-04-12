"""Download and prepare all benchmark datasets locally for TwoTrim.
Supported: LongBench (21+ tasks), SCBench, MMLU, GSM8K, HumanEval, Qasper, ZeroSCROLLS, MiniLongBench, RULER.
Includes resume support to avoid re-downloading existing files.
"""

import os
import json
import random
from pathlib import Path
from typing import Any, Dict, List

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found. Please install it with 'pip install datasets'.")
    exit(1)

# Configuration
DATA_DIR = Path("benchmarks/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
LIMIT_PER_TASK = 50 

def save_jsonl(data: List[Dict[str, Any]], filename: Path):
    """Save list of dicts to JSONL."""
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} items to {filename.name}")

def download_dataset(hf_path: str, subset: str = None, split: str = "test", name: str = None, force: bool = False):
    """Generic dataset downloader with resume support."""
    name = name or (f"{hf_path}_{subset}" if subset else hf_path).replace("/", "_")
    target_file = DATA_DIR / f"{name}.jsonl"
    
    if target_file.exists() and not force:
        print(f"Skipping {hf_path} ({subset if subset else 'main'}) - Already exists.")
        return

    print(f"\nDownloading {hf_path} ({subset if subset else 'main'})...")
    try:
        ds = load_dataset(hf_path, subset, split=split, trust_remote_code=True)
        data = []
        for i, item in enumerate(ds):
            if i >= LIMIT_PER_TASK: break
            data.append(item)
        save_jsonl(data, target_file)
    except Exception as e:
        print(f"Failed to download {hf_path} ({subset}): {e}")

def main():
    # 1. LongBench
    longbench_tasks = [
        "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", 
        "musique", "gov_report", "qmsum", "multi_news", "passage_count", 
        "passage_retrieval_en", "lcc", "repobench-p"
    ]
    for task in longbench_tasks:
        download_dataset("THUDM/LongBench", task, name=f"longbench_{task}")

    # 2. SCBench
    scbench_subsets = ["scbench_kv", "scbench_summary", "scbench_choice_eng", "scbench_qa_eng"]
    for subset in scbench_subsets:
        download_dataset("microsoft/SCBench", subset, name=f"scbench_{subset}")

    # 3. GSM8K, MMLU, HumanEval
    download_dataset("openai/gsm8k", "main", name="gsm8k")
    download_dataset("cais/mmlu", "abstract_algebra", name="mmlu")
    download_dataset("openai_humaneval", split="test", name="humaneval")

    # 4. ZeroSCROLLS (Fixed names)
    zeroscrolls_tasks = ["gov_report", "qmsum", "squality", "musique"] 
    for task in zeroscrolls_tasks:
        download_dataset("tau/zero_scrolls", task, name=f"zeroscrolls_{task}")

    # 5. MiniLongBench
    download_dataset("THUDM/LongBench", "narrativeqa", name="minilongbench_narrativeqa")

    # 6. RULER / Needle Generator
    target_ruler = DATA_DIR / "ruler_synthetic.jsonl"
    if not target_ruler.exists():
        print("\nGenerating Needle-in-a-Haystack and RULER tasks...")
        needles = ["The secret code is 12345.", "Whiskers the cat is the CEO."]
        lengths = [8000, 16000]
        ruler_data = []
        for length in lengths:
            for needle in needles:
                context = "Filler text. " * (length // 10) + f" {needle} " + "More filler. " * (length // 10)
                ruler_data.append({"context": context, "input": "What is the secret or CEO name?", "answers": [needle]})
        save_jsonl(ruler_data, target_ruler)
    else:
        print("Skipping RULER synthetic - Already exists.")

    print("\nBatch process finished. Check output for any remaining failures.")

if __name__ == "__main__":
    main()
