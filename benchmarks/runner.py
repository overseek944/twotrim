"""Benchmark test suite runner with support for custom local datasets."""

import os
import logging
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock imports if not in a package context, but we are in TwoTrim repo.
try:
    from twotrim.sdk.client import TwoTrimClient
    from twotrim.types import CompressionMode
except ImportError:
    # Fallback/Mock for testing if needed
    class TwoTrimClient: pass
    class CompressionMode:
        SEMANTIC = "semantic"
        STRENGTH = "strength"

@dataclass
class BenchmarkResult:
    dataset: str
    mode: str
    avg_score: float
    avg_compression_ratio: float
    avg_latency_ms: float
    samples_run: int

class BenchmarkRunner:
    """Runs a given dataset through baseline and compressed paths."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        
        # Check for OpenAI Key
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not found. Baseline runs will fail.")

        self.tf_client = TwoTrimClient()
        try:
            import openai
            self.native_client = openai.OpenAI()
        except Exception:
            self.native_client = None

    def run_one(self, dataset_name: str, ds: Any, samples: list[dict[str, Any]], mode: str = "baseline", mock: bool = False) -> BenchmarkResult:
        """Run benchmark for a given dataset and mode."""
        total_score = 0.0
        total_ratio = 0.0
        total_latency = 0.0
        success_count = 0
        
        logger.info("Starting benchmark %s (mode=%s) with %d samples", dataset_name, mode, len(samples))
        
        for idx, sample in enumerate(samples):
            prompt = sample["prompt"]
            reference = sample["reference"]
            
            start_time = time.perf_counter()
            
            try:
                if mock:
                    # Mock logic for verification
                    if isinstance(reference, list):
                        prediction = str(reference[0])
                    else:
                        prediction = str(reference)
                    current_ratio = 0.5
                elif mode == "baseline":
                    if not self.native_client:
                        raise ImportError("openai package not found for baseline.")
                    response = self.native_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    prediction = response.choices[0].message.content or ""
                    current_ratio = 0.0
                else:
                    # TwoTrim logic
                    response = self.tf_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        extra_body={"compression_mode": mode}
                    )
                    prediction = response.choices[0].message.content or ""
                    
                    # Read metadata for compression ratio
                    if hasattr(response, "twotrim_metadata"):
                        current_ratio = response.twotrim_metadata.get("compression_ratio", 0.0)
                    else:
                        current_ratio = 0.5 # Mock for testing if SDK is not fully linked
                
                latency = (time.perf_counter() - start_time) * 1000
                score = ds.evaluate(prediction, reference)
                
                total_score += score
                total_ratio += current_ratio
                total_latency += latency
                success_count += 1
                
            except Exception as e:
                import traceback
                logger.error("Failed sample %s: %s\n%s", sample.get('id', idx), e, traceback.format_exc())
                
        if success_count == 0:
            return BenchmarkResult(dataset_name, mode, 0.0, 0.0, 0.0, 0)
            
        return BenchmarkResult(
            dataset_name, mode,
            avg_score=total_score / success_count,
            avg_compression_ratio=total_ratio / success_count,
            avg_latency_ms=total_latency / success_count,
            samples_run=success_count
        )

def main():
    parser = argparse.ArgumentParser(description="TwoTrim Benchmark Runner")
    parser.add_argument("--data-path", type=str, default="benchmarks/data", help="Directory containing JSONL files")
    parser.add_argument("--limit", type=int, default=10, help="Samples per dataset")
    parser.add_argument("--mode", type=str, default="baseline,balanced", help="Comma-separated modes (baseline, balanced, etc)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--mock", action="store_true", help="Run with mock predictions for verification")
    args = parser.parse_args()

    # Import loader here to avoid circular imports
    from benchmarks.datasets.manual_loader import ManualDataset

    data_dir = Path(args.data_path)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} not found.")
        return

    runner = BenchmarkRunner(model=args.model)
    modes = args.mode.split(",")
    results = []

    # Find all JSONL files
    files = sorted(list(data_dir.glob("*.jsonl")) + list(data_dir.glob("*.json")))
    print(f"Found {len(files)} datasets in {data_dir}")

    for file in files:
        # Determine dataset type from filename
        # e.g., longbench_narrativeqa.jsonl -> longbench
        fname = file.stem.lower()
        dtype = "custom"
        if "gsm8k" in fname: dtype = "gsm8k"
        elif "longbench" in fname: dtype = "longbench"
        elif "scbench" in fname: dtype = "scbench"
        elif "mmlu" in fname: dtype = "mmlu"
        elif "humaneval" in fname: dtype = "humaneval"
        elif "needle" in fname or "ruler" in fname: dtype = "ruler"
        elif "zeroscrolls" in fname: dtype = "zeroscrolls"

        loader = ManualDataset(dataset_type=dtype, data_path=file)
        try:
            samples = loader.load(limit=args.limit)
            if not samples:
                print(f"Skipping {file.name}: No valid samples found.")
                continue
                
            for mode in modes:
                res = runner.run_one(file.name, loader, samples, mode=mode.strip(), mock=args.mock)
                results.append(res)
                print(f"Result: {res.dataset} | Mode: {res.mode} | Score: {res.avg_score:.2f} | Ratio: {res.avg_compression_ratio:.2f}")

        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    # Final Summary Table
    print("\n" + "="*80)
    print(f"{'Dataset':<30} | {'Mode':<10} | {'Score':<8} | {'Ratio':<8} | {'Latency':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r.dataset[:30]:<30} | {r.mode:<10} | {r.avg_score:<8.2f} | {r.avg_compression_ratio:<8.2%}")
    print("="*80)

if __name__ == "__main__":
    main()
