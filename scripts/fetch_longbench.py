import pandas as pd
from pathlib import Path
import json

# Create data directory if it doesn't exist
output_dir = Path("benchmarks/data")
output_dir.mkdir(parents=True, exist_ok=True)

# Select a subset of LongBench tasks for demonstration
# Tasks: narrativeqa (QA), qasper (QA), multifieldqa_en (QA)
TASKS = ["narrativeqa", "qasper", "multifieldqa_en"]

def fetch_task(task_name):
    print(f"Fetching LongBench task '{task_name}' from HuggingFace...")
    try:
        # Use pandas + hf:// to pull Parquet split
        url = f"hf://datasets/THUDM/LongBench/{task_name}/test-00000-of-00001.parquet"
        df = pd.read_parquet(url)
        
        output_path = output_dir / f"longbench_{task_name}.jsonl"
        df.to_json(output_path, orient="records", lines=True)
        
        print(f"Successfully saved {len(df)} samples to {output_path}")
        return True
    except Exception as e:
        print(f"Error fetching task {task_name}: {e}")
        return False

if __name__ == "__main__":
    success_count = 0
    for task in TASKS:
        if fetch_task(task):
            success_count += 1
            
    print(f"\nDone! Successfully fetched {success_count}/{len(TASKS)} LongBench tasks.")
    print(f"Files are located in: {output_dir.absolute()}")
