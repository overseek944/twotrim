import pandas as pd
from pathlib import Path

# Create data directory if it doesn't exist
output_dir = Path("benchmarks/data")
output_dir.mkdir(parents=True, exist_ok=True)

splits = {
    'train': 'socratic/train-00000-of-00001.parquet', 
    'test': 'socratic/test-00000-of-00001.parquet'
}

print("Fetching GSM8K 'train' split from HuggingFace...")
try:
    df_train = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])
    train_path = output_dir / "gsm8k_train.jsonl"
    df_train.to_json(train_path, orient="records", lines=True)
    print(f"Successfully saved {len(df_train)} train records to {train_path}")

    print("Fetching GSM8K 'test' split from HuggingFace...")
    df_test = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["test"])
    test_path = output_dir / "gsm8k_test.jsonl"
    df_test.to_json(test_path, orient="records", lines=True)
    print(f"Successfully saved {len(df_test)} test records to {test_path}")
    
except Exception as e:
    print(f"Error fetching or saving dataset: {e}")
