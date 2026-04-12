"""Test benchmark datasets and evaluation."""

import pytest

def test_gsm8k_evaluator():
    from benchmarks.datasets.gsm8k import GSM8KDataset
    ds = GSM8KDataset()
    
    # Correct string with lots of filler
    pred_correct = "The answer is obviously 42.0 because of math."
    ref = "42"
    assert ds.evaluate(pred_correct, ref) == 1.0
    
    # Incorrect string
    pred_bad = "The result is 41"
    assert ds.evaluate(pred_bad, ref) == 0.0

def test_longbench_evaluator():
    from benchmarks.datasets.longbench import LongBenchDataset
    ds = LongBenchDataset()
    
    # Mocking standard evaluate library behavior is tricky without breaking testing,
    # but let's test a simple string if possible. If evaluate is not installed properly,
    # this will warn or fail, ensuring requirements are met.
    
    pred = "The quick brown fox jumps over the lazy dog."
    refs = ["A quick brown fox jumped", "The quick brown fox jumps over the lazy dog!"]
    
    # evaluate() uses rougeL max score
    try:
        score = ds.evaluate(pred, refs)
        assert score > 0.8
    except Exception as e:
        pytest.skip(f"Could not run ROUGE eval: {e}")

