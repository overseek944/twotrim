import asyncio
import logging
from twotrim.compression.semantic import SemanticCompressor
from twotrim.types import PolicyDecision, StrategyName
from twotrim.config import load_config

# Setup logging to see what's happening
logging.basicConfig(level=logging.DEBUG)

async def test():
    # Load default config to get model names etc.
    load_config()
    
    # Initialize compressor with low threshold for testing
    comp = SemanticCompressor(
        min_input_length=10, 
        prefer_extractive=False # Ensure we test abstractive first
    )
    
    context = (
        "The TwoTrim project is a high-performance token compression fabric designed for LLM applications. "
        "It uses multiple strategies including rule-based, embedding-based, and semantic summarization. "
        "The goal is to reduce token counts by 40-80% while retaining 95%+ of the original information quality. "
        "Recently, it was updated with question-aware compression to improve accuracy on benchmarks like Qasper."
    )
    query = "What is the goal of TwoTrim?"
    
    print("\n--- Phase 2: Testing Abstractive with Question-Awareness ---")
    res = await comp.compress(context, query=query)
    print(f"Original Tokens: {res.original_tokens}")
    print(f"Compressed Tokens: {res.compressed_tokens}")
    print(f"Compressed Text: {res.compressed_text}")
    print(f"Method Used: {res.metadata.get('method')}")

    print("\n--- Phase 3: Testing Smart Extractive Scoring ---")
    # Force extractive
    comp.prefer_extractive = True
    res_ext = await comp.compress(context, query=query)
    print(f"Extractive Text: {res_ext.compressed_text}")
    print(f"Method Used: {res_ext.metadata.get('method')}")

if __name__ == "__main__":
    asyncio.run(test())
