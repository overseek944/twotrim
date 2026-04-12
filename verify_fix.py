import asyncio
import logging
from twotrim.compression.semantic import SemanticCompressor

logging.basicConfig(level=logging.INFO)

async def test():
    print("Testing SemanticCompressor...")
    # Using a short min_input_length for the test
    compressor = SemanticCompressor(min_input_length=10)
    
    text = (
        "TwoTrim is a powerful middleware for LLM applications. "
        "It aims to reduce token consumption by 40 to 80 percent without significant quality loss. "
        "It uses rule-based, embedding-based, and semantic compression strategies. "
        "The semantic strategy uses abstractive summarization models to rewrite verbose prompts."
    )
    
    print("\nText to compress:")
    print(text)
    
    result = await compressor.compress(text)
    
    print("\nCompression Results:")
    print(f"Method used: {result.metadata.get('method')}")
    print(f"Original Tokens: {result.original_tokens}")
    print(f"Compressed Tokens: {result.compressed_tokens}")
    print(f"Ratio: {result.compression_ratio:.1%}")
    print("\nCompressed Text:")
    print(result.compressed_text)
    
    if result.metadata.get('method') == 'abstractive':
        print("\n✅ SUCCESS: Abstractive summarization is working!")
    else:
        print("\n❌ FAILURE: Fell back to extractive compression.")

if __name__ == "__main__":
    asyncio.run(test())
