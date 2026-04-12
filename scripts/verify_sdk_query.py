import asyncio
import re
from twotrim.sdk.client import TwoTrimClient
from twotrim.types import CompressionMode
from twotrim.config import load_config

async def test_sdk_query():
    load_config()
    client = TwoTrimClient(compression_mode="aggressive")
    
    # 1. Qasper-style prompt (Question: marker)
    prompt_qasper = "Historical context about Napoleon's life and battles... \n\n Question: Where was Napoleon born?"
    
    # 2. Natural language style (Question mark at end)
    prompt_natural = "The weather today in Paris is sunny. Is it a good day for a walk?"
    
    for prompt in [prompt_qasper, prompt_natural]:
        print(f"\n--- Testing Prompt: {prompt[:50]}... ---")
        # Simulate the call that triggers _compress_async
        metadata, kwargs = await client._compress_async({"messages": [{"role": "user", "content": prompt}]})
        
        final_content = kwargs["messages"][0]["content"]
        print(f"Final Prompt (Sample): {final_content[:150]}...")
        
        if "Question:" in prompt:
            if "Question: Where was Napoleon born?" in final_content:
                print("✅ Question preserved correctly via marker.")
        elif "?" in prompt:
            if "walk?" in final_content:
                print("✅ Natural question preserved correctly via detection.")

if __name__ == "__main__":
    asyncio.run(test_sdk_query())
