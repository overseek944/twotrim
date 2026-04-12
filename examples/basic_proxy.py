"""Example: Basic proxy usage.

Start the TwoTrim proxy server, then send requests to it
using a standard OpenAI client.

Usage:
    1. Start the proxy:   twotrim serve
    2. Run this script:   python examples/basic_proxy.py
"""

import os
from openai import OpenAI

# Point the OpenAI client at the TwoTrim proxy
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.environ.get("OPENAI_API_KEY", "sk-..."),
)

# Make a request — it will be automatically compressed
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a very helpful and knowledgeable AI assistant. "
                "Please provide detailed and accurate answers to all questions. "
                "Make sure that your responses are comprehensive and thorough. "
                "It is important to note that you should always be helpful."
            ),
        },
        {
            "role": "user",
            "content": "What is machine learning? Explain in 2 sentences.",
        },
    ],
)

print("Response:", response.choices[0].message.content)
print()

# Check the TwoTrim metadata (available via raw response)
import httpx

raw = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    },
    headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
)
data = raw.json()
if "_twotrim" in data:
    tf = data["_twotrim"]
    print(f"Compression ratio: {tf.get('ratio', 0):.1%}")
    print(f"Strategies: {tf.get('strategies', [])}")
    print(f"Total time: {tf.get('total_time_ms', 0):.0f}ms")
