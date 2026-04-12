"""Example: SDK client usage.

Use the TwoTrim SDK as a drop-in replacement for the OpenAI client.
"""

import os
from twotrim.sdk.client import TwoTrimClient

# Option 1: Through a running proxy
client = TwoTrimClient(
    proxy_url="http://localhost:8000/v1",
    api_key=os.environ.get("OPENAI_API_KEY"),
    compression_mode="balanced",
)

# Make requests just like OpenAI
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Explain quantum computing in one paragraph."},
    ],
)

print("Response:", response["choices"][0]["message"]["content"])

# Check stats
stats = client.get_stats()
print(f"\nAggregate stats:")
print(f"  Total requests: {stats.get('total_requests', 0)}")
print(f"  Tokens saved: {stats.get('total_tokens_saved', 0)}")
print(f"  Cost saved: ${stats.get('total_cost_saved_usd', 0):.4f}")

# Health check
health = client.health()
print(f"\nProxy status: {health.get('status')}")
