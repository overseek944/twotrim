"""Example: Custom compression policy."""

import asyncio
from twotrim.compression.pipeline import CompressionPipeline
from twotrim.types import CompressionMode, PolicyDecision, StrategyName


async def main():
    pipeline = CompressionPipeline()

    text = """
    You are a highly skilled and experienced software engineer with deep expertise
    in Python programming. Your primary objective is to review the following code
    and provide constructive feedback. Please make sure to identify any bugs,
    performance issues, or style problems. It is important to note that you should
    also suggest improvements where applicable. Remember that code quality is
    essential for maintainability.

    Here is the code to review:

    ```python
    def fibonacci(n):
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            return fibonacci(n-1) + fibonacci(n-2)
    ```

    Please provide a thorough review.
    """

    # Lossless mode — only deterministic transformations
    lossless = PolicyDecision(
        mode=CompressionMode.LOSSLESS,
        strategies=[StrategyName.RULE_BASED, StrategyName.CANONICALIZE],
        target_reduction=0.10,
        max_degradation=0.01,
    )
    result = await pipeline.compress(text, lossless)
    print(f"LOSSLESS: {result.original_tokens} -> {result.compressed_tokens} "
          f"({result.overall_ratio:.1%} reduction)")

    # Balanced mode
    balanced = PolicyDecision(
        mode=CompressionMode.BALANCED,
        strategies=[
            StrategyName.RULE_BASED,
            StrategyName.CANONICALIZE,
            StrategyName.STRUCTURED,
        ],
        target_reduction=0.30,
        max_degradation=0.05,
    )
    result = await pipeline.compress(text, balanced)
    print(f"BALANCED: {result.original_tokens} -> {result.compressed_tokens} "
          f"({result.overall_ratio:.1%} reduction)")

    # Aggressive mode
    aggressive = PolicyDecision(
        mode=CompressionMode.AGGRESSIVE,
        strategies=[
            StrategyName.RULE_BASED,
            StrategyName.CANONICALIZE,
            StrategyName.STRUCTURED,
            StrategyName.EMBEDDING,
        ],
        target_reduction=0.60,
        max_degradation=0.15,
    )
    result = await pipeline.compress(text, aggressive)
    print(f"AGGRESSIVE: {result.original_tokens} -> {result.compressed_tokens} "
          f"({result.overall_ratio:.1%} reduction)")
    print(f"\nCompressed text:\n{result.compressed_text}")


if __name__ == "__main__":
    asyncio.run(main())
