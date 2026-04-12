# Contributing to TwoTrim

Thank you for your interest in contributing to TwoTrim! We welcome community contributions to help make AI context compression more efficient, accurate, and accessible for everyone.

## Development Workflow

1. **Fork the Repository**: Clone your fork locally to get started.
2. **Install Dependencies**: 
   ```bash
   pip install -e .[dev]
   ```
3. **Create a Branch**: Create a feature branch (`git checkout -b feature/your-feature-name`).
4. **Make Changes**: Keep your commits focused and logical.
5. **Run the Tests**: 
   ```bash
   python -m pytest
   ```
   Ensure all 50+ backend evaluation tests pass. If you are modifying the core compression algorithms, you **must** run the benchmark suite against `gsm8k` or `longbench` to prove you haven't degraded accuracy.
   ```bash
   python benchmarks/runner.py --limit 10
   ```
6. **Submit a Pull Request**: Detail the exact logic behind any algorithmic changes and include your benchmark delta scores if applicable!

## Code Style

- We follow standard PEP 8 guidelines.
- Use strict type hinting wherever possible.
- If you touch `src/twotrim/compression/semantic.py`, be highly cautious of mathematical weighting variables that could silently shift LLM retention logic. 

## Reporting Bugs

Please use the GitHub Issue Tracker. Include your base `config.yaml` overrides and the dataset/prompt size that provoked the failure. 
