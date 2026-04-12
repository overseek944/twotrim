"""TwoTrim CLI — command-line interface.

Commands:
    twotrim serve     — Start the proxy server
    twotrim compress  — Compress a single prompt
    twotrim stats     — Show compression statistics
    twotrim health    — Check proxy health
    twotrim config    — Show effective configuration
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

@click.group()
@click.version_option(version="0.1.0", prog_name="twotrim")
def cli() -> None:
    """TwoTrim — Universal Token Compression Fabric for LLMs."""
    pass


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), default=None,
              help="Path to config YAML file")
@click.option("--host", "-h", default=None, help="Bind host")
@click.option("--port", "-p", type=int, default=None, help="Bind port")
@click.option("--workers", "-w", type=int, default=None, help="Number of workers")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default=None, help="Log level")
def serve(
    config: str | None,
    host: str | None,
    port: int | None,
    workers: int | None,
    log_level: str | None,
) -> None:
    """Start the TwoTrim proxy server."""
    import uvicorn
    from twotrim.config import load_config

    cfg = load_config(config)

    bind_host = host or cfg.server.host
    bind_port = port or cfg.server.port
    n_workers = workers or cfg.server.workers
    level = (log_level or cfg.observability.logging.level).lower()

    click.echo(f"Starting TwoTrim proxy on {bind_host}:{bind_port}")
    click.echo(f"  Mode: {cfg.compression.mode}")
    click.echo(f"  Upstream: {cfg.upstream.default_base_url}")
    click.echo(f"  Workers: {n_workers}")

    uvicorn.run(
        "twotrim.interceptor.proxy:create_app",
        host=bind_host,
        port=bind_port,
        workers=n_workers,
        log_level=level,
        factory=True,
    )


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--input", "-i", "input_file", type=click.Path(exists=True),
              help="Input file to compress")
@click.option("--text", "-t", type=str, help="Text to compress (inline)")
@click.option("--mode", "-m", type=click.Choice(["lossless", "balanced", "aggressive"]),
              default="balanced", help="Compression mode")
@click.option("--config", "-c", type=click.Path(exists=True), default=None)
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output file (default: stdout)")
@click.option("--json-output", is_flag=True, help="Output as JSON with metadata")
def compress(
    input_file: str | None,
    text: str | None,
    mode: str,
    config: str | None,
    output: str | None,
    json_output: bool,
) -> None:
    """Compress a single prompt or text file."""
    from twotrim.config import load_config
    from twotrim.compression.pipeline import get_pipeline
    from twotrim.policy.profiles import get_profile
    from twotrim.types import CompressionMode

    load_config(config)

    # Get input text
    if input_file:
        content = Path(input_file).read_text(encoding="utf-8")
    elif text:
        content = text
    elif not sys.stdin.isatty():
        content = sys.stdin.read()
    else:
        click.echo("Error: provide --input, --text, or pipe via stdin", err=True)
        sys.exit(1)

    pipeline = get_pipeline()
    decision = get_profile(CompressionMode(mode))

    result = asyncio.run(pipeline.compress(content, decision))

    if json_output:
        out = {
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "compression_ratio": round(result.overall_ratio, 4),
            "compression_time_ms": round(result.compression_time_ms, 2),
            "strategies": [
                {
                    "name": s.strategy.value,
                    "ratio": round(s.compression_ratio, 4),
                }
                for s in result.strategies_applied
            ],
            "compressed_text": result.compressed_text,
        }
        text_out = json.dumps(out, indent=2)
    else:
        text_out = result.compressed_text

    if output:
        Path(output).write_text(text_out, encoding="utf-8")
        click.echo(f"Written to {output}")
        click.echo(f"  Original:   {result.original_tokens} tokens")
        click.echo(f"  Compressed: {result.compressed_tokens} tokens")
        click.echo(f"  Ratio:      {result.overall_ratio:.1%}")
        click.echo(f"  Time:       {result.compression_time_ms:.1f}ms")
    else:
        click.echo(text_out)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--proxy", "-p", default="http://localhost:8000",
              help="Proxy URL")
@click.option("--last", "-n", type=int, default=0,
              help="Show last N request metrics")
def stats(proxy: str, last: int) -> None:
    """Show compression statistics from a running proxy."""
    import httpx

    try:
        with httpx.Client() as client:
            if last > 0:
                resp = client.get(f"{proxy}/stats/recent?n={last}", timeout=10)
                data = resp.json()
                for item in data:
                    click.echo(
                        f"  [{item['request_id']}] {item['model']} "
                        f"ratio={item['compression_ratio']:.1%} "
                        f"saved={item['tokens_saved']} tokens "
                        f"cost=${item['estimated_cost_saved_usd']:.4f} "
                        f"time={item['compression_time_ms']:.0f}ms"
                    )
            else:
                resp = client.get(f"{proxy}/stats", timeout=10)
                data = resp.json()
                click.echo("TwoTrim Aggregate Statistics")
                click.echo(f"  Total requests:     {data.get('total_requests', 0)}")
                click.echo(f"  Tokens original:    {data.get('total_tokens_original', 0):,}")
                click.echo(f"  Tokens compressed:  {data.get('total_tokens_compressed', 0):,}")
                click.echo(f"  Tokens saved:       {data.get('total_tokens_saved', 0):,}")
                click.echo(f"  Cost saved:         ${data.get('total_cost_saved_usd', 0):.4f}")
                click.echo(f"  Avg ratio:          {data.get('avg_compression_ratio', 0):.1%}")
                click.echo(f"  Cache hit rate:     {data.get('cache_hit_rate', 0):.1%}")
                click.echo(f"  Avg quality:        {data.get('avg_quality_score', 0):.3f}")
                click.echo(f"  Avg latency:        {data.get('avg_compression_time_ms', 0):.1f}ms")

                usage = data.get("strategy_usage", {})
                if usage:
                    click.echo("  Strategy usage:")
                    for strategy, count in sorted(usage.items(), key=lambda x: -x[1]):
                        click.echo(f"    {strategy}: {count}")
    except httpx.ConnectError:
        click.echo(f"Error: cannot connect to proxy at {proxy}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--proxy", "-p", default="http://localhost:8000")
def health(proxy: str) -> None:
    """Check proxy health."""
    import httpx

    try:
        with httpx.Client() as client:
            resp = client.get(f"{proxy}/health", timeout=5)
            data = resp.json()
            click.echo(f"Status: {data.get('status', 'unknown')}")
            click.echo(f"Version: {data.get('version', 'unknown')}")
    except httpx.ConnectError:
        click.echo(f"Error: proxy at {proxy} is not reachable", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@cli.command("config")
@click.option("--config", "-c", type=click.Path(exists=True), default=None)
def show_config(config: str | None) -> None:
    """Show effective configuration."""
    from twotrim.config import load_config
    cfg = load_config(config)
    click.echo(json.dumps(cfg.model_dump(), indent=2, default=str))


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--results", "-r", type=click.Path(exists=True),
              default=".twotrim/eval_results.jsonl",
              help="Path to evaluation results JSONL")
@click.option("--last", "-n", type=int, default=100)
def evaluate(results: str, last: int) -> None:
    """Show evaluation results summary."""
    p = Path(results)
    if not p.exists():
        click.echo(f"No results file found at {results}", err=True)
        sys.exit(1)

    lines = p.read_text().strip().split("\n")
    entries = [json.loads(l) for l in lines[-last:]]

    if not entries:
        click.echo("No evaluation results found.")
        return

    scores = [e["similarity_score"] for e in entries]
    passed = sum(1 for e in entries if e["passed"])

    click.echo(f"Evaluation Summary (last {len(entries)} results)")
    click.echo(f"  Pass rate:      {passed}/{len(entries)} ({passed/len(entries):.1%})")
    click.echo(f"  Avg similarity: {sum(scores)/len(scores):.3f}")
    click.echo(f"  Min similarity: {min(scores):.3f}")
    click.echo(f"  Max similarity: {max(scores):.3f}")

    ratios = [e["compression_ratio"] for e in entries]
    click.echo(f"  Avg compression: {sum(ratios)/len(ratios):.1%}")


@cli.command("benchmark")
@click.option("--dataset", type=str, default="gsm8k", help="Dataset name (gsm8k, longbench, or custom)")
@click.option("--limit", type=int, default=10, help="Number of samples to run")
@click.option("--model", type=str, default="gpt-4o-mini", help="Model to evaluate against")
@click.option("--data-path", type=click.Path(exists=True), default=None, help="Path to a local .jsonl dataset file")
def benchmark(dataset: str, limit: int, model: str, data_path: str | None) -> None:
    """Run standardized compression benchmarks."""
    import os
    import time
    # Increase HF timeout globally for this command
    os.environ["HF_HUB_HTTP_TIMEOUT"] = "300"
    
    console.print(f"[bold blue]Running Benchmark: {dataset.upper()}[/] (limit={limit}, model={model})")
    
    try:
        import sys
        from pathlib import Path
        # Add repo root to sys.path
        repo_root = Path(__file__).parent.parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
            
        from benchmarks.runner import BenchmarkRunner
    except ImportError as e:
        console.print(f"[red]Error:[/] benchmarks package not found ({e}). Ensure you are running from the source tree.")
        raise typer.Exit(1)
        
    runner = BenchmarkRunner(model=model)
    
    # Pre-load dataset to avoid connection spam
    console.print("[yellow]Attempting to load dataset...[/]")
    is_local = False
    
    # If a data path is provided, use the manual loader immediately
    if data_path:
        console.print(f"[green]Loading manual dataset from {data_path}...[/]")
        from benchmarks.datasets.manual_loader import ManualDataset
        ds = ManualDataset(dataset_type=dataset, data_path=data_path)
        samples = ds.load(limit=limit)
    else:
        for attempt in range(3):
            try:
                if attempt == 0:
                    # First attempt: Try standard HF
                    if dataset == "gsm8k":
                        from benchmarks.datasets.gsm8k import GSM8KDataset
                        ds = GSM8KDataset()
                    elif dataset == "longbench":
                        from benchmarks.datasets.longbench import LongBenchDataset
                        ds = LongBenchDataset()
                elif attempt == 1:
                    # Second attempt: Try mirror
                    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                    console.print("[yellow]Attempt 2: Retrying via mirror...[/]")
                
                samples = ds.load(limit=limit)
                console.print("[green]Dataset loaded successfully from HuggingFace.[/]")
                break
            except Exception as e:
                if attempt < 2:
                    console.print(f"[dim]HuggingFace attempt {attempt+1} unsuccessful.[/]")
                    time.sleep(2)
                    continue
                
                # Final fallback: Use local tiny data
                console.print(f"[yellow]HuggingFace unavailable. Falling back to local tiny dataset...[/]")
                from benchmarks.datasets.local import LocalDataset
                ds = LocalDataset(dataset_type=dataset)
                samples = ds.load(limit=limit)
                is_local = True
                break
            
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Running baseline...", total=1)
        baseline = runner.run(dataset, ds, samples, mode="baseline")
        
        progress.update(task, description="[yellow]Running lossless mode...")
        lossless = runner.run(dataset, ds, samples, mode="lossless")
        
        progress.update(task, description="[magenta]Running aggressive mode...")
        aggressive = runner.run(dataset, ds, samples, mode="aggressive")
        
    from rich.table import Table
    table = Table(title=f"Benchmark Results: {dataset.upper()}")
    table.add_column("Mode", justify="left", style="cyan", no_wrap=True)
    table.add_column("Avg Score", justify="right", style="green")
    table.add_column("Avg Reduction", justify="right", style="magenta")
    table.add_column("Avg Latency (ms)", justify="right", style="yellow")
    
    def format_row(res):
        return [
            res.mode.capitalize(),
            f"{res.avg_score:.2f}",
            f"{res.avg_compression_ratio*100:.1f}%",
            f"{res.avg_latency_ms:.0f}"
        ]
        
    table.add_row(*format_row(baseline))
    table.add_row(*format_row(lossless))
    table.add_row(*format_row(aggressive))
    
    console.print(table)


if __name__ == "__main__":
    cli()
