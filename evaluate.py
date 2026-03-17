"""
evaluate.py — Test the quality of your RAG system.

Run this after building the system to measure how good it is.
This is what separates juniors from engineers — you can MEASURE your work.

Metrics we check:
    - Does it answer? (response rate)
    - Does it cite sources? (grounding rate)  
    - How fast is it? (latency)
    - Does it handle edge cases? (robustness)

At Siemens interview: "How do you know your system works?"
Answer: "I measured it. Here are the numbers."
"""

import time
import json
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table

from rag_pipeline import RAGPipeline

console = Console()

# ── Test questions ─────────────────────────────────────────────
# These test different capabilities of the system
TEST_CASES = [
    {
        "question": "What is the main purpose of this document?",
        "category": "General",
        "expect_answer": True,
    },
    {
        "question": "What file formats are mentioned?",
        "category": "Specific fact",
        "expect_answer": True,
    },
    {
        "question": "Explain the most important technical requirement.",
        "category": "Technical",
        "expect_answer": True,
    },
    {
        "question": "What is 2 + 2?",
        "category": "Out of scope",
        "expect_answer": False,  # Should say "not in documentation"
    },
    {
        "question": "",
        "category": "Edge case: empty",
        "expect_answer": False,
    },
    {
        "question": "ما هو الهدف الرئيسي من هذا الوثيقة؟",  # Arabic
        "category": "Arabic query",
        "expect_answer": True,
    },
]


def run_evaluation():
    console.print("\n[bold cyan]Fasih-Docs — System Evaluation[/bold cyan]\n")

    pipeline = RAGPipeline()

    if not pipeline.is_ready:
        console.print("[red]Pipeline not ready. Run ingest.py first.[/red]")
        return

    results = []

    for case in TEST_CASES:
        question = case["question"]
        category = case["category"]

        console.print(f"Testing: [yellow]{category}[/yellow]")

        start = time.time()
        result = pipeline.query(question)
        latency = round(time.time() - start, 2)

        answer = result["answer"]
        sources = result["sources"]
        has_answer = not result["error"] and len(answer) > 20
        has_sources = len(sources) > 0
        not_hallucinating = "not available in the loaded" in answer.lower() or has_sources

        results.append({
            "category": category,
            "has_answer": has_answer,
            "has_sources": has_sources,
            "latency_s": latency,
            "answer_preview": answer[:80] + "..." if len(answer) > 80 else answer,
        })

    # ── Print results table ────────────────────────────────────
    table = Table(title="Evaluation Results", show_header=True)
    table.add_column("Category", style="cyan", width=20)
    table.add_column("Answered", justify="center", width=10)
    table.add_column("Has Sources", justify="center", width=12)
    table.add_column("Latency (s)", justify="center", width=12)
    table.add_column("Preview", width=40)

    for r in results:
        table.add_row(
            r["category"],
            "[green]Yes[/green]" if r["has_answer"] else "[red]No[/red]",
            "[green]Yes[/green]" if r["has_sources"] else "[yellow]No[/yellow]",
            str(r["latency_s"]),
            r["answer_preview"],
        )

    console.print(table)

    # ── Summary stats ──────────────────────────────────────────
    answered = sum(1 for r in results if r["has_answer"])
    sourced = sum(1 for r in results if r["has_sources"])
    avg_latency = round(sum(r["latency_s"] for r in results) / len(results), 2)

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Response rate:    [green]{answered}/{len(results)}[/green]")
    console.print(f"  Source citation:  [green]{sourced}/{len(results)}[/green]")
    console.print(f"  Avg latency:      [cyan]{avg_latency}s[/cyan]")

    # Save results to JSON (good for your GitHub README)
    report = {
        "timestamp": datetime.now().isoformat(),
        "response_rate": f"{answered}/{len(results)}",
        "source_citation_rate": f"{sourced}/{len(results)}",
        "avg_latency_seconds": avg_latency,
        "test_cases": results,
    }

    Path("logs").mkdir(exist_ok=True)
    with open("logs/eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]Full report saved to logs/eval_report.json[/green]")
    console.print("[dim]Use these numbers in your CV and GitHub README.[/dim]")


if __name__ == "__main__":
    run_evaluation()
