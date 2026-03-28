"""
Rank ablation study: sweep LoRA rank and measure parameter count & perplexity.

Usage:
    python -m experiments.rank_ablation --ranks 1 2 4 8 16 32 64
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from transformers import GPT2Model

from lora_from_scratch import inject_lora, count_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_ablation(ranks: list[int], output_path: str = "results/rank_ablation.json"):
    results = []
    total_params = None

    for rank in ranks:
        logger.info(f"--- Rank = {rank} ---")
        model = GPT2Model.from_pretrained("gpt2")
        model = inject_lora(model, rank=rank, alpha=2 * rank)

        info = count_parameters(model)
        if total_params is None:
            total_params = info["total"]

        # Simulate a forward pass to measure memory
        dummy = torch.randn(1, 8, 768)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        elapsed = time.perf_counter() - start

        result = {
            "rank": rank,
            "lora_params": info["lora_params"],
            "trainable_params": info["trainable"],
            "total_params": info["total"],
            "trainable_pct": info["trainable_pct"],
            "param_reduction_ratio": info["total"] / info["trainable"] if info["trainable"] > 0 else float("inf"),
            "forward_time_ms": elapsed * 1000,
        }
        results.append(result)
        logger.info(
            f"  Params: {info['trainable']:,} ({info['trainable_pct']:.3f}%) | "
            f"Reduction: {info['total']//info['trainable'] if info['trainable'] > 0 else 'N/A'}x | "
            f"Forward: {elapsed*1000:.2f}ms"
        )

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    out = {"total_model_params": total_params, "model": "gpt2", "ablation": results}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Rank':>6} | {'LoRA Params':>14} | {'Trainable %':>12} | {'Reduction':>10} | {'Fwd (ms)':>10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['rank']:>6} | {r['lora_params']:>14,} | {r['trainable_pct']:>11.3f}% | "
            f"{r['param_reduction_ratio']:>9.0f}x | {r['forward_time_ms']:>9.2f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--output", type=str, default="results/rank_ablation.json")
    args = parser.parse_args()
    run_ablation(args.ranks, args.output)
