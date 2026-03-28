"""
Benchmark: compare full fine-tuning vs LoRA on parameter count, memory, and throughput.

Usage:
    python -m experiments.benchmark
"""

import json
import logging
import time
from pathlib import Path

import torch
from transformers import GPT2Model

from lora_from_scratch import inject_lora, count_parameters, merge_lora

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def measure_forward(model, batch_size=4, seq_len=128, n_iters=50):
    """Measure average forward pass time."""
    dummy = torch.randn(batch_size, seq_len, 768)
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            start = time.perf_counter()
            _ = model(dummy)
            times.append(time.perf_counter() - start)
    avg_ms = sum(times) / len(times) * 1000
    return avg_ms


def measure_memory(model, batch_size=4, seq_len=128):
    """Estimate model memory footprint in MB."""
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    return param_mem


def run_benchmark(output_path: str = "results/benchmark.json"):
    results = {}
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    logger.info(f"Device: {device_name}")

    # Full model baseline
    logger.info("Benchmarking full GPT-2...")
    model_full = GPT2Model.from_pretrained("gpt2")
    info_full = count_parameters(model_full)  # will be 100% trainable since no LoRA
    fwd_full = measure_forward(model_full)
    mem_full = measure_memory(model_full)

    results["full_finetuning"] = {
        "total_params": info_full["total"],
        "trainable_params": info_full["total"],
        "trainable_pct": 100.0,
        "forward_ms": fwd_full,
        "model_size_mb": mem_full,
    }
    del model_full

    # LoRA variants
    for rank in [4, 8, 16, 32]:
        logger.info(f"Benchmarking LoRA rank={rank}...")
        model_lora = GPT2Model.from_pretrained("gpt2")
        model_lora = inject_lora(model_lora, rank=rank, alpha=2 * rank, verbose=False)
        info_lora = count_parameters(model_lora)
        fwd_lora = measure_forward(model_lora)
        mem_lora = measure_memory(model_lora)

        # Merged inference
        merge_lora(model_lora)
        fwd_merged = measure_forward(model_lora)
        mem_merged = measure_memory(model_lora)

        results[f"lora_r{rank}"] = {
            "rank": rank,
            "total_params": info_lora["total"],
            "trainable_params": info_lora["trainable"],
            "trainable_pct": info_lora["trainable_pct"],
            "forward_ms": fwd_lora,
            "forward_merged_ms": fwd_merged,
            "model_size_mb": mem_lora,
            "model_merged_size_mb": mem_merged,
        }
        del model_lora

    results["device"] = device_name

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Print comparison table
    print("\n" + "=" * 100)
    print(f"{'Method':>20} | {'Params':>12} | {'Train %':>10} | {'Fwd (ms)':>10} | {'Merged (ms)':>12} | {'Size (MB)':>10}")
    print("-" * 100)
    ff = results["full_finetuning"]
    print(f"{'Full Fine-tuning':>20} | {ff['total_params']:>12,} | {'100.000':>10}% | {ff['forward_ms']:>9.2f} | {'N/A':>12} | {ff['model_size_mb']:>9.1f}")
    for rank in [4, 8, 16, 32]:
        r = results[f"lora_r{rank}"]
        print(
            f"{'LoRA r=' + str(rank):>20} | {r['total_params']:>12,} | {r['trainable_pct']:>9.3f}% | "
            f"{r['forward_ms']:>9.2f} | {r['forward_merged_ms']:>11.2f} | {r['model_size_mb']:>9.1f}"
        )
    print("=" * 100)


if __name__ == "__main__":
    run_benchmark()
