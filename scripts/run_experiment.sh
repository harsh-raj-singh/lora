#!/usr/bin/env bash
set -euo pipefail

# LoRA From Scratch — Experiment Runner
# Usage: bash scripts/run_experiment.sh

echo "=== LoRA From Scratch: GPT-2 Fine-tuning on WikiText-2 ==="
echo ""

RANK=${1:-8}
EPOCHS=${2:-3}
LR=${3:-3e-4}

echo "Config: rank=$RANK, epochs=$EPOCHS, lr=$LR"
echo ""

# Rank ablation study
echo "--- Running Rank Ablation Study ---"
python -m experiments.rank_ablation --ranks 1 2 4 8 16 32 64
echo ""

# Full training run
echo "--- Training LoRA (rank=$RANK) ---"
python -m experiments.train_gpt2_lora --rank $RANK --epochs $EPOCHS --lr $LR
echo ""

# Benchmark
echo "--- Benchmarking ---"
python -m experiments.benchmark
echo ""

echo "=== All experiments complete. Check results/ directory. ==="
