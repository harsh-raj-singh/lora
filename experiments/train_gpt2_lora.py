"""
Fine-tune GPT-2 with LoRA on WikiText-2.

Usage:
    python -m experiments.train_gpt2_lora --rank 8 --epochs 3
"""

import argparse
import logging
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Model

from lora_from_scratch import inject_lora, train, count_parameters, ExperimentConfig, LoRAConfig, TrainConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_dataset(tokenizer, dataset, max_length):
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


def create_dataloader(tokenized_dataset, batch_size, shuffle=True):
    from torch.utils.data import DataLoader

    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 with LoRA")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results/checkpoints", help="Output directory")
    args = parser.parse_args()

    set_seed(args.seed)

    # Config
    config = ExperimentConfig(
        lora=LoRAConfig(rank=args.rank, alpha=args.alpha, dropout=args.dropout),
        train=TrainConfig(
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
            output_dir=args.output_dir,
        ),
    )

    # Load model & tokenizer
    logger.info(f"Loading {config.train.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.train.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(config.train.model_name)

    # Inject LoRA
    logger.info("Injecting LoRA adapters...")
    model = inject_lora(
        model,
        target_modules=config.lora.target_modules,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
    )

    param_info = count_parameters(model)
    logger.info(f"Total params:     {param_info['total']:,}")
    logger.info(f"Trainable params: {param_info['trainable']:,} ({param_info['trainable_pct']:.2f}%)")
    logger.info(f"LoRA params:      {param_info['lora_params']:,}")

    # Load dataset
    logger.info(f"Loading {config.train.dataset}:{config.train.dataset_config}...")
    dataset = load_dataset(config.train.dataset, config.train.dataset_config)
    train_dataset = tokenize_dataset(tokenizer, dataset["train"], config.train.max_seq_length)
    val_dataset = tokenize_dataset(tokenizer, dataset["validation"], config.train.max_seq_length)

    train_loader = create_dataloader(train_dataset, config.train.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, config.train.batch_size, shuffle=False)

    # Train
    logger.info("Starting training...")
    metrics = train(model, train_loader, val_loader, config, tokenizer)

    logger.info(f"Final eval perplexity: {metrics['eval_perplexity'][-1]:.2f}")
    logger.info(f"Total training time: {metrics['total_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
