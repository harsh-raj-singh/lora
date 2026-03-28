"""Lightweight training loop for LoRA fine-tuning."""

import math
import time
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup

from .config import ExperimentConfig
from .inject import count_parameters

logger = logging.getLogger(__name__)


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: ExperimentConfig,
    tokenizer=None,
) -> dict:
    """Train LoRA adapters on a language modeling task.

    Returns a metrics dict with training loss curve and eval perplexity.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # Gather only trainable (LoRA) parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    total_steps = (
        len(train_loader)
        * config.train.num_epochs
        // config.train.gradient_accumulation_steps
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.train.warmup_steps,
        num_training_steps=total_steps,
    )

    param_info = count_parameters(model)
    logger.info(f"Trainable params: {param_info['trainable']:,} / {param_info['total']:,} ({param_info['trainable_pct']:.2f}%)")

    metrics = {"train_loss": [], "eval_loss": [], "eval_perplexity": [], "lr": []}
    global_step = 0
    start_time = time.time()

    for epoch in range(config.train.num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

            # Shift for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            loss = loss / config.train.gradient_accumulation_steps
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, config.train.max_grad_norm)

            if (step + 1) % config.train.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                metrics["train_loss"].append(loss.item() * config.train.gradient_accumulation_steps)
                metrics["lr"].append(scheduler.get_last_lr()[0])

                if global_step % config.train.log_every == 0:
                    elapsed = time.time() - start_time
                    avg_loss = sum(metrics["train_loss"][-config.train.log_every:]) / len(metrics["train_loss"][-config.train.log_every:])
                    logger.info(
                        f"Epoch {epoch+1} | Step {global_step} | "
                        f"Loss {avg_loss:.4f} | LR {scheduler.get_last_lr()[0]:.2e} | "
                        f"Elapsed {elapsed:.0f}s"
                    )

            epoch_loss += loss.item() * config.train.gradient_accumulation_steps

        avg_epoch_loss = epoch_loss / len(train_loader)

        # Validation
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            ppl = math.exp(min(val_loss, 20))  # cap to avoid overflow
            metrics["eval_loss"].append(val_loss)
            metrics["eval_perplexity"].append(ppl)
            logger.info(
                f"Epoch {epoch+1} | Train Loss {avg_epoch_loss:.4f} | "
                f"Val Loss {val_loss:.4f} | Val PPL {ppl:.2f}"
            )

    total_time = time.time() - start_time
    metrics["total_time_seconds"] = total_time
    metrics["parameter_info"] = param_info
    metrics["config"] = {
        "rank": config.lora.rank,
        "alpha": config.lora.alpha,
        "lr": config.train.learning_rate,
        "epochs": config.train.num_epochs,
    }

    # Save metrics
    out_dir = Path(config.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Training complete in {total_time:.1f}s. Metrics saved to {out_dir / 'metrics.json'}")
    return metrics


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Compute average cross-entropy loss on a dataset."""
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        total_loss += loss.item()

    return total_loss / len(dataloader)
