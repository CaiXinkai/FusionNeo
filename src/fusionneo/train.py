from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .common import load_yaml, set_seed
from .masking import JunctionMaskingCollator


def freeze_model(model: torch.nn.Module, unfreeze_last_n: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    candidates = [
        getattr(getattr(model, "esm", None), "encoder", None),
        getattr(getattr(model, "base_model", None), "encoder", None),
    ]
    encoder = next((candidate for candidate in candidates if candidate is not None), None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise ValueError("Could not locate transformer layers in the selected model")
    for layer in layers[-unfreeze_last_n:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for name in ("lm_head", "classifier"):
        module = getattr(model, name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ESM with junction-aware masking.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--contexts", default=None)
    parser.add_argument("--strategy", choices=["random_fixed", "cosine_uniform", "cosine_junction"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    contexts_path = args.contexts or str(Path(cfg["data"]["prepared_dir"]) / "contexts.parquet")
    model_name = args.model or cfg["model"]["name"]
    output_dir = args.output or cfg["training"]["output_dir"]
    strategy = args.strategy or cfg["masking"]["strategy"]

    frame = pd.read_parquet(contexts_path)
    if args.smoke_test:
        frame = frame.groupby("split", group_keys=False).head(16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    freeze_model(model, int(cfg["model"]["unfreeze_last_n_layers"]))

    def tokenize(row: dict) -> dict:
        encoded = tokenizer(
            row["context_seq"],
            truncation=True,
            max_length=int(cfg["model"]["max_length"]),
            add_special_tokens=True,
        )
        # ESM adds a leading CLS token, hence +1.
        encoded["junction_token"] = min(row["context_junction"] + 1, len(encoded["input_ids"]) - 2)
        return encoded

    datasets = {}
    for split in ("train", "validation", "test"):
        split_frame = frame[frame["split"] == split][["context_seq", "context_junction"]]
        dataset = Dataset.from_pandas(split_frame, preserve_index=False)
        datasets[split] = dataset.map(tokenize, remove_columns=dataset.column_names)

    batch_size = int(cfg["training"]["batch_size"])
    steps_per_epoch = max(1, math.ceil(len(datasets["train"]) / batch_size))
    mask_cfg = cfg["masking"]
    collator = JunctionMaskingCollator(
        tokenizer=tokenizer,
        strategy=strategy,
        min_rate=float(mask_cfg["min_rate"]),
        max_rate=float(mask_cfg["max_rate"]),
        fixed_rate=float(mask_cfg["fixed_rate"]),
        junction_boost=float(mask_cfg["junction_boost"]),
        junction_tau=float(mask_cfg["junction_tau"]),
        steps_per_epoch=steps_per_epoch,
    )
    train_cfg = cfg["training"]
    use_fp16 = bool(train_cfg["fp16"]) and torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1 if args.smoke_test else float(train_cfg["epochs"]),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        fp16=use_fp16,
        logging_steps=1 if args.smoke_test else int(train_cfg["logging_steps"]),
        save_strategy=train_cfg["save_strategy"],
        evaluation_strategy=train_cfg["eval_strategy"],
        report_to="none",
        remove_unused_columns=False,
        seed=int(cfg["seed"]),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics = trainer.evaluate(datasets["test"])
    metrics["test_pseudo_perplexity"] = math.exp(metrics["eval_loss"])
    trainer.save_metrics("test", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
