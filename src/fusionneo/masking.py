from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def cosine_rate(step_in_epoch: int, steps_per_epoch: int, min_rate: float, max_rate: float) -> float:
    if steps_per_epoch <= 1:
        return max_rate
    progress = min(max(step_in_epoch / (steps_per_epoch - 1), 0.0), 1.0)
    return min_rate + 0.5 * (max_rate - min_rate) * (1.0 - math.cos(math.pi * progress))


def junction_weights(
    length: int, junction: int, boost: float = 4.0, tau: float = 6.0
) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32)
    distance = torch.abs(positions - float(junction))
    return 1.0 + boost * torch.exp(-distance / tau)


@dataclass
class JunctionMaskingCollator:
    tokenizer: object
    strategy: str = "cosine_junction"
    min_rate: float = 0.15
    max_rate: float = 0.40
    fixed_rate: float = 0.15
    junction_boost: float = 4.0
    junction_tau: float = 6.0
    steps_per_epoch: int = 100

    def __post_init__(self) -> None:
        self.call_count = 0
        if self.strategy not in {"random_fixed", "cosine_uniform", "cosine_junction"}:
            raise ValueError(f"Unknown masking strategy: {self.strategy}")

    def _rate(self) -> float:
        if self.strategy == "random_fixed":
            return self.fixed_rate
        return cosine_rate(
            self.call_count % max(self.steps_per_epoch, 1),
            self.steps_per_epoch,
            self.min_rate,
            self.max_rate,
        )

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        features = [dict(feature) for feature in features]
        junctions = [int(feature.pop("junction_token")) for feature in features]
        batch = self.tokenizer.pad(features, return_tensors="pt")
        input_ids = batch["input_ids"].clone()
        labels = torch.full_like(input_ids, -100)
        rate = self._rate()
        self.call_count += 1

        special_mask = torch.tensor(
            [
                self.tokenizer.get_special_tokens_mask(row.tolist(), already_has_special_tokens=True)
                for row in input_ids
            ],
            dtype=torch.bool,
        )
        valid_mask = (~special_mask) & batch["attention_mask"].bool()

        for row_index in range(input_ids.shape[0]):
            valid_positions = torch.where(valid_mask[row_index])[0]
            if len(valid_positions) == 0:
                continue
            n_mask = max(1, round(rate * len(valid_positions)))
            if self.strategy == "cosine_junction":
                weights = junction_weights(
                    input_ids.shape[1],
                    junctions[row_index],
                    self.junction_boost,
                    self.junction_tau,
                )[valid_positions]
            else:
                weights = torch.ones(len(valid_positions), dtype=torch.float32)
            chosen_index = torch.multinomial(weights, min(n_mask, len(valid_positions)), False)
            chosen = valid_positions[chosen_index]
            labels[row_index, chosen] = input_ids[row_index, chosen]
            input_ids[row_index, chosen] = self.tokenizer.mask_token_id

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch
