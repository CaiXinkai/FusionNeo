from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


def clean_sequence(sequence: str) -> str:
    return AA_RE.sub("", str(sequence).upper().replace("*", ""))


def split_fusion_genes(value: str) -> tuple[str, str]:
    text = str(value).strip()
    for sep in ("::", "--", "–", "-", "/"):
        if sep in text:
            head, tail = text.split(sep, 1)
            return head.strip(), tail.strip()
    raise ValueError(f"Cannot split fusion gene pair: {value!r}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

