from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FusOn-DB from Hugging Face.")
    parser.add_argument("--repo", default="ChatterjeeLab/FusOn-DB")
    parser.add_argument("--output", default="data/raw/fuson_db")
    parser.add_argument("--split", default=None, help="Optional Hugging Face split.")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.repo, split=args.split)

    if isinstance(dataset, DatasetDict):
        dataset.save_to_disk(str(output))
        print({name: len(split) for name, split in dataset.items()})
    else:
        dataset.save_to_disk(str(output))
        print({"rows": len(dataset), "columns": dataset.column_names})


if __name__ == "__main__":
    main()

