from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def split_value(group: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if value < 0.8 else ("validation" if value < 0.9 else "test")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize an IEDB/SysteMHC/CEDAR export into a benchmark table."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--peptide-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--positive-value", required=True)
    parser.add_argument("--allele-column", default=None)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = Path(args.input)
    frame = pd.read_csv(path, sep="\t" if path.suffix in {".tsv", ".txt"} else ",")
    peptide = frame[args.peptide_column].astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    valid = peptide.map(lambda value: 8 <= len(value) <= 25 and set(value) <= VALID_AA)
    frame = frame.loc[valid].copy()
    frame["peptide"] = peptide.loc[valid]
    frame["label"] = (
        frame[args.label_column].astype(str).str.casefold()
        == str(args.positive_value).casefold()
    ).astype(int)
    if args.allele_column:
        frame["allele"] = frame[args.allele_column].fillna("unknown").astype(str)
    else:
        frame["allele"] = "unknown"
    group = (
        frame[args.group_column].astype(str)
        if args.group_column
        else frame["peptide"].astype(str)
    )
    frame["split"] = group.map(lambda value: split_value(value, args.seed))
    keep = ["peptide", "label", "allele", "split"]
    frame[keep].drop_duplicates().to_csv(args.output, index=False)
    print(frame[keep]["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()

