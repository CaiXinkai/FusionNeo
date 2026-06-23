from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

from .breakpoints import BreakpointCall, infer_breakpoint
from .common import clean_sequence, split_fusion_genes, write_json
from .peptides import (
    enumerate_junction_peptides,
    enumerate_nonjunction_background,
    enumerate_protein_background,
    junction_context,
    peptide_context,
)
from .splits import stable_group_split

CONFIDENCE = {"low": 0, "medium": 1, "high": 2}


def _as_dataset(obj: Dataset | DatasetDict) -> Dataset:
    if isinstance(obj, DatasetDict):
        return obj[next(iter(obj.keys()))]
    return obj


def load_reference_fasta(path: str) -> dict[str, str]:
    return {
        record.id.split("|")[0]: clean_sequence(str(record.seq))
        for record in SeqIO.parse(path, "fasta")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer breakpoints and create FusionNeo datasets.")
    parser.add_argument("--dataset", default="data/raw/fuson_db")
    parser.add_argument("--reference-fasta", required=True)
    parser.add_argument(
        "--breakpoint-table",
        default=None,
        help="Optional TSV/CSV with seq_id and zero-based breakpoint columns; overrides alignment.",
    )
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--flank", type=int, default=50)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8, 9, 10, 11, 12, 13, 14])
    parser.add_argument("--min-side-residues", type=int, default=1)
    parser.add_argument("--min-confidence", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--background-per-length", type=int, default=10)
    parser.add_argument(
        "--cancer-only",
        action="store_true",
        help="Exclude rows whose cancers field is exactly 'non-cancer'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ds = _as_dataset(load_from_disk(args.dataset))
    references = load_reference_fasta(args.reference_fasta)
    explicit_breakpoints = {}
    if args.breakpoint_table:
        bp_path = Path(args.breakpoint_table)
        bp_frame = pd.read_csv(
            bp_path, sep="\t" if bp_path.suffix in {".tsv", ".txt"} else ","
        )
        explicit_breakpoints = dict(
            zip(bp_frame["seq_id"].astype(str), bp_frame["breakpoint"].astype(int))
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    sequence_rows: list[dict] = []
    peptide_rows: list[dict] = []
    failures: list[dict] = []
    rows = ds if args.limit is None else ds.select(range(min(args.limit, len(ds))))

    for row in tqdm(rows, desc="Preparing fusion sequences"):
        if args.cancer_only and str(row.get("cancers", "")).strip().lower() == "non-cancer":
            continue
        seq_id = str(row.get("seq_id", ""))
        fusion_genes = str(row.get("fusiongenes", ""))
        sequence = clean_sequence(row.get("aa_seq", ""))
        try:
            head_gene, tail_gene = split_fusion_genes(fusion_genes)
            head_ref = references[head_gene]
            tail_ref = references[tail_gene]
            if seq_id in explicit_breakpoints:
                breakpoint = explicit_breakpoints[seq_id]
                if not 0 < breakpoint < len(sequence):
                    raise ValueError(f"Explicit breakpoint outside sequence: {breakpoint}")
                call = BreakpointCall(
                    head_end=breakpoint,
                    tail_start=breakpoint,
                    breakpoint=breakpoint,
                    overlap=0,
                    gap=0,
                    head_identity=float("nan"),
                    tail_identity=float("nan"),
                    confidence="high",
                    method="explicit_table",
                )
            else:
                call = infer_breakpoint(sequence, head_ref, tail_ref)
        except (ValueError, KeyError, IndexError) as exc:
            failures.append({"seq_id": seq_id, "fusiongenes": fusion_genes, "reason": str(exc)})
            continue

        if CONFIDENCE[call.confidence] < CONFIDENCE[args.min_confidence]:
            failures.append(
                {
                    "seq_id": seq_id,
                    "fusiongenes": fusion_genes,
                    "reason": f"breakpoint confidence={call.confidence}",
                }
            )
            continue

        context, context_junction = junction_context(sequence, call.breakpoint, args.flank)
        split = stable_group_split(fusion_genes, args.seed)
        base = {
            "seq_id": seq_id,
            "fusiongenes": fusion_genes,
            "head_gene": head_gene,
            "tail_gene": tail_gene,
            "cancers": row.get("cancers"),
            "primary_sources": row.get("primary_sources"),
            "secondary_sources": row.get("secondary_sources"),
            "breakpoint": call.breakpoint,
            "breakpoint_confidence": call.confidence,
            "head_identity": call.head_identity,
            "tail_identity": call.tail_identity,
            "overlap": call.overlap,
            "gap": call.gap,
            "breakpoint_method": call.method,
            "context_seq": context,
            "context_junction": context_junction,
            "split": split,
        }
        sequence_rows.append(base)

        for peptide in enumerate_junction_peptides(
            sequence, call.breakpoint, args.lengths, args.min_side_residues
        ):
            context_start = call.breakpoint - context_junction
            peptide_rows.append(
                {
                    **base,
                    **peptide,
                    "embedding_seq": context,
                    "embedding_peptide_start": peptide["peptide_start"] - context_start,
                    "embedding_peptide_end": peptide["peptide_end"] - context_start,
                    "embedding_junction": context_junction,
                    "label": 1,
                    "class_name": "junction",
                }
            )
        for peptide in enumerate_nonjunction_background(
            sequence, call.breakpoint, args.lengths, args.background_per_length
        ):
            local_seq, local_start, local_end = peptide_context(
                sequence, peptide["peptide_start"], peptide["peptide_end"], args.flank
            )
            peptide_rows.append(
                {
                    **base,
                    **peptide,
                    "embedding_seq": local_seq,
                    "embedding_peptide_start": local_start,
                    "embedding_peptide_end": local_end,
                    "embedding_junction": -1,
                    "label": 0,
                    "class_name": "fusion_nonjunction",
                }
            )
        for source_name, reference in (("wildtype_head", head_ref), ("wildtype_tail", tail_ref)):
            for peptide in enumerate_protein_background(
                reference, args.lengths, args.background_per_length
            ):
                local_seq, local_start, local_end = peptide_context(
                    reference, peptide["peptide_start"], peptide["peptide_end"], args.flank
                )
                peptide_rows.append(
                    {
                        **base,
                        **peptide,
                        "embedding_seq": local_seq,
                        "embedding_peptide_start": local_start,
                        "embedding_peptide_end": local_end,
                        "embedding_junction": -1,
                        "label": 0,
                        "class_name": source_name,
                    }
                )

    seq_df = pd.DataFrame(sequence_rows)
    pep_df = pd.DataFrame(peptide_rows)
    fail_df = pd.DataFrame(failures)
    seq_df.to_parquet(output / "contexts.parquet", index=False)
    pep_df.to_parquet(output / "peptides.parquet", index=False)
    fail_df.to_csv(output / "failures.tsv", sep="\t", index=False)
    summary = {
        "input_sequences": len(rows),
        "accepted_sequences": len(seq_df),
        "peptides": len(pep_df),
        "failure_count": len(fail_df),
        "splits": Counter(seq_df["split"]) if len(seq_df) else {},
        "confidence": Counter(seq_df["breakpoint_confidence"]) if len(seq_df) else {},
    }
    write_json(output / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
