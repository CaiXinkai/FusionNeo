from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from datasets import Dataset, DatasetDict, load_from_disk

from .common import split_fusion_genes

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"


def _as_dataset(obj: Dataset | DatasetDict) -> Dataset:
    if isinstance(obj, DatasetDict):
        return obj[next(iter(obj.keys()))]
    return obj


def fetch_gene(gene: str, timeout: int = 30) -> dict[str, str] | None:
    params = {
        "query": f"(gene_exact:{gene}) AND (organism_id:9606) AND (reviewed:true)",
        "format": "tsv",
        "fields": "accession,gene_primary,protein_name,sequence,length",
        "size": 10,
    }
    response = requests.get(UNIPROT_URL, params=params, timeout=timeout)
    response.raise_for_status()
    lines = response.text.strip().splitlines()
    if len(lines) < 2:
        return None
    rows = [line.split("\t") for line in lines[1:]]
    rows.sort(key=lambda row: int(row[-1]), reverse=True)
    accession, gene_name, protein_name, sequence, length = rows[0]
    return {
        "gene": gene,
        "uniprot_gene": gene_name,
        "accession": accession,
        "protein_name": protein_name,
        "sequence": sequence,
        "length": length,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch reviewed human canonical-like proteins for FusOn-DB genes."
    )
    parser.add_argument("--dataset", default="data/raw/fuson_db")
    parser.add_argument("--output-fasta", default="data/reference/uniprot_human_fusion_genes.fasta")
    parser.add_argument("--output-table", default="data/reference/uniprot_human_fusion_genes.tsv")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None, help="Useful for a smoke test.")
    args = parser.parse_args()

    ds = _as_dataset(load_from_disk(args.dataset))
    genes: set[str] = set()
    for pair in ds["fusiongenes"]:
        try:
            genes.update(split_fusion_genes(pair))
        except ValueError:
            continue
    genes = {gene for gene in genes if gene}
    ordered = sorted(genes)
    if args.limit:
        ordered = ordered[: args.limit]

    records: list[dict[str, str]] = []
    missing: list[str] = []
    for index, gene in enumerate(ordered, 1):
        try:
            result = fetch_gene(gene)
        except requests.RequestException as exc:
            print(f"[{index}/{len(ordered)}] {gene}: request failed: {exc}")
            missing.append(gene)
            continue
        if result is None:
            missing.append(gene)
        else:
            records.append(result)
        if index % 100 == 0:
            print(f"Fetched {index}/{len(ordered)} genes; found {len(records)}")
        time.sleep(args.sleep)

    fasta_path = Path(args.output_fasta)
    table_path = Path(args.output_table)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(
        [
            SeqRecord(
                Seq(row["sequence"]),
                id=row["gene"],
                description=f'{row["accession"]} {row["protein_name"]}',
            )
            for row in records
        ],
        fasta_path,
        "fasta",
    )
    pd.DataFrame(records).to_csv(table_path, sep="\t", index=False)
    Path(str(table_path) + ".missing.txt").write_text("\n".join(missing), encoding="utf-8")
    print({"found": len(records), "missing": len(missing), "fasta": str(fasta_path)})


if __name__ == "__main__":
    main()

