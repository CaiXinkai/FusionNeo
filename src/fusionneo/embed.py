from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def mean_region(hidden: torch.Tensor, start: int, end: int) -> np.ndarray:
    start = max(1, start)
    end = min(hidden.shape[0] - 1, end)
    if end <= start:
        return hidden[1:-1].mean(dim=0).float().cpu().numpy()
    return hidden[start:end].mean(dim=0).float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export context, junction, and peptide embeddings.")
    parser.add_argument("--model", required=True, help="Fine-tuned checkpoint or Hugging Face model.")
    parser.add_argument("--peptides", default="data/processed/peptides.parquet")
    parser.add_argument("--output", default="outputs/embeddings")
    parser.add_argument("--junction-radius", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    frame = pd.read_parquet(args.peptides)
    if args.limit:
        frame = frame.head(args.limit)
    if frame.empty:
        raise ValueError("No peptide rows were found in the input table")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    context_vectors = []
    junction_vectors = []
    peptide_vectors = []
    with torch.inference_mode():
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="Embedding"):
            encoded = tokenizer(row.embedding_seq, return_tensors="pt", truncation=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state[0]
            peptide_start_token = row.embedding_peptide_start + 1
            peptide_end_token = row.embedding_peptide_end + 1

            context_vectors.append(hidden[1:-1].mean(dim=0).float().cpu().numpy())
            if row.embedding_junction >= 0:
                junction_token = row.embedding_junction + 1
                junction_vectors.append(
                    mean_region(
                        hidden,
                        junction_token - args.junction_radius,
                        junction_token + args.junction_radius,
                    )
                )
            else:
                junction_vectors.append(
                    np.full(hidden.shape[-1], np.nan, dtype=np.float32)
                )
            peptide_vectors.append(
                mean_region(hidden, peptide_start_token, peptide_end_token)
            )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = frame.copy()
    metadata.insert(0, "embedding_id", np.arange(len(metadata)))
    metadata.to_parquet(output / "metadata.parquet", index=False)
    np.save(output / "context_embeddings.npy", np.stack(context_vectors))
    np.save(output / "junction_embeddings.npy", np.stack(junction_vectors))
    np.save(output / "peptide_embeddings.npy", np.stack(peptide_vectors))
    print({"rows": len(metadata), "dimension": len(peptide_vectors[0]), "output": str(output)})


if __name__ == "__main__":
    main()
