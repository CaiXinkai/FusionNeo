from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a normalized external peptide benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if frame.empty:
        raise ValueError("The normalized benchmark table is empty")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    vectors = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(frame), args.batch_size), desc="Embedding benchmark"):
            peptides = frame["peptide"].iloc[start : start + args.batch_size].tolist()
            encoded = tokenizer(peptides, padding=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].bool()
            mask[:, 0] = False
            lengths = mask.sum(dim=1)
            for row_index, length in enumerate(lengths.tolist()):
                # Remove the terminal EOS token from the mean.
                valid = torch.where(mask[row_index])[0][:-1]
                vectors.append(hidden[row_index, valid].mean(dim=0).float().cpu().numpy())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "metadata.csv", index=False)
    np.save(output / "peptide_embeddings.npy", np.stack(vectors))
    print({"rows": len(frame), "dimension": len(vectors[0])})


if __name__ == "__main__":
    main()
