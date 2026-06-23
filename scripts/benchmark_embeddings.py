from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-safe linear probe for prepared FusionNeo embeddings."
    )
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--label", default="label")
    parser.add_argument("--output", default="outputs/benchmark.json")
    args = parser.parse_args()

    x = np.load(args.embeddings)
    metadata_path = Path(args.metadata)
    metadata = (
        pd.read_parquet(metadata_path)
        if metadata_path.suffix == ".parquet"
        else pd.read_csv(metadata_path)
    )
    train_mask = metadata["split"].eq("train").to_numpy()
    test_mask = metadata["split"].eq("test").to_numpy()
    y = metadata[args.label].to_numpy()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_mask])
    x_test = scaler.transform(x[test_mask])
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(x_train, y[train_mask])
    score = model.predict_proba(x_test)[:, 1]
    metrics = {
        "auroc": roc_auc_score(y[test_mask], score),
        "average_precision": average_precision_score(y[test_mask], score),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
