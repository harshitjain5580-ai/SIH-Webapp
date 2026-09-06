"""Evaluate transcription retrieval with task-appropriate CER, WER and exact match."""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
from PIL import Image

from train import embedding


def distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = current
    return row[-1]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def score(records: list[dict], artifact: dict) -> tuple[dict, list[dict]]:
    predictions, refs = [], []
    for item in records:
        _, indexes = artifact["model"].kneighbors([embedding(item["image"])])
        predictions.append(artifact["records"][int(indexes[0][0])]["text"])
        refs.append(item["text"])
    cer = sum(distance(norm(p), norm(r)) / max(1, len(norm(r))) for p, r in zip(predictions, refs)) / max(1, len(refs))
    wers = []
    for p, r in zip(predictions, refs):
        wers.append(distance(norm(p), norm(r)) / max(1, len(norm(r).split())))
    return {
        "examples": len(records),
        "character_error_rate": cer,
        "word_error_rate": sum(wers) / max(1, len(wers)),
        "exact_match": sum(norm(p) == norm(r) for p, r in zip(predictions, refs)) / max(1, len(refs)),
    }, [{"image": x["image"], "prediction": p, "ground_truth": r} for x, p, r in zip(records, predictions, refs)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="training/dataset")
    parser.add_argument("--model", default="training/outputs/best_model/model.pkl")
    parser.add_argument("--output", default="training/outputs/metrics.json")
    args = parser.parse_args()
    with Path(args.model).open("rb") as handle:
        artifact = pickle.load(handle)
    splits = {}
    all_predictions = {}
    for split in ("validation", "test"):
        records = [json.loads(line) for line in (Path(args.dataset) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
        splits[split], all_predictions[split] = score(records, artifact)
    metrics = {"validation": splits["validation"], "test": splits["test"], "predictions": all_predictions}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
