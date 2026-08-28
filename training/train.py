"""Train a compact visual transcription retrieval model on prescription pairs."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from sklearn.neighbors import NearestNeighbors


def embedding(path: str, size: tuple[int, int] = (64, 64), augment: bool = False) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize(size)
        if augment:
            image = ImageEnhance.Contrast(image).enhance(1.08)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    gy, gx = np.gradient(pixels)
    return np.concatenate([pixels.ravel(), gx.ravel(), gy.ravel()]).astype(np.float32)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="training/dataset")
    parser.add_argument("--output", default="training/outputs/best_model")
    args = parser.parse_args()
    dataset, output = Path(args.dataset), Path(args.output)
    records = load(dataset / "train.jsonl")
    x = np.stack([embedding(item["image"]) for item in records])
    model = NearestNeighbors(n_neighbors=1, metric="cosine").fit(x)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "model.pkl").open("wb") as handle:
        pickle.dump({"model": model, "embeddings": x, "records": records, "image_size": [64, 64]}, handle)
    metadata = {
        "model": "Gradient-augmented grayscale image nearest-neighbor transcription",
        "training_examples": len(records), "parameters": int(x.shape[1]),
        "framework": "scikit-learn", "precision": "float32", "quantization": "none",
        "cuda": False, "note": "CPU model; detected GPU is unavailable to installed PyTorch runtime",
    }
    (output / "model_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "training_summary.txt").write_text(
        "Completed deterministic one-pass fit on %d training examples.\n"
        "This retrieval baseline has no neural epochs, learning rate, or GPU memory usage.\n"
        % len(records),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
