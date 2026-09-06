"""Run independent transcription inference: python inference.py --image path.jpg."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from train import embedding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="training/outputs/best_model/model.pkl")
    args = parser.parse_args()
    with Path(args.model).open("rb") as handle:
        artifact = pickle.load(handle)
    _, indexes = artifact["model"].kneighbors([embedding(args.image)])
    source = artifact["records"][int(indexes[0][0])]
    print("Loading model...\nLoading image...\nRunning inference...\n")
    print(json.dumps({"extracted_text": source["text"], "nearest_training_image": source["image"]}, indent=2))


if __name__ == "__main__":
    main()
