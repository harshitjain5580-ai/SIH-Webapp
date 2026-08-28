"""Validate the Kaggle image/JSON pairs and create leakage-safe manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image


TEXT_KEYS = {
    "complete_extracted_text", "full_extracted_text", "complete_raw_text",
    "extracted_text", "extracted_text_in_order", "full_text_content",
}


def collect_text(value: Any) -> list[str]:
    if isinstance(value, dict):
        preferred = [value[k] for k in TEXT_KEYS if k in value]
        if preferred:
            return collect_text(preferred[0])
        result: list[str] = []
        for item in value.values():
            result.extend(collect_text(item))
        return result
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(collect_text(item))
        return result
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="training/dataset")
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    records, errors, hashes = [], [], {}
    for annotation in sorted(root.rglob("*.json")):
        image_candidates = list(annotation.parent.glob("*.jpg"))
        if not image_candidates:
            errors.append({"file": str(annotation), "reason": "missing image"})
            continue
        image = image_candidates[0]
        try:
            with Image.open(image) as handle:
                handle.verify()
            data = json.loads(annotation.read_text(encoding="utf-8"))
            texts = collect_text(data)
            text = texts[0] if texts else ""
            if not text:
                errors.append({"file": str(annotation), "reason": "missing supported text annotation"})
                continue
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes:
                errors.append({"file": str(annotation), "reason": "duplicate image", "duplicate_of": hashes[digest]})
                continue
            hashes[digest] = str(image)
            records.append({"image": str(image), "annotation": str(annotation), "text": text})
        except Exception as exc:
            errors.append({"file": str(annotation), "reason": f"{type(exc).__name__}: {exc}"})

    random.Random(42).shuffle(records)
    n = len(records)
    train_end, val_end = int(n * 0.8), int(n * 0.9)
    splits = {"train": records[:train_end], "validation": records[train_end:val_end], "test": records[val_end:]}
    output.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        (output / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=True) for item in items) + "\n", encoding="utf-8"
        )
    summary = {
        "total_json": len(list(root.rglob("*.json"))), "valid_pairs": len(records),
        "errors": errors, "splits": {k: len(v) for k, v in splits.items()},
        "annotation_format": "JSON sidecars; transcription from complete_extracted_text or equivalent full-text key",
        "classes": [], "bounding_boxes": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
