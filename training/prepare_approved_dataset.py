from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    source = Path("training/approved_cases.jsonl")
    target = Path("training/dataset/approved_learning_dataset.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError("No approved cases found at training/approved_cases.jsonl")

    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not payload.get("approved_by_doctor"):
            continue
        question = str(payload.get("user_question", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        language = str(payload.get("language", "English")).strip() or "English"
        if not question or not answer:
            continue
        prompt = (
            "You are a safe clinical intake interviewer. Ask one question only. "
            "Never diagnose or prescribe medicine. Reply in the patient's language.\n"
            f"User: {question}\nAssistant: {answer}\n"
        )
        records.append({"language": language, "text": prompt})

    if not records:
        raise ValueError("Approved training data is empty; add doctor-approved cases first.")

    target.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
    summary = {
        "approved_cases": len(records),
        "language_distribution": {k: sum(1 for item in records if item["language"].lower() == k.lower()) for k in sorted({item["language"] for item in records})},
        "source": str(source),
        "output": str(target),
    }
    Path("training/dataset/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
