"""Build a small doctor-reviewed medical training dataset without auto-training on live patient data."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    approved_path = root / "approved_cases.jsonl"
    output_path = root / "dataset" / "doctor_approved_cases.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    if approved_path.exists():
        for line in approved_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                records.append(item)
            except json.JSONDecodeError:
                continue

    if not records:
        sample = {
            "patient_id": "demo-patient",
            "transcript": "Mere pet me dard hai. Aaj subah se shuru hua. Achi tarah se khana nahi kha pa raha hoon.",
            "summary": "Abdominal pain started in the morning and is associated with poor oral intake.",
            "diagnoses": ["Abdominal pain"],
            "treatment_plan": ["Further history and physical assessment required."],
            "approved_by": "doctor-demo",
            "tags": ["abdominal-pain", "pain", "follow-up"],
        }
        records = [sample]

    examples = []
    for record in records:
        examples.append({
            "prompt": (
                "You are a safe clinical intake interviewer. Ask one question only. "
                "Never diagnose or prescribe medicine. Reply in the patient's language.\n"
                f"Patient message: {record['transcript']}"
            ),
            "response": record["summary"],
            "labels": {"diagnoses": record.get("diagnoses", []), "tags": record.get("tags", [])},
        })

    output_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in examples) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(examples)} curated doctor-approved examples to {output_path}")


if __name__ == "__main__":
    main()
