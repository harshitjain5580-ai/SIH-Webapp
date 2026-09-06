"""Train a Qwen intake model with consistent ChatML formatting for English/Hindi/Hinglish follow-up questions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

DEVANAGARI_MAP = {
    "अ":"a","आ":"aa","इ":"i","ई":"ee","उ":"u","ऊ":"oo","ए":"e","ऐ":"ai","ओ":"o","औ":"au",
    "क":"k","ख":"kh","ग":"g","घ":"gh","च":"ch","छ":"chh","ज":"j","झ":"jh","ट":"t","ठ":"th",
    "ड":"d","ढ":"dh","ण":"n","त":"t","थ":"th","द":"d","ध":"dh","न":"n","प":"p","फ":"ph",
    "ब":"b","भ":"bh","म":"m","य":"y","र":"r","ल":"l","व":"v","श":"sh","ष":"sh","स":"s","ह":"h",
    "ं":"n","ः":"h","।":".","़":"",
}


def romanize_hindi(text: str) -> str:
    """Create a readable Romanized-Hindi/Hinglish training variant."""
    result = []
    for char in text:
        result.append(DEVANAGARI_MAP.get(char, char))
    return "".join(result).replace("aa p", "aap").replace(" hai", " hai")


def _build_chatml_examples(frame: pd.DataFrame) -> list[dict]:
    """Create consistent ChatML examples with transcript + question pairs instead of category-only prompts."""
    records = []
    for _, row in frame.iterrows():
        category = str(row.get("category") or "").strip()
        transcript = str(row.get("transcript") or row.get("patient transcript") or row.get("input") or "").strip()
        if not transcript:
            transcript = f"Patient reported a symptom in the {category} category." if category else "Patient reported a symptom."
        english_question = str(row.get("English question") or row.get("question_en") or "").strip()
        hindi_question = str(row.get("Hindi question (Devanagari)") or row.get("question_hi") or "").strip()
        hinglish_question = str(row.get("Hinglish question (Roman)") or row.get("question_hinglish") or "").strip()
        if not english_question and not hindi_question and not hinglish_question:
            continue
        if not hindi_question and english_question:
            hindi_question = romanize_hindi(english_question)
        if not hinglish_question and english_question:
            hinglish_question = english_question

        variants = []
        if english_question:
            variants.append(("English", english_question))
        if hindi_question:
            variants.append(("Hindi", hindi_question))
        if hinglish_question:
            variants.append(("Hinglish", hinglish_question))
        for language, question in variants:
            records.append({
                "messages": [
                    {"role": "system", "content": "You are a safe clinical intake interviewer. Ask exactly one short follow-up question. Never diagnose or prescribe medicine."},
                    {"role": "user", "content": f"Patient transcript: {transcript}. Continue the interview in {language}."},
                    {"role": "assistant", "content": question},
                ],
                "language": language,
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bilingual_clinical_conversation_questions.xlsx")
    parser.add_argument("--output", default="training/outputs/qwen2.5-1.5b-bilingual-lora")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for a quick dry-run or smoke training loop.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the dataset and print a sample without running full training.")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    frame_path = Path(args.dataset)
    if not frame_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {frame_path}")

    with pd.ExcelFile(frame_path) as workbook:
        sheet_names = workbook.sheet_names
        sheet_name = "Training_Data" if "Training_Data" in sheet_names else sheet_names[0]
        frame = pd.read_excel(frame_path, sheet_name=sheet_name)
    required_columns = ["English question", "Hindi question (Devanagari)"]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    frame = frame.dropna(subset=["English question", "Hindi question (Devanagari)"])
    examples = _build_chatml_examples(frame)
    if args.max_samples is not None:
        examples = examples[: args.max_samples]
    if not examples:
        raise ValueError("No usable training examples were generated from the dataset.")

    if args.dry_run:
        print(json.dumps({
            "sample_count": len(examples),
            "first_example": examples[0],
            "device": device,
        }, ensure_ascii=False, indent=2))
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_list([
        {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)}
        for example in examples
    ])
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=512),
        batched=True,
        remove_columns=["text"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": 0} if device == "cuda" else None,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model.add_adapter(lora_cfg)
    output = Path(args.output)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output / "checkpoints"),
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-5,
            use_cpu=device == "cpu",
            bf16=device == "cuda",
            fp16=False,
            logging_steps=1,
            save_strategy="epoch",
            report_to="none",
            optim="adamw_torch",
            remove_unused_columns=False,
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "training_summary.json").write_text(
        json.dumps({
            "base_model": args.model,
            "method": "LoRA",
            "format": "ChatML",
            "examples": len(examples),
            "languages": ["English", "Hindi", "Hinglish"],
            "epochs": 3,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 5e-5,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "vram_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3 if device == "cuda" else None,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
