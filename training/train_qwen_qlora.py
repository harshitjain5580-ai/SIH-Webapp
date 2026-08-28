"""Fine-tune Qwen2.5-1.5B-Instruct with 4-bit QLoRA on prescription transcriptions."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bilingual_clinical_conversation_questions.xlsx")
    parser.add_argument("--output", default="training/outputs/qwen2.5-1.5b-bilingual-lora")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit QLoRA; install a CUDA-enabled PyTorch build.")

    frame = pd.read_excel(args.dataset, sheet_name="Training_Data").dropna(
        subset=["English question", "Hindi question (Devanagari)"]
    )
    records = frame.to_dict("records")
    examples = []
    for record in records:
        safety = str(record.get("safety note", ""))
        examples.extend([
            f"System: You are a safe clinical intake interviewer. Ask one question only. Never diagnose or prescribe medicine.\n"
            f"User: Continue the interview in English. Category: {record['category']}. Ask the next question.\n"
            f"Assistant: {record['English question']}\nSafety: {safety}",
            f"System: आप सुरक्षित स्वास्थ्य-साक्षात्कार सहायक हैं। एक बार में केवल एक प्रश्न पूछें। निदान या दवा न लिखें।\n"
            f"User: हिंदी में बातचीत जारी रखें। श्रेणी: {record['category']}. अगला प्रश्न पूछें।\n"
            f"Assistant: {record['Hindi question (Devanagari)']}\nSafety: {safety}",
            f"System: You are a safe clinical intake interviewer. Ask one question only. Never diagnose or prescribe medicine.\n"
            f"User: Continue the interview in Hinglish (Roman Hindi). Category: {record['category']}. Ask the next question.\n"
            f"Assistant: {romanize_hindi(record['Hindi question (Devanagari)'])}\nSafety: {safety}",
        ])
    dataset = Dataset.from_dict({"text": examples})
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=512),
        batched=True,
        remove_columns=["text"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": 0}, dtype=torch.bfloat16
    )
    model.gradient_checkpointing_enable()
    model.add_adapter(
        LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
    )
    output = Path(args.output)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output / "checkpoints"),
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-5,
            bf16=True,
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
            "base_model": args.model, "method": "LoRA",
            "quantization": "none (bitsandbytes 4-bit loader crashes on this Windows runtime)",
            "examples": len(examples), "source_question_pairs": len(records), "languages": ["English", "Hindi", "Hinglish"],
            "epochs": 3, "batch_size": 1,
            "gradient_accumulation_steps": 8, "learning_rate": 5e-5,
            "gpu": torch.cuda.get_device_name(0),
            "vram_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
