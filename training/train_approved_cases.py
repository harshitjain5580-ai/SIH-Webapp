from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="training/dataset/approved_learning_dataset.jsonl")
    parser.add_argument("--output", default="training/outputs/doctor-approved-lora")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the retraining pipeline. Use a GPU-enabled environment.")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Approved training dataset not found: {dataset_path}")

    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = [item["text"] for item in rows]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized = Dataset.from_dict({"text": examples}).map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=512),
        batched=True,
        remove_columns=["text"],
    )

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map={"": 0}, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.add_adapter(LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

    output = Path(args.output)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output / "checkpoints"),
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-5,
            bf16=True,
            logging_steps=1,
            save_strategy="epoch",
            report_to="none",
            optim="adamw_torch",
            remove_unused_columns=False,
        ),
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    print(json.dumps({"status": "training_complete", "examples": len(examples), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
