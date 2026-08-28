"""Fine-tune Qwen2.5-1.5B-Instruct with 4-bit QLoRA on prescription transcriptions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="training/dataset/train.jsonl")
    parser.add_argument("--output", default="training/outputs/qwen2.5-1.5b-qlora")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit QLoRA; install a CUDA-enabled PyTorch build.")

    records = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line]
    dataset = Dataset.from_dict({"text": [f"Prescription transcription:\n{r['text']}" for r in records]})
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
            "base_model": args.model, "method": "LoRA fallback",
            "quantization": "none (bitsandbytes 4-bit loader crashes on this Windows runtime)",
            "examples": len(records), "epochs": 3, "batch_size": 1,
            "gradient_accumulation_steps": 8, "learning_rate": 5e-5,
            "gpu": torch.cuda.get_device_name(0),
            "vram_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
