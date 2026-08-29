"""Use the base Qwen model with ChatML formatting for the bilingual intake endpoint."""
from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_model = None
_tokenizer = None
_lock = Lock()


def _load():
    global _model, _tokenizer
    if _model is None:
        with _lock:
            if _model is None:
                model_name = "Qwen/Qwen2.5-1.5B-Instruct"
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

                _tokenizer = AutoTokenizer.from_pretrained(model_name)
                _tokenizer.pad_token = _tokenizer.eos_token

                base = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map={"": 0} if device == "cuda" else None,
                )

                adapter = Path(os.environ.get("LOCAL_MODEL_ADAPTER", "training/outputs/qwen2.5-1.5b-bilingual-lora"))
                use_adapter = os.environ.get("USE_LOCAL_ADAPTER", "false").lower() == "true"
                if use_adapter and not adapter.is_absolute():
                    adapter = Path.cwd() / adapter

                if use_adapter and adapter.exists() and (adapter / "adapter_model.safetensors").exists():
                    _model = PeftModel.from_pretrained(base, str(adapter)).eval()
                else:
                    _model = base.eval()

                if device == "cpu":
                    _model.to(device)
    return _model, _tokenizer


def ask(transcript: str) -> str:
    model, tokenizer = _load()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a safe clinical intake interviewer. Ask one question only. "
                "Never diagnose or prescribe medicine. Reply in the patient's language."
            ),
        },
        {
            "role": "user",
            "content": f"Continue the interview based on this patient message: {transcript}",
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
