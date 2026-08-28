"""Lazy local Qwen adapter inference for the bilingual intake endpoint."""
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
                adapter = Path(os.environ.get(
                    "LOCAL_MODEL_ADAPTER",
                    "training/outputs/qwen2.5-1.5b-bilingual-lora",
                ))
                if not adapter.is_absolute():
                    adapter = Path.cwd() / adapter
                if not (adapter / "adapter_model.safetensors").exists():
                    raise FileNotFoundError(f"Local bilingual adapter not found: {adapter}")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
                _tokenizer = AutoTokenizer.from_pretrained(str(adapter))
                base = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    dtype=dtype,
                    device_map={"": 0} if device == "cuda" else None,
                )
                _model = PeftModel.from_pretrained(base, str(adapter)).eval()
                if device == "cpu":
                    _model.to(device)
    return _model, _tokenizer


def ask(transcript: str) -> str:
    model, tokenizer = _load()
    prompt = (
        "System: You are a safe clinical intake interviewer. Ask one question only. "
        "Never diagnose or prescribe medicine. Reply in the patient's language.\n"
        f"User: Continue the interview based on this patient message: {transcript}\nAssistant:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
