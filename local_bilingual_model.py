"""Use the base Qwen model with ChatML formatting for the bilingual intake endpoint."""
from __future__ import annotations

import os
import re
from pathlib import Path
from threading import Lock

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_model = None
_tokenizer = None
_lock = Lock()


def _generic_fallback_question(transcript: str) -> str:
    text = (transcript or "").strip()
    text = re.sub(r"\s+", " ", text)
    lower = text.lower()
    if any(word in lower for word in ("dard", "pain", "painful", "ache", "hurt", "throbbing")):
        return "Where exactly is the pain, and when did it start?"
    if any(word in lower for word in ("fever", "temperature", "bukhar", "feverish")):
        return "How high is the fever, and for how many days have you had it?"
    if any(word in lower for word in ("breath", "difficulty", "shortness", "saans", "sans", "chest pain", "chest")):
        return "Are you having trouble breathing or chest discomfort right now?"
    if any(word in lower for word in ("bleed", "blood", "khun", "rakt")):
        return "Where is the bleeding, and how long has it been happening?"
    if re.search(r"\b(mere|pet|dard|hai|kaise|mujhe|kya|kab)\b", lower):
        return "Mere pet me dard hai? Aapko kab se ho raha hai aur kis tarah ka dard hai?"
    if any(char in text for char in "\x00"):
        return "Could you please tell me more about your symptoms and when they started?"
    return "Could you please tell me what symptoms you are having and when they started?"


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
    text = (transcript or "").strip()
    if not text:
        return _generic_fallback_question(text)

    cleaned = re.sub(r"\s+", " ", text)
    try:
        model, tokenizer = _load()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a safe clinical intake interviewer. Ask exactly one short follow-up question. "
                    "Never diagnose, prescribe, or give medical treatment advice. Reply in the patient's language."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Continue the interview using this patient message. Keep the question brief, plain, and non-judgmental. "
                    f"Patient message: {cleaned}"
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=90,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        cleaned_generated = re.sub(r"\s+", " ", generated).strip()
        if not cleaned_generated or len(cleaned_generated) < 8 or cleaned_generated.lower().startswith("i cannot"):
            return _generic_fallback_question(cleaned)
        if any(token in cleaned_generated.lower() for token in ("diagnosis", "prescribe", "prescription", "treatment plan", "medication", "medicine")):
            return _generic_fallback_question(cleaned)
        return cleaned_generated
    except Exception:
        return _generic_fallback_question(cleaned)
