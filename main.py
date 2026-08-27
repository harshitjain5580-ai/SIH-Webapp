"""
MediKiosk: AI Clinical History Software Platform
Backend entrypoint (FastAPI).

Implements the /extract-history and /extract-from-image endpoints per
claude.md.md:
- Pydantic v2 BaseModel data contracts with explicit Field(description=...)
- OpenAI native Structured Outputs (client.beta.chat.completions.parse) —
  never hand-parse LLM string output.
- async def routing.
- SOCRATES pain framework + AYUSH Dashavidha Pariksha parameters captured
  in the schema below.
- Emergency red-flag detection (cardiac / neurological) forces
  red_flags_detected = True.
- Database Rules: supabase-py for all DB operations, extracted histories are
  stored in patient_histories, and prescription images are uploaded to the
  medical_documents Storage bucket before their public URL is sent to the
  OpenAI Vision model.
"""

import logging
import os
import uuid
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from openai import OpenAI, OpenAIError
from supabase import Client, create_client

logger = logging.getLogger("medikiosk")

# ---------------------------------------------------------------------------
# Pydantic Data Contracts (Section 4 of claude.md.md)
# ---------------------------------------------------------------------------


class Medication(BaseModel):
    """A single current medication entry."""

    name: str = Field(description="Name of the medication as reported by the patient.")
    dosage: str = Field(description="Dosage of the medication, e.g. '500mg'.")
    frequency: str = Field(description="How often the medication is taken, e.g. 'twice daily'.")


class AyushParameters(BaseModel):
    """AYUSH Dashavidha Pariksha parameters captured from the conversation."""

    prakriti: str = Field(
        description="Patient's baseline constitutional type (Vata/Pitta/Kapha balance) as assessed from the conversation."
    )
    vikriti: str = Field(
        description="Patient's current state of doshic imbalance as assessed from the conversation."
    )


class ClinicalHistorySummary(BaseModel):
    """
    Primary JSON data contract for backend-to-frontend clinical history
    communication (Section 4 of claude.md.md).
    """

    chief_complaint: str = Field(
        description="The patient's primary reason for the visit, in 1-2 sentences."
    )
    hpi_socrates: str = Field(
        description=(
            "Detailed narrative of the History of Present Illness, structured using the "
            "SOCRATES framework: Site, Onset, Character, Radiation, Associated symptoms, "
            "Time course, Exacerbating/relieving factors, and Severity."
        )
    )
    past_medical_history: List[str] = Field(
        description="List of past medical history items reported by the patient."
    )
    current_medications: List[Medication] = Field(
        description="List of medications the patient is currently taking."
    )
    ayush_parameters: AyushParameters = Field(
        description="AYUSH Dashavidha Pariksha parameters (Prakriti, Vikriti) captured from the conversation."
    )
    red_flags_detected: bool = Field(
        description=(
            "True if the transcript contains markers for acute cardiac events "
            "(e.g. chest pain, dyspnoea) or neurological deficits (e.g. stroke symptoms)."
        )
    )


class TranscriptRequest(BaseModel):
    """Request body for the /extract-history endpoint."""

    transcript: str = Field(description="Raw patient conversation transcript to extract structured history from.")


# ---------------------------------------------------------------------------
# App & OpenAI client setup
# ---------------------------------------------------------------------------

app = FastAPI(title="MediKiosk API", version="0.1.0")

# The OpenAI SDK requires a non-empty api_key at construction time. Fall back to
# a placeholder so the app can still start (and /health respond) even before
# OPENAI_API_KEY is configured; real calls will fail with a clear 502 until a
# valid key is set in the environment.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "not-set")

# Same fallback pattern for Supabase: allow the app to start before
# SUPABASE_URL / SUPABASE_KEY are configured; real DB/Storage calls will fail
# with a clear 502 until valid credentials are set in the environment.
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL") or "https://not-set.supabase.co",
    os.environ.get("SUPABASE_KEY") or "not-set",
)

PATIENT_HISTORIES_TABLE = "patient_histories"
MEDICAL_DOCUMENTS_BUCKET = "medical_documents"

SYSTEM_PROMPT = (
    "You are a clinical history extraction engine for MediKiosk, an AI clinical "
    "history intake kiosk. Given a patient conversation transcript, extract a "
    "structured clinical history summary.\n\n"
    "When the patient reports pain, apply the SOCRATES framework (Site, Onset, "
    "Character, Radiation, Associated symptoms, Time course, Exacerbating/relieving "
    "factors, Severity) in the hpi_socrates narrative.\n\n"
    "Capture AYUSH Dashavidha Pariksha parameters (Prakriti, Vikriti) whenever the "
    "transcript provides information relevant to Ayurvedic constitutional assessment, "
    "including Agni and Ahara-Vihara cues where mentioned.\n\n"
    "Set red_flags_detected to true if the transcript contains any markers of acute "
    "cardiac events (e.g. chest pain, dyspnoea) or neurological deficits (e.g. stroke "
    "symptoms such as facial droop, slurred speech, sudden weakness). Otherwise set it "
    "to false."
)


# ---------------------------------------------------------------------------
# Persistence (Database Rules, Section 5 of claude.md.md)
# ---------------------------------------------------------------------------


def _persist_history(summary: ClinicalHistorySummary) -> None:
    """
    Store an extracted clinical history in the patient_histories table via
    the official supabase-py client. Best-effort: a persistence failure is
    logged but does not prevent the extracted summary from reaching the
    caller, since the kiosk's primary function is the extraction itself.
    """
    try:
        supabase.table(PATIENT_HISTORIES_TABLE).insert(
            {
                "chief_complaint": summary.chief_complaint,
                "hpi_socrates": summary.hpi_socrates,
                "current_medications": [m.model_dump() for m in summary.current_medications],
                "ayush_parameters": summary.ayush_parameters.model_dump(),
                "red_flags_detected": summary.red_flags_detected,
            }
        ).execute()
    except Exception:
        logger.exception("Failed to persist clinical history to Supabase.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/extract-history", response_model=ClinicalHistorySummary)
async def extract_history(request: TranscriptRequest) -> ClinicalHistorySummary:
    """
    Extract a structured ClinicalHistorySummary from a raw patient transcript
    using OpenAI's native Structured Outputs. The LLM's JSON output is never
    hand-parsed: the SDK guarantees the response conforms to the Pydantic
    schema via client.beta.chat.completions.parse.
    """
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.transcript},
            ],
            response_format=ClinicalHistorySummary,
        )
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc

    message = completion.choices[0].message

    if message.refusal:
        raise HTTPException(status_code=422, detail=f"Model refused to process transcript: {message.refusal}")

    parsed = message.parsed
    if parsed is None:
        raise HTTPException(status_code=502, detail="Model did not return a parsed structured output.")

    _persist_history(parsed)

    return parsed


@app.post("/extract-from-image", response_model=ClinicalHistorySummary)
async def extract_from_image(file: UploadFile = File(...)) -> ClinicalHistorySummary:
    """
    Extract a structured ClinicalHistorySummary from a prescription/document
    image. Per the Database Rules, the image is uploaded to the
    medical_documents Storage bucket first, and only its public URL is sent
    to the OpenAI Vision model — the raw image bytes are never sent to OpenAI.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    extension = os.path.splitext(file.filename or "")[1] or ".jpg"
    object_path = f"{uuid.uuid4()}{extension}"

    try:
        supabase.storage.from_(MEDICAL_DOCUMENTS_BUCKET).upload(
            object_path,
            contents,
            {"content-type": file.content_type or "application/octet-stream"},
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Supabase Storage upload failed: {exc}") from exc

    public_url = supabase.storage.from_(MEDICAL_DOCUMENTS_BUCKET).get_public_url(object_path)

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract the structured clinical history from this prescription/document image.",
                        },
                        {"type": "image_url", "image_url": {"url": public_url}},
                    ],
                },
            ],
            response_format=ClinicalHistorySummary,
        )
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc

    message = completion.choices[0].message

    if message.refusal:
        raise HTTPException(status_code=422, detail=f"Model refused to process image: {message.refusal}")

    parsed = message.parsed
    if parsed is None:
        raise HTTPException(status_code=502, detail="Model did not return a parsed structured output.")

    _persist_history(parsed)

    return parsed
