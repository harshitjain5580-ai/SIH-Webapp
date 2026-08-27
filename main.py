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

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from openai import OpenAI, OpenAIError
from supabase import Client, create_client

load_dotenv()

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


class PatientHistoryRecord(BaseModel):
    """A row from the patient_histories Supabase table, as returned after insert."""

    id: str = Field(description="UUID primary key of the patient_histories row.")
    created_at: str = Field(description="Timestamp the row was created, as returned by Postgres.")
    chief_complaint: str = Field(description="The patient's primary reason for the visit.")
    hpi_socrates: str = Field(description="Detailed narrative of the History of Present Illness (SOCRATES).")
    current_medications: List[Medication] = Field(description="Medications the patient is currently taking.")
    ayush_parameters: AyushParameters = Field(description="AYUSH Dashavidha Pariksha parameters (Prakriti, Vikriti).")
    red_flags_detected: bool = Field(description="True if emergency red flags were detected.")


class ExtractedPrescription(BaseModel):
    """Structured prescription data extracted from a document image via the OpenAI Vision model."""

    medications: List[Medication] = Field(description="List of medications identified in the prescription image.")


class DocumentUploadResult(BaseModel):
    """Response for /upload-document: the Supabase Storage path plus the extracted prescription data."""

    storage_path: str = Field(
        description="Path of the uploaded file within the medical_documents Supabase Storage bucket."
    )
    extracted_prescription: ExtractedPrescription = Field(
        description="Structured prescription data extracted by the OpenAI Vision model."
    )


class WaitingRoomResponse(BaseModel):
    """Response for GET /api/v1/waiting-room: the most recently created patient histories."""

    histories: List[PatientHistoryRecord] = Field(
        description="The 10 most recently created patient histories, ordered newest first."
    )


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


def _persist_history(summary: ClinicalHistorySummary) -> dict:
    """
    Insert an extracted clinical history into the patient_histories table via
    the official supabase-py client, using the parsed ClinicalHistorySummary's
    exact JSON shape for the JSONB columns, and return the inserted row as
    Supabase returns it (including its generated id and created_at).
    """
    try:
        response = (
            supabase.table(PATIENT_HISTORIES_TABLE)
            .insert(
                {
                    "chief_complaint": summary.chief_complaint,
                    "hpi_socrates": summary.hpi_socrates,
                    "current_medications": [m.model_dump() for m in summary.current_medications],
                    "ayush_parameters": summary.ayush_parameters.model_dump(),
                    "red_flags_detected": summary.red_flags_detected,
                }
            )
            .execute()
        )
    except Exception as exc:
        logger.exception("Failed to persist clinical history to Supabase.")
        raise HTTPException(status_code=502, detail=f"Failed to persist clinical history to Supabase: {exc}") from exc

    if not response.data:
        raise HTTPException(status_code=502, detail="Supabase insert returned no data.")

    return response.data[0]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/extract-history", response_model=PatientHistoryRecord)
async def extract_history(request: TranscriptRequest) -> PatientHistoryRecord:
    """
    Extract a structured ClinicalHistorySummary from a raw patient transcript
    using OpenAI's native Structured Outputs, insert it as a new row in the
    patient_histories Supabase table, and return the inserted database
    record. The LLM's JSON output is never hand-parsed: the SDK guarantees
    the response conforms to the Pydantic schema via
    client.beta.chat.completions.parse.
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

    record = _persist_history(parsed)

    return record


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


@app.post("/upload-document", response_model=DocumentUploadResult)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResult:
    """
    Upload a prescription/document image to the medical_documents Supabase
    Storage bucket, then extract its medications via the OpenAI gpt-4o
    Vision model using Structured Outputs. Only the image's public Storage
    URL is sent to OpenAI — never the raw image bytes.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    extension = os.path.splitext(file.filename or "")[1] or ".jpg"
    storage_path = f"{uuid.uuid4()}{extension}"

    # Step 1: upload the raw file bytes to Supabase Storage.
    try:
        supabase.storage.from_(MEDICAL_DOCUMENTS_BUCKET).upload(
            storage_path,
            contents,
            {"content-type": file.content_type or "application/octet-stream"},
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Supabase Storage upload failed: {exc}") from exc

    # Step 2: retrieve the public URL for the uploaded image.
    public_url = supabase.storage.from_(MEDICAL_DOCUMENTS_BUCKET).get_public_url(storage_path)

    # Step 3: pass the public URL to the OpenAI gpt-4o Vision model, forcing
    # Structured Outputs to conform to ExtractedPrescription.
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a prescription extraction engine. Extract every medication listed "
                        "in the provided prescription image, including its name, dosage, and frequency."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all medications from this prescription image."},
                        {"type": "image_url", "image_url": {"url": public_url}},
                    ],
                },
            ],
            response_format=ExtractedPrescription,
        )
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc

    message = completion.choices[0].message

    if message.refusal:
        raise HTTPException(status_code=422, detail=f"Model refused to process image: {message.refusal}")

    parsed = message.parsed
    if parsed is None:
        raise HTTPException(status_code=502, detail="Model did not return a parsed structured output.")

    return DocumentUploadResult(storage_path=storage_path, extracted_prescription=parsed)


@app.get("/api/v1/waiting-room", response_model=WaitingRoomResponse)
async def get_waiting_room() -> WaitingRoomResponse:
    """
    Return the 10 most recently created clinical histories from the
    patient_histories table, ordered by created_at descending.
    """
    try:
        response = (
            supabase.table(PATIENT_HISTORIES_TABLE)
            .select("*")
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to query patient_histories: {exc}") from exc

    return WaitingRoomResponse(histories=response.data)
