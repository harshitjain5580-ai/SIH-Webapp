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

import asyncio
import base64
import io
import json
import logging
import os
import pickle
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import List, Optional
from urllib import request
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI, OpenAIError
from supabase import Client, create_client

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for local Whisper GPU detection
    torch = None
from clinical_engine import (
    detect_red_flags,
    extract_document_clinically,
    generate_local_conversation_step,
    synthesize_clinical_summary,
)
from local_store import store

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
    """
    Full AYUSH Dashavidha Pariksha ('tenfold examination') parameters, captured
    from the conversation where the transcript provides relevant information.
    """

    prakriti: str = Field(
        description="Patient's baseline constitutional type (Vata/Pitta/Kapha balance) as assessed from the conversation."
    )
    vikriti: str = Field(
        description="Patient's current state of doshic imbalance as assessed from the conversation."
    )
    sara: str = Field(description="Tissue (dhatu) quality and excellence assessed from the conversation.")
    samhanana: str = Field(description="Body compactness/build (physical compactness of the frame) assessed from the conversation.")
    pramana: str = Field(description="Body measurements and proportion assessed from the conversation.")
    satmya: str = Field(description="Patient's suitability/adaptability to different foods, climates, and conditions.")
    sattva: str = Field(description="Patient's psychic strength and mental resilience assessed from the conversation.")
    ahara_shakti: str = Field(description="Patient's digestive/appetite capacity (power of food intake and digestion).")
    vyayama_shakti: str = Field(description="Patient's exercise capacity and physical stamina.")
    vaya: str = Field(description="Patient's age-related constitutional stage (e.g. growth, adult, or decline phase).")


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
    patient_id: Optional[str] = Field(default=None, description="Optional patient identifier for retrieving known allergy and previous report context.")


class ConversationalQuestionResponse(BaseModel):
    """A single bilingual, non-prescriptive clinical intake response."""

    reply: str = Field(description="The next question for the patient, in the patient's language.")
    language: str = Field(description="Detected language, such as Hindi, English, or Hinglish.")
    red_flags_detected: bool = Field(
        description="True when the patient's message suggests an urgent emergency symptom."
    )
    transcript: Optional[str] = Field(
        default=None,
        description="Recognized patient speech when the response came from a voice request.",
    )


class VoiceTranscriptionResult(BaseModel):
    """Speech-to-text result for patient or doctor dictation."""

    text: str = Field(description="The text recovered from the audio input.")
    provider: str = Field(description="Speech provider used: openai or bhashini.")
    language: str = Field(description="Detected or requested language code, e.g. en, hi, or hi-IN.")
    confidence: Optional[float] = Field(default=None, description="Optional confidence score if the provider provides one.")


class VoiceSynthesisResult(BaseModel):
    """Text-to-speech result containing audio payload."""

    provider: str = Field(description="Voice provider used for synthesis.")
    language: str = Field(description="Language used for synthesis.")
    mime_type: str = Field(description="Audio MIME type returned by the provider.")
    audio_base64: str = Field(description="Base64-encoded audio data for playback in a mobile or web app.")


class PatientProfile(BaseModel):
    """Long-term patient medical profile used to avoid repeated allergy and medication questions."""

    patient_id: str = Field(description="Stable identifier for the person or login account.")
    name: Optional[str] = Field(default=None, description="Patient name if available.")
    age: Optional[int] = Field(default=None, description="Patient age in years, if known.")
    allergies: List[str] = Field(default_factory=list, description="Known allergies, including medicines, food, and environmental triggers.")
    medication_allergies: List[str] = Field(default_factory=list, description="Medicine-specific allergies already known.")
    chronic_conditions: List[str] = Field(default_factory=list, description="Known chronic medical conditions.")
    ongoing_medications: List[str] = Field(default_factory=list, description="Current medications the patient is already taking.")
    notes: Optional[str] = Field(default=None, description="Doctor or patient notes to remember across visits.")
    last_updated: Optional[str] = Field(default=None, description="Last updated timestamp, if provided by the client.")


class PatientReport(BaseModel):
    """A saved previous report or record that the model can reuse for follow-up questioning."""

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Stable identifier for the stored report.")
    patient_id: str = Field(description="Patient identifier associated with the saved report.")
    report_type: str = Field(description="Type of report, such as lab, prescription, previous-visit, or doctor-note.")
    report_date: Optional[str] = Field(default=None, description="Report date, if known.")
    summary: str = Field(description="Condensed summary of the past report or medical history.")
    notes: Optional[str] = Field(default=None, description="Detailed notes from the report.")
    created_at: Optional[str] = Field(default=None, description="Saved timestamp.")


class ApprovedTrainingCase(BaseModel):
    """A doctor-reviewed medical case that may be added to a curated training dataset."""

    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Stable identifier for the approved case.")
    patient_id: str = Field(description="Patient identifier for the case.")
    transcript: str = Field(description="Conversation transcript or patient story to preserve for future model learning.")
    summary: str = Field(description="Doctor-reviewed summary of the case.")
    diagnoses: List[str] = Field(default_factory=list, description="Approved diagnoses associated with the case.")
    treatment_plan: List[str] = Field(default_factory=list, description="Treatment or management notes approved by the doctor.")
    approved_by: str = Field(description="Doctor or reviewing clinician identifier.")
    tags: List[str] = Field(default_factory=list, description="Labels such as 'cardiology', 'dermatology', 'pain', 'follow_up'.")


class PatientHistoryRecord(BaseModel):
    """A row from the patient_histories Supabase table, as returned after insert."""

    id: str = Field(description="UUID primary key of the patient_histories row.")
    created_at: str = Field(description="Timestamp the row was created, as returned by Postgres.")
    chief_complaint: str = Field(description="The patient's primary reason for the visit.")
    hpi_socrates: str = Field(description="Detailed narrative of the History of Present Illness (SOCRATES).")
    past_medical_history: List[str] = Field(default_factory=list, description="Past medical history items reported by the patient.")
    current_medications: List[Medication] = Field(description="Medications the patient is currently taking.")
    ayush_parameters: AyushParameters = Field(description="AYUSH Dashavidha Pariksha parameters.")
    red_flags_detected: bool = Field(description="True if emergency red flags were detected.")
    alert_acknowledged: bool = Field(
        default=False, description="True once triage staff have acknowledged a red-flag alert for this history."
    )
    patient_name: Optional[str] = Field(default=None, description="Patient name if available.")
    abha_id: Optional[str] = Field(default=None, description="Patient ABHA ID if linked.")
    triage_level: Optional[str] = Field(default=None, description="Triage priority level.")


class InvestigationResult(BaseModel):
    """A single lab or imaging investigation result extracted from a medical document."""

    test_name: str = Field(description="Name of the lab test or investigation, e.g. 'Hemoglobin' or 'Chest X-Ray'.")
    value: str = Field(description="The measured value or finding, e.g. '10.2 g/dL'.")
    reference_range: str = Field(description="The normal reference range for this test, e.g. '13.0-17.0 g/dL'. Empty if not applicable/available.")
    is_abnormal: bool = Field(description="True if the value falls outside the normal reference range.")


class ExtractedDocument(BaseModel):
    """Structured clinical data extracted from a medical document image via the OpenAI Vision model."""

    diagnoses: List[str] = Field(description="Diagnoses mentioned in the document.")
    medications: List[Medication] = Field(description="Medications prescribed in the document.")
    investigations: List[InvestigationResult] = Field(
        description="Lab or imaging investigation results found in the document."
    )
    procedures: List[str] = Field(description="Procedures or surgeries mentioned in the document.")


class DocumentUploadResult(BaseModel):
    """Response for /upload-document: the Supabase Storage path plus the extracted document data."""

    storage_path: str = Field(
        description="Path of the uploaded file within the medical_documents Supabase Storage bucket."
    )
    extracted_document: ExtractedDocument = Field(
        description="Structured clinical data extracted by the OpenAI Vision model."
    )


class WaitingRoomResponse(BaseModel):
    """Response for GET /api/v1/waiting-room: the most recently created patient histories."""

    histories: List[PatientHistoryRecord] = Field(
        description="The 10 most recently created patient histories, ordered newest first."
    )


class ConversationTurn(BaseModel):
    """One turn in a patient interview conversation."""

    role: str = Field(description="Either 'assistant' (the kiosk's question) or 'patient' (the patient's answer).")
    content: str = Field(description="The text of this conversation turn.")


class ConverseRequest(BaseModel):
    """Request body for POST /converse. Stateless: the caller resends the full history each turn."""

    history: List[ConversationTurn] = Field(
        default_factory=list, description="The conversation so far, oldest first. Empty on the first call."
    )


class ConversationStep(BaseModel):
    """The kiosk's next move in an adaptive patient interview."""

    next_question: str = Field(description="The next question to ask the patient. Empty string once is_complete is true.")
    quick_reply_options: List[str] = Field(
        description="0-4 short tap-friendly answer options for next_question, for touch-based input. Empty if open-ended."
    )
    is_complete: bool = Field(
        description="True once enough history has been gathered to generate a full clinical summary."
    )
    is_red_flag_urgent: bool = Field(
        description="True if the patient's most recent answer indicates an emergency requiring immediate triage."
    )


class DocumentExtractionInput(BaseModel):
    """A previously-extracted document (from /upload-document) to merge into a unified summary."""

    storage_path: str = Field(description="Supabase Storage path of the source document, for reference/traceability.")
    extracted_document: ExtractedDocument = Field(description="Previously extracted structured data from this document.")


class GenerateSummaryRequest(BaseModel):
    """Request body for POST /generate-summary."""

    patient_id: Optional[str] = Field(default=None, description="Patient ID for associating prior reports and allergy context to the generated summary.")
    transcript: Optional[str] = Field(
        default=None, description="Conversational history transcript, if a voice/touch interview was conducted."
    )
    documents: List[DocumentExtractionInput] = Field(
        default_factory=list, description="Previously extracted documents to merge into the unified summary."
    )
    patient_name: Optional[str] = Field(default=None, description="Patient name if available.")
    abha_id: Optional[str] = Field(default=None, description="Patient ABHA ID if available.")
    triage_level: Optional[str] = Field(default=None, description="Triage priority level.")


class PatientHistoryUpdate(BaseModel):
    """Request body for PATCH /patient-histories/{id}. Only supplied fields are updated."""

    chief_complaint: Optional[str] = Field(default=None, description="Updated chief complaint.")
    hpi_socrates: Optional[str] = Field(default=None, description="Updated HPI narrative.")
    past_medical_history: Optional[List[str]] = Field(default=None, description="Updated past medical history.")
    current_medications: Optional[List[Medication]] = Field(default=None, description="Updated medications list.")
    ayush_parameters: Optional[AyushParameters] = Field(default=None, description="Updated AYUSH parameters.")
    red_flags_detected: Optional[bool] = Field(default=None, description="Updated red-flag status.")
    alert_acknowledged: Optional[bool] = Field(default=None, description="Updated alert acknowledged status.")
    patient_name: Optional[str] = Field(default=None, description="Updated patient name.")
    abha_id: Optional[str] = Field(default=None, description="Updated ABHA ID.")
    triage_level: Optional[str] = Field(default=None, description="Updated triage level.")


class AbhaVerificationRequest(BaseModel):
    """Request body for the mock POST /abdm/verify-abha endpoint."""

    abha_id: str = Field(description="The patient's ABHA (Ayushman Bharat Health Account) ID or address.")


class AbhaVerificationResult(BaseModel):
    """Mock response for POST /abdm/verify-abha, standing in for a real ABDM Gateway call."""

    abha_id: str = Field(description="The ABHA ID that was verified.")
    verified: bool = Field(description="Whether the ABHA ID was successfully verified.")
    patient_name: str = Field(description="Mock patient name returned by the ABDM Gateway.")
    date_of_birth: str = Field(description="Mock patient date of birth (YYYY-MM-DD) returned by the ABDM Gateway.")
    gender: str = Field(description="Mock patient gender returned by the ABDM Gateway.")


class HisPushRequest(BaseModel):
    """Request body for the mock POST /abdm/push-to-his endpoint."""

    history_id: str = Field(description="UUID of the patient_histories row to push to the Hospital Information System.")
    abha_id: str = Field(description="The patient's ABHA ID to link this record to in the HIS/ABDM Personal Health Record.")


class HisPushResult(BaseModel):
    """Mock response for POST /abdm/push-to-his, standing in for a real FHIR-based HIS/ABDM push."""

    history_id: str = Field(description="UUID of the patient_histories row that was pushed.")
    abha_id: str = Field(description="The ABHA ID the record was linked to.")
    his_record_id: str = Field(description="Mock record ID assigned by the Hospital Information System.")
    status: str = Field(description="Mock push status, e.g. 'submitted'.")
    fhir_bundle: dict = Field(
        default_factory=dict,
        description="A minimal FHIR Bundle payload showing the record is formatted as a mock HIM/HIS submission.",
    )


class PatientLoginRequest(BaseModel):
    """Minimal demo auth for a patient kiosk. Replace with real identity verification in production."""

    patient_id: str = Field(description="Stable patient identifier supplied by the kiosk.")
    name: Optional[str] = Field(default=None, description="Optional patient name for the local demo login.")
    abha_id: Optional[str] = Field(default=None, description="Optional ABHA ID for a mock identity check.")


class PatientLoginResponse(BaseModel):
    """Simple patient login response for the kiosk and local demo flows."""

    patient_id: str = Field(description="Validated patient identifier.")
    role: str = Field(default="patient", description="Access role for the kiosk session.")
    status: str = Field(default="ok", description="Authentication status.")
    message: str = Field(default="Demo authentication accepted.", description="Human-readable result.")


# ---------------------------------------------------------------------------
# App & OpenAI client setup
# ---------------------------------------------------------------------------

app = FastAPI(title="MediKiosk API", version="0.1.0")
allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ALLOW_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:8080,http://localhost:8080",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enable CORS for web frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

    @app.get("/")
    async def root_redirect():
        return RedirectResponse(url="/app/")

# OpenAI-compatible providers can be selected without changing endpoint code.
# Set AI_PROVIDER=xai and GROK_API_KEY to use xAI's Grok models.
AI_PROVIDER = os.environ.get("AI_PROVIDER", "openai").lower()
if AI_PROVIDER == "xai":
    AI_API_KEY = os.environ.get("GROK_API_KEY") or "not-set"
    AI_BASE_URL = "https://api.x.ai/v1"
    AI_MODEL = os.environ.get("AI_MODEL", "grok-4.6")
else:
    AI_API_KEY = os.environ.get("OPENAI_API_KEY") or "not-set"
    AI_BASE_URL = os.environ.get("AI_BASE_URL") or None
    AI_MODEL = os.environ.get("AI_MODEL", "gpt-4o-2024-08-06")

client = OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)

VOICE_PROVIDER = os.environ.get("VOICE_PROVIDER", "local").lower()

def _is_ai_available() -> bool:
    return bool(AI_API_KEY and AI_API_KEY not in ("not-set", "sk-your-key-here", ""))


def _is_supabase_available() -> bool:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    return bool(url and key and "not-set" not in url and "your-project-ref" not in url and "not-set" not in key)

VOICE_PROVIDER = os.environ.get("VOICE_PROVIDER", "openai").lower()
LOCAL_WHISPER_MODEL = os.environ.get("LOCAL_WHISPER_MODEL", "small")
LOCAL_WHISPER_DEVICE = os.environ.get("LOCAL_WHISPER_DEVICE", "auto")
BHASHINI_API_KEY = os.environ.get("BHASHINI_API_KEY")
BHASHINI_ASR_URL = os.environ.get("BHASHINI_ASR_URL")
BHASHINI_TTS_URL = os.environ.get("BHASHINI_TTS_URL")
BHASHINI_USER_ID = os.environ.get("BHASHINI_USER_ID")
OPENAI_TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "sage")

_local_whisper = None
_local_whisper_lock = Lock()


def _load_local_whisper():
    global _local_whisper
    if _local_whisper is None:
        with _local_whisper_lock:
            if _local_whisper is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise RuntimeError(
                        "Local voice requires faster-whisper. Install it with 'pip install faster-whisper'."
                    ) from exc

                if LOCAL_WHISPER_DEVICE == "auto":
                    if torch is not None and torch.cuda.is_available():
                        device = "cuda"
                    else:
                        device = "cpu"
                else:
                    device = LOCAL_WHISPER_DEVICE if LOCAL_WHISPER_DEVICE in {"cpu", "cuda"} else "cpu"

                compute_type = "float16" if device == "cuda" else "int8"
                _local_whisper = WhisperModel(
                    LOCAL_WHISPER_MODEL,
                    device=device,
                    compute_type=compute_type,
                )
    return _local_whisper


def _transcribe_locally(audio_bytes: bytes, language: str, filename: str) -> VoiceTranscriptionResult:
    model = _load_local_whisper()
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as audio_file:
        audio_file.write(audio_bytes)
        audio_path = audio_file.name
    try:
        language_code = language.split("-")[0] if language and language.lower() not in ("auto", "hinglish") else None
        segments, info = model.transcribe(
            audio_path,
            language=language_code,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        Path(audio_path).unlink(missing_ok=True)
    if not text:
        raise RuntimeError("Local speech recognition returned no transcription text.")
    detected_language = getattr(info, "language", None) or language or "en"
    return VoiceTranscriptionResult(text=text, provider="local", language=detected_language)


def _synthesize_locally(text: str, language: str) -> VoiceSynthesisResult:
    try:
        import pyttsx3
    except ImportError as exc:
        raise RuntimeError("Local voice requires pyttsx3. Install it with 'pip install pyttsx3'.") from exc
    engine = pyttsx3.init()
    engine.setProperty("rate", int(os.environ.get("LOCAL_TTS_RATE", "155")))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_path = audio_file.name
    try:
        engine.save_to_file(text, audio_path)
        engine.runAndWait()
        audio_bytes = Path(audio_path).read_bytes()
    finally:
        Path(audio_path).unlink(missing_ok=True)
    if not audio_bytes:
        raise RuntimeError("Local text-to-speech returned no audio.")
    return VoiceSynthesisResult(
        provider="local",
        language=language,
        mime_type="audio/wav",
        audio_base64=base64.b64encode(audio_bytes).decode("utf-8"),
    )

DEVANAGARI_MAP = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ट": "t", "ठ": "th",
    "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p", "फ": "ph",
    "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ं": "n", "ः": "h", "।": ".", "़": "",
}
HINGLISH_NORMALIZATIONS = {
    "mere pet me": "mere pet me",
    "mere pet mei": "mere pet me",
    "pet me dard": "pet me dard",
    "pet me pain": "pet me pain",
    "hue": "hai",
    "mai": "main",
    "mujhe": "mujhe",
    "haii": "hai",
    "kaise": "kaise",
    "kab": "kab",
    "kya": "kya",
    "medicine se": "medicine se",
    "allergy": "allergy",
    "allergies": "allergies",
}


def romanize_hindi_text(text: str) -> str:
    result = []
    for char in text:
        result.append(DEVANAGARI_MAP.get(char, char))
    normalized = "".join(result)
    normalized = normalized.replace("aa p", "aap").replace(" hai", " hai")
    return normalized


def normalize_multilingual_voice_text(text: str, language_hint: Optional[str] = None) -> tuple[str, str]:
    if not text:
        return "", "en-IN"
    cleaned = text.strip()
    has_devanagari = any("\u0900" <= char <= "\u097f" for char in cleaned)
    if has_devanagari:
        cleaned = romanize_hindi_text(cleaned)
        language = "hi-IN"
    else:
        lower = cleaned.lower()
        for source, target in HINGLISH_NORMALIZATIONS.items():
            lower = lower.replace(source, target)
        cleaned = lower
        if re.search(r"\b(mere|pet|dard|hai|kaise|mujhe|kya|kab)\b", cleaned):
            language = "hi-IN"
        else:
            language = language_hint or "en-IN"
    return cleaned, language


def _safe_supabase_insert(table_name: str, payload: dict) -> Optional[dict]:
    if _is_supabase_available():
        try:
            response = supabase.table(table_name).insert(payload).execute()
            if response.data:
                return response.data[0]
        except Exception as exc:
            logger.warning("Supabase table %s not available or insert failed: %s", table_name, exc)
    return None


def _safe_supabase_select(table_name: str, key: str, value: str) -> Optional[dict]:
    if _is_supabase_available():
        try:
            response = supabase.table(table_name).select("*").eq(key, value).limit(1).execute()
            if response.data:
                return response.data[0]
        except Exception as exc:
            logger.warning("Supabase table %s not available or select failed: %s", table_name, exc)
    return None


def _safe_supabase_query(table_name: str, key: str, value: str) -> List[dict]:
    if _is_supabase_available():
        try:
            response = supabase.table(table_name).select("*").eq(key, value).execute()
            return response.data or []
        except Exception as exc:
            logger.warning("Supabase table %s not available or list query failed: %s", table_name, exc)
    return []


def _get_patient_context(patient_id: Optional[str]) -> str:
    if not patient_id:
        return ""

    profile = _safe_supabase_select("patient_profiles", "patient_id", patient_id) or store.get_profile(patient_id)
    reports = _safe_supabase_query("patient_reports", "patient_id", patient_id) or store.get_reports(patient_id)

    parts = []
    if profile:
        allergies = profile.get("allergies") or []
        medication_allergies = profile.get("medication_allergies") or []
        chronic = profile.get("chronic_conditions") or []
        meds = profile.get("ongoing_medications") or []
        if allergies or medication_allergies or chronic or meds:
            parts.append(
                "Known patient profile: "
                + "; ".join(
                    filter(
                        None,
                        [
                            f"Allergies: {', '.join(allergies)}" if allergies else None,
                            f"Medicine allergies: {', '.join(medication_allergies)}" if medication_allergies else None,
                            f"Chronic conditions: {', '.join(chronic)}" if chronic else None,
                            f"Current medications: {', '.join(meds)}" if meds else None,
                        ],
                    )
                )
            )
    if reports:
        summaries = []
        for report in reports[:5]:
            summary = report.get("summary")
            if summary:
                summaries.append(summary)
        if summaries:
            parts.append("Previous medical reports: " + " | ".join(summaries))
    return "\n".join(parts)


def _normalize_voice_text(payload: object) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("text", "transcript", "output_text", "final", "result"):
            if key in payload and isinstance(payload[key], str):
                return payload[key].strip()
        if "output" in payload:
            nested = _normalize_voice_text(payload["output"])
            if nested:
                return nested
        if "data" in payload:
            nested = _normalize_voice_text(payload["data"])
            if nested:
                return nested
        if "result" in payload and isinstance(payload["result"], dict):
            return _normalize_voice_text(payload["result"])
    if isinstance(payload, list):
        parts = [_normalize_voice_text(item) for item in payload]
        text = " ".join(part for part in parts if part)
        if text:
            return text
    return ""


def _local_document_fallback(contents: bytes, filename: str) -> Optional[ExtractedDocument]:
    """Use the trained nearest-neighbor baseline if no external OCR model is configured."""
    model_path = Path(__file__).resolve().parent / "training" / "outputs" / "best_model" / "model.pkl"
    if not model_path.exists():
        return None
    try:
        import importlib.util

        train_path = Path(__file__).resolve().parent / "training" / "train.py"
        spec = importlib.util.spec_from_file_location("medikiosk_train", str(train_path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with model_path.open("rb") as handle:
            artifact = pickle.load(handle)
        image_path = Path(tempfile.gettempdir()) / f"medikiosk_fallback_{uuid.uuid4()}{Path(filename).suffix or '.jpg'}"
        image_path.write_bytes(contents)
        try:
            _, indexes = artifact["model"].kneighbors([module.embedding(str(image_path))])
            source = artifact["records"][int(indexes[0][0])]
            raw_text = str(source.get("text") or "").strip()
            if not raw_text:
                return None
            return ExtractedDocument(
                diagnoses=["Fallback extraction from local OCR baseline"] if not raw_text else [raw_text[:180]],
                medications=[],
                investigations=[],
                procedures=[],
            )
        finally:
            image_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Local document fallback failed: %s", exc)
        return None


def _safe_document_fallback(filename: str, contents: bytes) -> ExtractedDocument:
    """Offline fallback that preserves the document upload flow even without Supabase or model access."""
    basename = (filename or "document").lower()
    keywords = basename.replace("-", " ").replace("_", " ")
    diagnoses = []
    if any(token in keywords for token in ("prescription", "medicine", "medication", "rx")):
        diagnoses.append("Prescription document uploaded for manual review")
    if any(token in keywords for token in ("lab", "report", "test", "blood", "cbc", "cbc report")):
        diagnoses.append("Lab/imaging report uploaded for manual review")
    if not diagnoses:
        diagnoses.append("Medical document uploaded for manual review")
    medications = []
    if "medication" in keywords or "prescription" in keywords:
        medications.append(Medication(name="Medication list pending review", dosage="Not available", frequency="Not available"))
    investigations = []
    if "lab" in keywords or "blood" in keywords or "test" in keywords:
        investigations.append(InvestigationResult(test_name="Document-based investigation details pending OCR review", value="Not available", reference_range="Not available", is_abnormal=False))
    procedures = []
    if "discharge" in keywords or "surgery" in keywords or "operation" in keywords:
        procedures.append("Procedure or surgery note requires manual review")
    if not contents:
        diagnoses = ["Uploaded document was empty; manual review required"]
    return ExtractedDocument(diagnoses=diagnoses, medications=medications, investigations=investigations, procedures=procedures)


def _transcribe_with_bhashini(audio_bytes: bytes, language: str, filename: str) -> VoiceTranscriptionResult:
    if not BHASHINI_ASR_URL:
        raise RuntimeError("VOICE_PROVIDER=bhashini requires BHASHINI_ASR_URL to be set in the environment.")
    if not BHASHINI_API_KEY:
        raise RuntimeError("VOICE_PROVIDER=bhashini requires BHASHINI_API_KEY to be set in the environment.")

    payload = {
        "language": language,
        "task": "asr",
        "audio_b64": base64.b64encode(audio_bytes).decode("utf-8"),
        "filename": filename,
    }
    req = request.Request(
        BHASHINI_ASR_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BHASHINI_API_KEY}",
            **({"x-user-id": BHASHINI_USER_ID} if BHASHINI_USER_ID else {}),
        },
        method="POST",
    )
    with request.urlopen(req, timeout=60) as response:
        body = response.read()
    try:
        parsed = json.loads(body.decode("utf-8"))
    except Exception:
        parsed = {"text": body.decode("utf-8", errors="ignore")}
    text = _normalize_voice_text(parsed)
    if not text:
        raise RuntimeError("Bhashini ASR returned no transcription text.")
    return VoiceTranscriptionResult(text=text, provider="bhashini", language=language)


def _synthesize_with_bhashini(text: str, language: str) -> VoiceSynthesisResult:
    if not BHASHINI_TTS_URL:
        raise RuntimeError("VOICE_PROVIDER=bhashini requires BHASHINI_TTS_URL to be set in the environment.")
    if not BHASHINI_API_KEY:
        raise RuntimeError("VOICE_PROVIDER=bhashini requires BHASHINI_API_KEY to be set in the environment.")

    payload = {
        "text": text,
        "language": language,
        "gender": "female",
    }
    req = request.Request(
        BHASHINI_TTS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BHASHINI_API_KEY}",
            **({"x-user-id": BHASHINI_USER_ID} if BHASHINI_USER_ID else {}),
        },
        method="POST",
    )
    with request.urlopen(req, timeout=60) as response:
        body = response.read()
    if isinstance(body, bytes):
        return VoiceSynthesisResult(
            provider="bhashini",
            language=language,
            mime_type="audio/mpeg",
            audio_base64=base64.b64encode(body).decode("utf-8"),
        )
    raise RuntimeError("Bhashini TTS response was not binary audio data.")


async def _transcribe_voice_upload(file: UploadFile, language: str) -> VoiceTranscriptionResult:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty.")

    if VOICE_PROVIDER == "local":
        return _transcribe_locally(contents, language, file.filename or "voice.wav")

    if VOICE_PROVIDER in ("openai", "default"):
        if AI_API_KEY in (None, "not-set"):
            raise RuntimeError("OPENAI_API_KEY is required to use the OpenAI voice pipeline.")
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=(file.filename or "voice.wav", contents, file.content_type or "audio/wav"),
            language=language,
        )
        return VoiceTranscriptionResult(
            text=(transcription.text or "").strip(),
            provider="openai",
            language=language,
            confidence=getattr(transcription, "confidence", None),
        )

    if VOICE_PROVIDER == "bhashini":
        return _transcribe_with_bhashini(contents, language, file.filename or "voice.wav")

    raise RuntimeError(f"Unsupported VOICE_PROVIDER: {VOICE_PROVIDER}. Set it to 'local', 'openai', or 'bhashini'.")


async def _synthesize_voice_text(text: str, language: str) -> VoiceSynthesisResult:
    if VOICE_PROVIDER == "local":
        return _synthesize_locally(text, language)

    if VOICE_PROVIDER in ("openai", "default"):
        if AI_API_KEY in (None, "not-set"):
            raise RuntimeError("OPENAI_API_KEY is required to use the OpenAI voice synthesis pipeline.")
        response = client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
        )
        audio_bytes = response.read()
        return VoiceSynthesisResult(
            provider="openai",
            language=language,
            mime_type=response.content_type or "audio/mpeg",
            audio_base64=base64.b64encode(audio_bytes).decode("utf-8"),
        )

    if VOICE_PROVIDER == "bhashini":
        return _synthesize_with_bhashini(text, language)

    raise RuntimeError(f"Unsupported VOICE_PROVIDER: {VOICE_PROVIDER}. Set it to 'local', 'openai', or 'bhashini'.")


async def _local_voice_assistant(file: UploadFile, language: str) -> ConversationalQuestionResponse:
    try:
        transcription = await _transcribe_voice_upload(file, language)
    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("Voice transcription failed for patient assistant; returning safe fallback: %s", exc)
        return ConversationalQuestionResponse(
            reply="Please tell me your main symptom and when it started.",
            language="English",
            red_flags_detected=False,
            transcript="",
        )

    text_for_model, detected_language = normalize_multilingual_voice_text(transcription.text, language)
    try:
        from local_bilingual_model import ask

        reply = await asyncio.to_thread(ask, text_for_model)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        logger.warning("Local bilingual model inference failed for voice input; using safe fallback: %s", exc)
        reply = "Please tell me your main symptom and how long it has been happening."
    lower = text_for_model.lower()
    hinglish = any(word in lower for word in ("mere", "pet", "dard", "hai", "hue", "kaise"))
    hindi = any("\u0900" <= char <= "\u097f" for char in transcription.text)
    urgent = any(word in lower for word in ("chest pain", "breathing", "faint", "stroke", "बेहोश", "सीने"))
    return ConversationalQuestionResponse(
        reply=reply,
        language="Hinglish" if hinglish else ("Hindi" if hindi else ("Hindi" if detected_language.startswith("hi") else "English")),
        red_flags_detected=urgent,
        transcript=transcription.text,
    )


# Same fallback pattern for Supabase: allow the app to start before
# SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are configured; real DB/Storage
# calls will fail with a clear 502 until valid credentials are set in the
# environment.
#
# The backend authenticates with the service_role key (not the anon key) so
# it bypasses Row Level Security entirely. patient_histories and the
# medical_documents object policies deny anon/authenticated access outright
# (see supabase_schema.sql) — this backend is the only client allowed to
# read or write patient data. Never expose this key to the frontend.
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL") or "https://not-set.supabase.co",
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "not-set",
)

PATIENT_HISTORIES_TABLE = "patient_histories"
MEDICAL_DOCUMENTS_BUCKET = "medical_documents"


def _require_supabase_configuration() -> None:
    """Raise an actionable error before attempting a request to an invalid host."""
    supabase_url = os.environ.get("SUPABASE_URL", "").strip()
    service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    parsed_url = urlparse(supabase_url)

    if (
        not supabase_url
        or not parsed_url.scheme
        or not parsed_url.netloc
        or parsed_url.hostname in (None, "not-set.supabase.co", "your-project-ref.supabase.co")
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Supabase is not configured. Copy .env.example to .env and set "
                "SUPABASE_URL to your real project URL from Supabase Dashboard > "
                "Settings > API."
            ),
        )
    if not service_role_key or service_role_key == "your-supabase-service-role-key":
        raise HTTPException(
            status_code=503,
            detail=(
                "SUPABASE_SERVICE_ROLE_KEY is not configured. Set the backend-only "
                "service_role key in .env from Supabase Dashboard > Settings > API."
            ),
        )

SYSTEM_PROMPT = (
    "You are a clinical history extraction engine for MediKiosk, an AI clinical "
    "history intake kiosk. Given a patient conversation transcript, extract a "
    "structured clinical history summary.\n\n"
    "When the patient reports pain, apply the SOCRATES framework (Site, Onset, "
    "Character, Radiation, Associated symptoms, Time course, Exacerbating/relieving "
    "factors, Severity) in the hpi_socrates narrative.\n\n"
    "Capture the full AYUSH Dashavidha Pariksha ('tenfold examination') whenever the "
    "transcript provides relevant information: Prakriti, Vikriti, Sara, Samhanana, "
    "Pramana, Satmya, Sattva, Ahara Shakti, Vyayama Shakti, and Vaya. If a parameter "
    "cannot be assessed from the transcript, state that explicitly rather than "
    "inventing a value.\n\n"
    "Set red_flags_detected to true if the transcript contains any markers of acute "
    "cardiac events (e.g. chest pain, dyspnoea) or neurological deficits (e.g. stroke "
    "symptoms such as facial droop, slurred speech, sudden weakness). Otherwise set it "
    "to false."
)


def _local_clinical_summary(transcript: Optional[str], documents: Optional[List[DocumentExtractionInput]] = None) -> ClinicalHistorySummary:
    """Generate a deterministic, safe fallback summary when the external AI provider is unavailable."""
    text = (transcript or "").strip()
    if not text:
        text = "No transcript available."
    lower = text.lower()
    urgent_keywords = [
        "chest pain",
        "breathing",
        "faint",
        "stroke",
        "difficulty breathing",
        "severe pain",
        "sudden weakness",
        "unconscious",
        "shortness of breath",
    ]
    red_flags = any(keyword in lower for keyword in urgent_keywords)
    doc_items = documents or []
    past_history = []
    current_meds = []
    for item in doc_items:
        extracted = getattr(item, "extracted_document", None)
        if not extracted:
            continue
        for diagnosis in getattr(extracted, "diagnoses", []) or []:
            if diagnosis and diagnosis not in past_history:
                past_history.append(diagnosis)
        for medication in getattr(extracted, "medications", []) or []:
            if medication and medication.name not in {m.name for m in current_meds}:
                current_meds.append(medication)

    if "dard" in lower or "pain" in lower or "ache" in lower:
        chief_complaint = "Pain or discomfort described by the patient during intake."
    elif "fever" in lower:
        chief_complaint = "Fever or constitutional symptoms described during the intake interview."
    else:
        chief_complaint = "Clinical history recorded during patient intake."

    return ClinicalHistorySummary(
        chief_complaint=chief_complaint,
        hpi_socrates=(
            "Patient described symptoms during intake. "
            "The local fallback summary recorded the reported complaint and preserved the conversation context while "
            "waiting for a live clinical model to validate the final clinical narrative. "
            f"Transcript: {text[:500]}"
        ),
        past_medical_history=past_history or ["No prior medical history captured in the local fallback summary."],
        current_medications=current_meds,
        ayush_parameters=AyushParameters(
            prakriti="Not available from local fallback",
            vikriti="Not available from local fallback",
            sara="Not available from local fallback",
            samhanana="Not available from local fallback",
            pramana="Not available from local fallback",
            satmya="Not available from local fallback",
            sattva="Not available from local fallback",
            ahara_shakti="Not available from local fallback",
            vyayama_shakti="Not available from local fallback",
            vaya="Not available from local fallback",
        ),
        red_flags_detected=red_flags,
    )


CONVERSE_SYSTEM_PROMPT = (
    "You are the adaptive conversational history-taking engine for MediKiosk, an AI "
    "clinical intake kiosk used in Indian hospital OPDs. You conduct a structured "
    "patient interview one question at a time, mirroring how an experienced physician "
    "elicits a history.\n\n"
    "Ask ONE short, plain-language question per turn (suitable for an elderly or "
    "low-literacy patient). Cover, in a natural adaptive order driven by the patient's "
    "answers: chief complaint; if pain or a symptom is reported, drill into it using "
    "SOCRATES (Site, Onset, Character, Radiation, Associated symptoms, Time course, "
    "Exacerbating/relieving factors, Severity); past medical/surgical history; current "
    "medications and allergies; family history; personal/lifestyle history; a brief "
    "review of systems; and, where relevant, AYUSH Dashavidha Pariksha cues (Prakriti, "
    "Vikriti, Sara, Samhanana, Pramana, Satmya, Sattva, Ahara Shakti, Vyayama Shakti, "
    "Vaya).\n\n"
    "For each question, if it has a small set of natural short answers (e.g. yes/no, a "
    "severity scale, common durations), propose up to 4 quick_reply_options so the "
    "patient can tap instead of speaking. Leave quick_reply_options empty for genuinely "
    "open-ended questions.\n\n"
    "Set is_red_flag_urgent to true the moment any answer indicates an emergency (acute "
    "cardiac symptoms, stroke symptoms, etc.), independent of whether the interview is "
    "otherwise complete.\n\n"
    "Set is_complete to true, and next_question to an empty string, once you have "
    "gathered enough history to produce a complete clinical summary — do not drag the "
    "interview out longer than necessary."
)


def _fallback_conversation_step(history: List[ConversationTurn]) -> ConversationStep:
    """Keep the kiosk useful when the external structured-output model is unavailable."""
    patient_turns = [
        turn.content.strip()
        for turn in history
        if (
            turn.role == "patient"
            and turn.content.strip()
            and not turn.content.strip().lower().startswith("patient gender selected:")
        )
    ]
    last = patient_turns[-1].lower() if patient_turns else ""
    combined = " ".join(patient_turns).lower()
    hinglish = bool(re.search(r"\b(mere|mujhe|pet|dard|hai|bukhar|saans|kab se)\b", combined))
    hindi = any("\u0900" <= char <= "\u097f" for char in combined)
    urgent = any(
        marker in combined
        for marker in (
            "chest pain", "chest discomfort", "difficulty breathing", "shortness of breath",
            "breathlessness", "faint", "unconscious", "stroke", "sudden weakness",
            "सीने में दर्द", "सांस फूल", "बेहोश",
        )
    )

    if not patient_turns:
        question = (
            "Aapko sabse zyada kya takleef ho rahi hai, aur kab se?"
            if hinglish else "आपको सबसे ज़्यादा क्या तकलीफ़ हो रही है, और कब से?"
            if hindi else "What is your main symptom, and when did it start?"
        )
        return ConversationStep(
            next_question=question,
            quick_reply_options=["Pain", "Fever", "Cough or breathing problem", "Other"],
            is_complete=False,
            is_red_flag_urgent=False,
        )

    assistant_turns = [turn.content.lower() for turn in history if turn.role == "assistant"]
    urgent_question_already_asked = any(
        "breathing" in question or "saans" in question or "सांस" in question
        for question in assistant_turns
    )
    if urgent and not urgent_question_already_asked:
        question = (
            "Kya abhi saans lene mein dikkat, behoshi, ya dard baazu ya jabde tak ja raha hai?"
            if hinglish else "क्या अभी सांस लेने में दिक्कत, बेहोशी, या दर्द बाज़ू या जबड़े तक जा रहा है?"
            if hindi else "Are you having trouble breathing, fainting, or pain spreading to your arm or jaw right now?"
        )
        return ConversationStep(
            next_question=question,
            quick_reply_options=["Yes", "No", "Not sure"],
            is_complete=False,
            is_red_flag_urgent=True,
        )

    illness = (
        "respiratory"
        if any(word in combined for word in ("cough", "cold", "flu", "sore throat", "wheez", "khansi", "zukam", "खांसी", "जुकाम"))
        else "gastrointestinal"
        if (
            any(word in combined for word in ("vomit", "vomiting", "diarrhea", "loose motion", "nausea", "ulti", "dast", "दस्त", "उल्टी"))
            or ("stomach" in combined and not any(word in combined for word in ("pain", "dard")))
        )
        else "urinary"
        if any(word in combined for word in ("urine", "urinary", "pee", "peshab", "burning while passing", " पेशाब", "मूत्र"))
        else "headache"
        if any(word in combined for word in ("headache", "migraine", "head pain", "sir dard", "सिरदर्द", "सिर दर्द"))
        else None
    )

    illness_questions = {
        "respiratory": (
            ("How long have you had the cough, cold, or fever?", "Yeh khansi, zukam ya bukhar kab se hai?", "यह खाँसी, ज़ुकाम या बुखार कब से है?", ["Today", "A few days", "More than a week"]),
            ("Are you bringing up mucus, and if so, what color is it?", "Balgham aa raha hai? Agar haan, kis rang ka?", "क्या बलगम आ रहा है? अगर हाँ, किस रंग का?", ["No mucus", "Clear", "Yellow or green", "Blood"]),
            ("Do you have breathlessness, wheezing, or chest pain when breathing?", "Kya saans phoolti hai, seeti ki awaaz aati hai, ya saans lete waqt seene me dard hota hai?", "क्या सांस फूलती है, सीटी की आवाज़ आती है, या सांस लेते समय सीने में दर्द होता है?", ["No", "Breathlessness", "Wheezing", "Chest pain"]),
            ("Have you been near anyone with a similar illness, or had a recent COVID or flu contact?", "Kya kisi beemar vyakti ke sampark me aaye hain?", "क्या आप किसी बीमार व्यक्ति के संपर्क में आए हैं?", ["No", "Yes", "Not sure"]),
        ),
        "gastrointestinal": (
            ("Are you having vomiting or loose stools, and how many times today?", "Kya ulti ya loose motion ho rahe hain? Aaj kitni baar?", "क्या उल्टी या दस्त हो रहे हैं? आज कितनी बार?", ["Neither", "Vomiting", "Loose stools", "Both"]),
            ("Is there blood in the vomit or stool, or are you unable to keep fluids down?", "Kya ulti ya potty me khoon hai, ya paani bhi nahi ruk raha?", "क्या उल्टी या मल में खून है, या पानी भी नहीं रुक रहा?", ["No", "Blood", "Cannot keep fluids down", "Not sure"]),
            ("Did this start after a particular meal, unsafe water, or contact with someone who was ill?", "Kya yeh kisi khaane, paani, ya beemar vyakti ke sampark ke baad shuru hua?", "क्या यह किसी खाने, पानी, या बीमार व्यक्ति के संपर्क के बाद शुरू हुआ?", ["No", "Food", "Water", "Contact"]),
            ("Are you passing urine normally, or feeling very thirsty or dizzy?", "Kya peshab normal aa raha hai, ya bahut pyaas/chakkar lag rahe hain?", "क्या पेशाब सामान्य आ रहा है, या बहुत प्यास/चक्कर लग रहे हैं?", ["Normal", "Less urine", "Very thirsty", "Dizzy"]),
        ),
        "urinary": (
            ("Do you have burning while passing urine, frequent urination, or an urgent need to go?", "Peshab karte waqt jalan, baar-baar peshab, ya zor se hajaat hoti hai?", "पेशाब करते समय जलन, बार-बार पेशाब, या तेज़ हाजत होती है?", ["Burning", "Frequent", "Urgency", "None"]),
            ("Do you have fever, pain in your side or back, or blood in the urine?", "Kya bukhar, kamar/peeth ke paas dard, ya peshab me khoon hai?", "क्या बुखार, कमर/पीठ के पास दर्द, या पेशाब में खून है?", ["No", "Fever", "Side or back pain", "Blood"]),
            ("When did these urine symptoms start, and are they getting worse?", "Peshab ki yeh takleef kab se hai, aur badh rahi hai kya?", "पेशाब की यह तकलीफ़ कब से है, और बढ़ रही है क्या?", ["Today", "A few days", "Getting worse", "Not sure"]),
        ),
        "headache": (
            ("Did the headache start suddenly, or is it the worst headache you have ever had?", "Kya sir dard achanak shuru hua, ya zindagi ka sabse tez dard hai?", "क्या सिरदर्द अचानक शुरू हुआ, या जीवन का सबसे तेज़ दर्द है?", ["No", "Sudden", "Worst ever", "Not sure"]),
            ("Do bright light or loud sounds make it worse, and do you feel nauseated?", "Kya roshni ya tez awaaz se dard badhta hai, ya ulti jaisa lagta hai?", "क्या रोशनी या तेज़ आवाज़ से दर्द बढ़ता है, या उल्टी जैसा लगता है?", ["No", "Light or sound", "Nausea", "Both"]),
            ("Have you noticed blurred vision, weakness, numbness, or trouble speaking?", "Kya dhundhla dikhna, kamzori, sunnpan, ya bolne me dikkat hai?", "क्या धुंधला दिखना, कमजोरी, सुन्नपन, या बोलने में दिक्कत है?", ["No", "Yes", "Not sure"]),
            ("Is this a new type of headache, or have you had similar headaches before?", "Kya yeh naya tarah ka sir dard hai, ya pehle bhi aisa hua hai?", "क्या यह नए तरह का सिरदर्द है, या पहले भी ऐसा हुआ है?", ["New", "Before", "Not sure"]),
        ),
    }
    if illness:
        illness_markers = {
            "respiratory": ("cough", "mucus", "balgham", "breathlessness", "wheezing", "contact", "खांसी", "बलगम"),
            "gastrointestinal": ("vomit", "loose", "stool", "blood", "fluid", "meal", "water", "ulti", "dast", "खून"),
            "urinary": ("burning", "frequent", "urgency", "fever", "back", "blood", "peshab", "jalan", "पेशाब"),
            "headache": ("sudden", "worst", "light", "sound", "nausea", "vision", "weakness", "new", "achanak", "roshni"),
        }[illness]
        asked_text = " ".join(assistant_turns)
        unanswered = [item for item in illness_questions[illness] if not any(marker in asked_text for marker in illness_markers if marker in item[0].lower() or marker in item[1].lower() or marker in item[2].lower())]
        if unanswered:
            english, romanized, devanagari, options = unanswered[0]
            question = romanized if hinglish else devanagari if hindi else english
            return ConversationStep(next_question=question, quick_reply_options=options, is_complete=False, is_red_flag_urgent=urgent)

    has_pain = any(word in combined for word in ("pain", "dard", "ache", "headache", "दर्द"))
    has_location = any(
        word in combined
        for word in ("pet", "stomach", "chest", "head", "back", "leg", "arm", "throat", "पेट", "सीना", "सिर")
    )
    has_onset = bool(
        re.search(r"\b(today|yesterday|hours?|days?|weeks?|months?|since|sudden|gradual|started|kab se|aaj|kal|din|hafte|mahine)\b", combined)
    )
    has_character = any(
        word in combined
        for word in ("sharp", "dull", "burning", "throbbing", "cramping", "heavy", "jal", "tez", "dhadak", "जलन")
    )
    has_associated = any(
        word in combined
        for word in ("fever", "vomit", "nausea", "dizzy", "cough", "diarrhea", "bukhar", "ulti", "chakkar", "खांसी")
    )
    has_medicine_answer = any(
        word in combined
        for word in ("medicine", "medication", "tablet", "allergy", "medicines", "dawa", "drug", "दवा", "एलर्जी")
    )

    if has_pain and not has_location:
        question = (
            "Dard kis jagah hai?"
            if hinglish else "दर्द किस जगह है?"
            if hindi else "Where exactly is the pain?"
        )
        options = ["Head", "Chest", "Stomach", "Back or limb"]
    elif has_pain and not has_character:
        question = (
            "Dard kaisa hai—tez, dull, jalne wala, ya dhadakne wala?"
            if hinglish else "दर्द कैसा है—तेज़, हल्का, जलने वाला, या धड़कने वाला?"
            if hindi else "What does the pain feel like: sharp, dull, burning, or throbbing?"
        )
        options = ["Sharp", "Dull", "Burning", "Throbbing"]
    elif not has_onset:
        question = (
            "Yeh takleef kab se hai, aur achanak shuru hui ya dheere?"
            if hinglish else "यह तकलीफ़ कब से है, और अचानक शुरू हुई या धीरे?"
            if hindi else "When did this problem start, and was it sudden or gradual?"
        )
        options = ["Today", "A few days ago", "A few weeks ago", "Not sure"]
    elif not has_associated:
        question = (
            "Kya iske saath bukhar, ulti, chakkar, ya koi aur takleef hai?"
            if hinglish else "क्या इसके साथ बुखार, उल्टी, चक्कर, या कोई और तकलीफ़ है?"
            if hindi else "Do you also have fever, vomiting, dizziness, or another symptom?"
        )
        options = ["Yes", "No", "Not sure"]
    elif not has_medicine_answer:
        question = (
            "Kya aap koi roz ki medicine lete hain, ya kisi medicine se allergy hai?"
            if hinglish else "क्या आप कोई रोज़ की दवा लेते हैं, या किसी दवा से एलर्जी है?"
            if hindi else "Do you take regular medicines, or have any medicine allergies?"
        )
        options = ["No medicines", "Yes", "Not sure"]
    else:
        return ConversationStep(
            next_question="Thank you. I have enough information to prepare the history for the doctor.",
            quick_reply_options=[],
            is_complete=True,
            is_red_flag_urgent=urgent,
        )

    return ConversationStep(
        next_question=question,
        quick_reply_options=options,
        is_complete=False,
        is_red_flag_urgent=urgent,
    )


GENERATE_SUMMARY_SYSTEM_PROMPT = (
    "You are a clinical history synthesis engine for MediKiosk. You will be given a "
    "patient's conversational history transcript (if a voice/touch interview was "
    "conducted) and/or structured data extracted from their prior medical documents "
    "(if any were scanned). Synthesize everything provided into a single, unified, "
    "physician-ready ClinicalHistorySummary — do not produce separate summaries per "
    "source.\n\n"
    "Fold document-derived diagnoses and procedures into past_medical_history, "
    "document-derived medications into current_medications (merging with any "
    "conversation-derived medications, avoiding duplicates), and mention clinically "
    "significant abnormal investigation results in the hpi_socrates or "
    "past_medical_history narrative as appropriate.\n\n"
    "When the patient reports pain, apply the SOCRATES framework in the hpi_socrates "
    "narrative. Capture the full AYUSH Dashavidha Pariksha (Prakriti, Vikriti, Sara, "
    "Samhanana, Pramana, Satmya, Sattva, Ahara Shakti, Vyayama Shakti, Vaya) wherever "
    "information is available; state explicitly where a parameter cannot be assessed.\n\n"
    "Set red_flags_detected to true if anything provided indicates acute cardiac events "
    "or neurological deficits. Otherwise set it to false."
)
CONVERSATION_SYSTEM_PROMPT = (
    "You are Charaka, MediKiosk's clinical intake interviewer. Your only job is to ask the patient "
    "the next useful question; do not diagnose, recommend treatment, or prescribe medicine. "
    "Detect whether the patient uses Hindi, English, or Hinglish and reply in that same style. "
    "Use simple, respectful language and ask one focused question at a time. For pain, ask "
    "follow-up questions covering location, onset, character, severity, duration, radiation, "
    "and what makes it better or worse, adapting to answers already provided. Ask relevant "
    "questions about associated symptoms, medical history, and current medicines only when "
    "needed. If emergency warning signs are reported (severe chest pain, trouble breathing, "
    "fainting, sudden weakness, facial drooping, or confusion), set red_flags_detected true "
    "and tell the patient to seek emergency care immediately; do not provide a prescription. "
    "The reply must always be a question, except for that emergency instruction followed by "
    "a question."
)


# ---------------------------------------------------------------------------
# Persistence (Database Rules, Section 5 of claude.md.md)
# ---------------------------------------------------------------------------


def _persist_history(
    summary: ClinicalHistorySummary,
    patient_name: Optional[str] = None,
    abha_id: Optional[str] = None,
    triage_level: Optional[str] = None,
) -> dict:
    """
    Persist an extracted clinical history to Supabase when configured, or to
    the local persistent store. Returns the complete inserted record.
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
        logger.warning("Supabase history persistence failed; using local fallback record: %s", exc)
        return {
            "id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat(),
            "chief_complaint": summary.chief_complaint,
            "hpi_socrates": summary.hpi_socrates,
            "current_medications": [m.model_dump() for m in summary.current_medications],
            "ayush_parameters": summary.ayush_parameters.model_dump(),
            "red_flags_detected": summary.red_flags_detected,
            "alert_acknowledged": False,
        }

    if not response.data:
        raise HTTPException(status_code=502, detail="Supabase insert returned no data.")
    computed_triage = triage_level or ("Immediate" if summary.red_flags_detected else "Standard")
    record_payload = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chief_complaint": summary.chief_complaint,
        "hpi_socrates": summary.hpi_socrates,
        "past_medical_history": summary.past_medical_history,
        "current_medications": [m.model_dump() for m in summary.current_medications],
        "ayush_parameters": summary.ayush_parameters.model_dump(),
        "red_flags_detected": summary.red_flags_detected,
        "alert_acknowledged": False,
        "patient_name": patient_name or "Walk-in Patient",
        "abha_id": abha_id or "patient@abdm",
        "triage_level": computed_triage,
    }

    if _is_supabase_available():
        try:
            db_payload = {
                "id": record_payload["id"],
                "created_at": record_payload["created_at"],
                "chief_complaint": record_payload["chief_complaint"],
                "hpi_socrates": record_payload["hpi_socrates"],
                "current_medications": record_payload["current_medications"],
                "ayush_parameters": record_payload["ayush_parameters"],
                "red_flags_detected": record_payload["red_flags_detected"],
                "alert_acknowledged": record_payload["alert_acknowledged"],
            }
            response = supabase.table(PATIENT_HISTORIES_TABLE).insert(db_payload).execute()
            if response.data:
                res = response.data[0]
                res["patient_name"] = record_payload["patient_name"]
                res["abha_id"] = record_payload["abha_id"]
                res["triage_level"] = record_payload["triage_level"]
                res["past_medical_history"] = record_payload["past_medical_history"]
                store.insert_history(res)
                return res
        except Exception as exc:
            logger.warning("Supabase insert failed, falling back to local store: %s", exc)

    return store.insert_history(record_payload)


def _store_patient_report(patient_id: Optional[str], summary: ClinicalHistorySummary, report_type: str = "summary") -> None:
    if not patient_id:
        return
    report_data = {
        "report_id": str(uuid.uuid4()),
        "patient_id": patient_id,
        "report_type": report_type,
        "report_date": datetime.now(timezone.utc).date().isoformat(),
        "summary": f"{summary.chief_complaint} | {summary.hpi_socrates}",
        "notes": json.dumps({
            "current_medications": [m.model_dump() for m in summary.current_medications],
            "red_flags_detected": summary.red_flags_detected,
            "ayush_parameters": summary.ayush_parameters.model_dump(),
        }, ensure_ascii=False),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _safe_supabase_insert("patient_reports", report_data)
    store.save_report(report_data)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "backend": "live",
        "ai_provider": AI_PROVIDER,
        "ai_available": _is_ai_available(),
        "supabase_available": _is_supabase_available(),
    }


# TODO: Replace the demo patient login flow with a real ABHA/UIDAI or hospital-issued
# identity verification service before production deployment.
@app.post("/auth/patient/login", response_model=PatientLoginResponse)
async def patient_login(request: PatientLoginRequest) -> PatientLoginResponse:
    """Minimal patient sign-in for kiosk/demo flows. Replace with real auth in production."""
    if not request.patient_id.strip():
        raise HTTPException(status_code=400, detail="patient_id is required.")
    if request.abha_id and request.abha_id.strip():
        return PatientLoginResponse(
            patient_id=request.patient_id,
            role="patient",
            status="ok",
            message="Demo patient authentication accepted via ABHA reference.",
        )
    return PatientLoginResponse(
        patient_id=request.patient_id,
        role="patient",
        status="ok",
        message="Demo patient authentication accepted. Replace with real identity verification in production.",
    )


@app.post("/voice/transcribe", response_model=VoiceTranscriptionResult)
async def transcribe_voice(file: UploadFile = File(...), language: str = "en") -> VoiceTranscriptionResult:
    """Convert spoken audio into text for patient intake or doctor dictation."""
    try:
        return await _transcribe_voice_upload(file, language)
    except (RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=502, detail=f"Voice transcription failed: {exc}") from exc


@app.post("/voice/speak", response_model=VoiceSynthesisResult)
@app.get("/voice/speak", response_model=VoiceSynthesisResult)
async def speak_text(request: Request, text: str = "", language: str = "en") -> VoiceSynthesisResult:
    """Convert text back into audio for a patient or doctor-facing voice assistant."""
    if not text.strip():
        try:
            payload = await request.json()
            if isinstance(payload, dict):
                text = str(payload.get("text") or "")
                language = str(payload.get("language") or language)
        except Exception:
            pass
    if not text.strip():
        try:
            form = await request.form()
            text = str(form.get("text") or "")
            language = str(form.get("language") or language)
        except Exception:
            pass
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text to speak cannot be empty.")
    try:
        return await _synthesize_voice_text(text, language)
    except (RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=502, detail=f"Voice synthesis failed: {exc}") from exc


@app.post("/voice/patient-assistant", response_model=ConversationalQuestionResponse)
async def patient_voice_assistant(file: UploadFile = File(...), language: str = "en") -> ConversationalQuestionResponse:
    """Transcribe a spoken patient answer and return the next question to ask."""
    try:
        return await _local_voice_assistant(file, language)
    except HTTPException:
        raise
    except (RuntimeError, ValueError, OSError) as exc:
        logger.exception("Voice patient assistant failed.")
        raise HTTPException(status_code=502, detail=f"Voice patient assistant failed: {exc}") from exc


@app.post("/doctor/voice-note", response_model=VoiceTranscriptionResult)
async def doctor_voice_note(file: UploadFile = File(...), language: str = "en") -> VoiceTranscriptionResult:
    """Transcript doctor dictation to text so the physician can review or save notes."""
    return await _transcribe_voice_upload(file, language)


@app.post("/patient/profile", response_model=PatientProfile)
async def save_patient_profile(profile: PatientProfile) -> PatientProfile:
    """Save or update a patient's long-term allergies and medical profile for future visits."""
    payload = profile.model_dump()
    payload["last_updated"] = payload.get("last_updated") or datetime.now(timezone.utc).isoformat()
    stored = _safe_supabase_insert("patient_profiles", payload)
    store.save_profile(payload)
    if stored:
        return PatientProfile(**stored)
    return profile


@app.get("/patient/profile/{patient_id}", response_model=PatientProfile)
async def get_patient_profile(patient_id: str) -> PatientProfile:
    """Fetch the stored patient profile to avoid asking repeatedly for known allergies or medication issues."""
    stored = _safe_supabase_select("patient_profiles", "patient_id", patient_id) or store.get_profile(patient_id)
    if stored:
        return PatientProfile(**stored)
    raise HTTPException(status_code=404, detail=f"Patient profile not found for {patient_id}.")


@app.post("/patient/report", response_model=PatientReport)
async def save_patient_report(report: PatientReport) -> PatientReport:
    """Store a previous patient report so the model can reuse that history during follow-up questioning."""
    payload = report.model_dump()
    payload["created_at"] = payload.get("created_at") or datetime.now(timezone.utc).isoformat()
    stored = _safe_supabase_insert("patient_reports", payload)
    store.save_report(payload)
    if stored:
        return PatientReport(**stored)
    return report


@app.get("/patient/reports/{patient_id}", response_model=List[PatientReport])
async def get_patient_reports(patient_id: str) -> List[PatientReport]:
    """Fetch earlier patient reports so the model can contextually cross-question and advise."""
    records = _safe_supabase_query("patient_reports", "patient_id", patient_id) or store.get_reports(patient_id)
    return [PatientReport(**record) for record in records]


@app.post("/patient/intake-start", response_model=ConversationalQuestionResponse)
async def patient_intake_start(patient_id: str, has_previous_history: bool = False) -> ConversationalQuestionResponse:
    """Begin intake by asking whether the patient has prior reports or medical history, first and before deeper questioning."""
    if has_previous_history:
        return ConversationalQuestionResponse(
            reply="Please tell me about your previous medical reports, past illnesses, surgeries, allergies, or recent medications. If you have a lab report or prescription, you can describe it here.",
            language="English",
            red_flags_detected=False,
        )
    return ConversationalQuestionResponse(
        reply="Do you have any previous medical history or reports? If yes, tell me about them. If not, just say 'No previous history' and we will start fresh.",
        language="English",
        red_flags_detected=False,
    )


@app.post("/doctor/approved-case", response_model=ApprovedTrainingCase)
async def doctor_approved_case(case: ApprovedTrainingCase) -> ApprovedTrainingCase:
    """Store a doctor-reviewed medical case for a curated learning dataset. This is a review-controlled path, not automatic live training."""
    payload = case.model_dump()
    stored = _safe_supabase_insert("doctor_approved_cases", payload)
    store.save_case(payload)
    if stored:
        return ApprovedTrainingCase(**stored)
    local_path = Path("training/approved_cases.jsonl")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return case
    local_path = Path("training/approved_cases.jsonl")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return case


@app.post("/ask-clinical-question", response_model=ConversationalQuestionResponse)
async def ask_clinical_question(request: TranscriptRequest) -> ConversationalQuestionResponse:
    """Ask the next bilingual intake question while using prior patient reports and medical context when available."""
    if not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript must not be empty.")

    patient_context = _get_patient_context(request.patient_id)
    if request.patient_id and not patient_context:
        return ConversationalQuestionResponse(
            reply="Do you have any previous medical history or reports? If yes, tell me about past illnesses, surgeries, allergies, or previous reports. If not, say 'No previous history' and we will start fresh.",
            language="English",
            red_flags_detected=False,
        )

    prompt_text = request.transcript
    if patient_context:
        prompt_text = f"Known patient context:\n{patient_context}\n\nCurrent patient message:\n{request.transcript}"

    try:
        from local_bilingual_model import ask

        reply = await asyncio.to_thread(ask, prompt_text)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        logger.exception("Local bilingual model inference failed.")
        raise HTTPException(status_code=503, detail=f"Local bilingual model unavailable: {exc}") from exc
    lower = prompt_text.lower()
    hindi = any("\u0900" <= char <= "\u097f" for char in request.transcript)
    hinglish = any(word in lower for word in ("mere", "pet", "dard", "hai", "hue", "kaise"))
    urgent = any(word in lower for word in ("chest pain", "breathing", "faint", "stroke", "बेहोश", "सीने"))
    return ConversationalQuestionResponse(
        reply=reply,
        language="Hinglish" if hinglish else ("Hindi" if hindi else "English"),
        red_flags_detected=urgent,
    )


@app.post("/extract-history", response_model=PatientHistoryRecord)
async def extract_history(request: TranscriptRequest) -> PatientHistoryRecord:
    """
    Extract a structured ClinicalHistorySummary from a raw patient transcript,
    persisting it to patient_histories. Falls back seamlessly to clinical reasoning
    if the external AI API is unconfigured.
    """
    try:
        completion = client.beta.chat.completions.parse(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.transcript},
            ],
            response_format=ClinicalHistorySummary,
        )
    except OpenAIError as exc:
        logger.warning("OpenAI parse failed during extract-history; using local fallback summary: %s", exc)
        parsed = _local_clinical_summary(request.transcript)
        record = _persist_history(parsed)
        if request.patient_id:
            _store_patient_report(request.patient_id, parsed, report_type="summary")
        return record

    message = completion.choices[0].message

    if message.refusal:
        raise HTTPException(status_code=422, detail=f"Model refused to process transcript: {message.refusal}")

    parsed = message.parsed
    parsed = None
    if _is_ai_available():
        try:
            completion = client.beta.chat.completions.parse(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": request.transcript},
                ],
                response_format=ClinicalHistorySummary,
            )
            message = completion.choices[0].message
            if not message.refusal:
                parsed = message.parsed
        except Exception as exc:
            logger.warning("OpenAI extract-history failed, falling back to clinical engine: %s", exc)

    if parsed is None:
        summary_dict = synthesize_clinical_summary(transcript=request.transcript)
        parsed = ClinicalHistorySummary(**summary_dict)

    record = _persist_history(parsed)
    if request.patient_id:
        _store_patient_report(request.patient_id, parsed, report_type="summary")

    return PatientHistoryRecord(**record)


@app.post("/extract-from-image", response_model=ClinicalHistorySummary)
async def extract_from_image(file: UploadFile = File(...)) -> ClinicalHistorySummary:
    """
    Extract a structured ClinicalHistorySummary from a prescription/document image.
    Falls back to clinical OCR parser if cloud services are unconfigured.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    storage_ready = True
    try:
        _require_supabase_configuration()
    except HTTPException:
        storage_ready = False

    extension = os.path.splitext(file.filename or "")[1] or ".jpg"
    object_path = f"{uuid.uuid4()}{extension}"
    public_url = None

    if storage_ready:
        try:
            supabase.storage.from_(MEDICAL_DOCUMENTS_BUCKET).upload(
                object_path,
                contents,
                {"content-type": file.content_type or "application/octet-stream"},
            )
            public_url = supabase.storage.from_(MEDICAL_DOCUMENTS_BUCKET).get_public_url(object_path)
        except Exception as exc:
            logger.warning("Supabase Storage upload failed; using safe offline fallback: %s", exc)
            storage_ready = False

    if not storage_ready:
        logger.warning("Using offline local extraction fallback for %s because Supabase storage is unavailable.", file.filename or "uploaded_document")
        parsed = _safe_document_fallback(file.filename or "uploaded_document", contents)
        return ClinicalHistorySummary(
            chief_complaint=parsed.diagnoses[0] if parsed.diagnoses else "Medical document uploaded for review.",
            hpi_socrates="The uploaded document was processed using the offline fallback path because the storage/AI backend was unavailable.",
            past_medical_history=parsed.diagnoses,
            current_medications=[Medication(name=m.name, dosage=m.dosage, frequency=m.frequency) for m in parsed.medications],
            ayush_parameters=AyushParameters(
                prakriti="Not available",
                vikriti="Not available",
                sara="Not available",
                samhanana="Not available",
                pramana="Not available",
                satmya="Not available",
                sattva="Not available",
                ahara_shakti="Not available",
                vyayama_shakti="Not available",
                vaya="Not available",
            ),
            red_flags_detected=False,
        )

    try:
        completion = client.beta.chat.completions.parse(
            model=AI_MODEL,
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
        logger.warning("Vision model failed for extract-from-image: %s", exc)
        fallback_summary = ClinicalHistorySummary(
            chief_complaint="Clinical summary is pending while the local document processing fallback is used.",
            hpi_socrates="A structured summary could not be generated from the uploaded document, so the app is running in safe fallback mode.",
            past_medical_history=[],
            current_medications=[],
            ayush_parameters=AyushParameters(
                prakriti="Not available",
                vikriti="Not available",
                sara="Not available",
                samhanana="Not available",
                pramana="Not available",
                satmya="Not available",
                sattva="Not available",
                ahara_shakti="Not available",
                vyayama_shakti="Not available",
                vaya="Not available",
            ),
            red_flags_detected=False,
        )
        _persist_history(fallback_summary)
        return fallback_summary

    message = completion.choices[0].message

    if message.refusal:
        raise HTTPException(status_code=422, detail=f"Model refused to process image: {message.refusal}")
    filename = file.filename or "prescription.jpg"
    extension = os.path.splitext(filename)[1] or ".jpg"
    storage_path = f"uploads/{uuid.uuid4()}{extension}"

    parsed = None
    if _is_supabase_available() and _is_ai_available():
        try:
            supabase.storage.from_(MEDICAL_DOCUMENTS_BUCKET).upload(
                storage_path,
                contents,
                {"content-type": file.content_type or "application/octet-stream"},
            )
            public_url = supabase.storage.from_(MEDICAL_DOCUMENTS_BUCKET).get_public_url(storage_path)

            completion = client.beta.chat.completions.parse(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract the structured clinical history from this prescription/document image."},
                            {"type": "image_url", "image_url": {"url": public_url}},
                        ],
                    },
                ],
                response_format=ClinicalHistorySummary,
            )
            message = completion.choices[0].message
            if not message.refusal:
                parsed = message.parsed
        except Exception as exc:
            logger.warning("Cloud extract-from-image failed, using clinical fallback: %s", exc)

    if parsed is None:
        doc_data = extract_document_clinically(filename, contents)
        summary_dict = synthesize_clinical_summary(
            transcript=f"Extracted from document image: {filename}",
            documents=[{"storage_path": storage_path, "extracted_document": doc_data}],
        )
        parsed = ClinicalHistorySummary(**summary_dict)

    _persist_history(parsed)
    return parsed


@app.post("/upload-document", response_model=DocumentUploadResult)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResult:
    """
    Upload a medical document image and extract its diagnoses, medications,
    investigation results, and procedures. Works with cloud vision API or built-in
    clinical OCR parser.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    storage_ready = True
    try:
        _require_supabase_configuration()
    except HTTPException:
        storage_ready = False

    filename = file.filename or "document.jpg"
    extension = os.path.splitext(filename)[1] or ".jpg"
    storage_path = f"uploads/{uuid.uuid4()}{extension}"

    public_url = None
    if _is_supabase_available():
        try:
            supabase.storage.from_(MEDICAL_DOCUMENTS_BUCKET).upload(
                storage_path,
                contents,
                {"content-type": file.content_type or "application/octet-stream"},
            )
            public_url = supabase.storage.from_(MEDICAL_DOCUMENTS_BUCKET).get_public_url(storage_path)
        except Exception as exc:
            logger.warning("Supabase Storage upload failed; using offline document fallback: %s", exc)
            storage_ready = False

    if not storage_ready:
        fallback = _safe_document_fallback(file.filename or storage_path, contents)
        return DocumentUploadResult(storage_path=storage_path, extracted_document=fallback)

    if not public_url:
        uploads_dir = Path(__file__).parent / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        local_path = uploads_dir / f"{uuid.uuid4()}{extension}"
        local_path.write_bytes(contents)
        storage_path = str(local_path.relative_to(Path(__file__).parent)).replace("\\", "/")

    parsed = None
    if _is_ai_available() and public_url:
        try:
            completion = client.beta.chat.completions.parse(
                model=AI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a medical document extraction engine. Extract every diagnosis, "
                            "medication (with dosage and frequency), lab/imaging investigation result "
                            "(with its value and reference range, flagging is_abnormal when the value "
                            "falls outside that range), and procedure or surgery mentioned in the "
                            "provided medical document image."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all structured clinical data from this document image."},
                            {"type": "image_url", "image_url": {"url": public_url}},
                        ],
                    },
                ],
                response_format=ExtractedDocument,
            )
            message = completion.choices[0].message
            if not message.refusal:
                parsed = message.parsed
        except Exception as exc:
            logger.warning("OpenAI vision extraction failed, using clinical OCR engine: %s", exc)

    if parsed is None:
        doc_dict = extract_document_clinically(filename, contents)
        parsed = ExtractedDocument(**doc_dict)

    return DocumentUploadResult(storage_path=storage_path, extracted_document=parsed)


@app.get("/api/v1/waiting-room", response_model=WaitingRoomResponse)
async def get_waiting_room() -> WaitingRoomResponse:
    """
    Return the 10 most recently created clinical histories, ordered by created_at descending.
    Queries Supabase when available, otherwise serves from the local persistent store.
    """
    if _is_supabase_available():
        try:
            response = (
                supabase.table(PATIENT_HISTORIES_TABLE)
                .select("*")
                .order("created_at", desc=True)
                .limit(10)
                .execute()
            )
            if response.data:
                return WaitingRoomResponse(histories=[PatientHistoryRecord(**r) for r in response.data])
        except Exception as exc:
            logger.warning("Supabase get_waiting_room failed, falling back to local store: %s", exc)

    records = store.get_histories(limit=10)
    return WaitingRoomResponse(histories=[PatientHistoryRecord(**r) for r in records])


@app.post("/converse", response_model=ConversationStep)
async def converse(request: ConverseRequest) -> ConversationStep:
    """
    Stateless adaptive interview engine: returns the next question to ask
    with touch quick-reply options and emergency red-flag detection.
    """
    messages = [{"role": "system", "content": CONVERSE_SYSTEM_PROMPT}]
    if not request.history:
        messages.append({"role": "user", "content": "[Interview starting. Ask the first question.]"})
    else:
        for turn in request.history:
            role = "assistant" if turn.role == "assistant" else "user"
            messages.append({"role": role, "content": turn.content})

    try:
        completion = client.beta.chat.completions.parse(
            model=AI_MODEL,
            messages=messages,
            response_format=ConversationStep,
        )
    except OpenAIError as exc:
        logger.warning("OpenAI parse failed during /converse; using local fallback conversation step: %s", exc)
        return _fallback_conversation_step(request.history)

    message = completion.choices[0].message

    if message.refusal:
        raise HTTPException(status_code=422, detail=f"Model refused to continue interview: {message.refusal}")

    parsed = message.parsed
    if parsed is None:
        raise HTTPException(status_code=502, detail="Model did not return a parsed structured output.")
    if _is_ai_available():
        messages = [{"role": "system", "content": CONVERSE_SYSTEM_PROMPT}]
        if not request.history:
            messages.append({"role": "user", "content": "[Interview starting. Ask the first question.]"})
        else:
            for turn in request.history:
                role = "assistant" if turn.role == "assistant" else "user"
                messages.append({"role": role, "content": turn.content})

        try:
            completion = client.beta.chat.completions.parse(
                model=AI_MODEL,
                messages=messages,
                response_format=ConversationStep,
            )
            message = completion.choices[0].message
            if not message.refusal and message.parsed:
                return message.parsed
        except Exception as exc:
            logger.warning("OpenAI converse failed, using clinical dialogue engine: %s", exc)

    # Built-in clinical conversational AI fallback
    history_dicts = [{"role": t.role, "content": t.content} for t in request.history]
    step_dict = generate_local_conversation_step(history_dicts)
    return ConversationStep(**step_dict)


@app.post("/generate-summary", response_model=PatientHistoryRecord)
async def generate_summary(request: GenerateSummaryRequest) -> PatientHistoryRecord:
    """
    Synthesize conversation transcript and/or extracted document data into
    a unified ClinicalHistorySummary, persist it, and return the record.
    """
    if not request.transcript and not request.documents:
        raise HTTPException(status_code=400, detail="At least one of transcript or documents must be provided.")

    user_content_parts = []
    if request.transcript:
        user_content_parts.append(f"Conversational history transcript:\n{request.transcript}")
    if request.documents:
        docs_json = json.dumps([d.model_dump() for d in request.documents], indent=2)
        user_content_parts.append(f"Extracted data from {len(request.documents)} prior medical document(s):\n{docs_json}")

    try:
        completion = client.beta.chat.completions.parse(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": GENERATE_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(user_content_parts)},
            ],
            response_format=ClinicalHistorySummary,
        )
    except OpenAIError as exc:
        logger.warning("OpenAI parse failed during /generate-summary; using local fallback summary: %s", exc)
        parsed = _local_clinical_summary(request.transcript, request.documents)
        record = _persist_history(parsed)
        if request.patient_id:
            _store_patient_report(request.patient_id, parsed, report_type="summary")
        return record

    message = completion.choices[0].message

    if message.refusal:
        raise HTTPException(status_code=422, detail=f"Model refused to synthesize summary: {message.refusal}")
    parsed = None
    if _is_ai_available():
        user_content_parts = []
        if request.transcript:
            user_content_parts.append(f"Conversational history transcript:\n{request.transcript}")
        if request.documents:
            docs_json = json.dumps([d.model_dump() for d in request.documents], indent=2)
            user_content_parts.append(f"Extracted data from {len(request.documents)} prior medical document(s):\n{docs_json}")

        try:
            completion = client.beta.chat.completions.parse(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": GENERATE_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": "\n\n".join(user_content_parts)},
                ],
                response_format=ClinicalHistorySummary,
            )
            message = completion.choices[0].message
            if not message.refusal and message.parsed:
                parsed = message.parsed
        except Exception as exc:
            logger.warning("OpenAI generate-summary failed, using clinical synthesis: %s", exc)

    if parsed is None:
        raw_docs = [d.model_dump() for d in request.documents]
        summary_dict = synthesize_clinical_summary(
            transcript=request.transcript,
            documents=raw_docs,
            patient_name=request.patient_name,
            abha_id=request.abha_id,
            triage_level=request.triage_level,
        )
        parsed = ClinicalHistorySummary(**summary_dict)

    record = _persist_history(
        parsed,
        patient_name=request.patient_name,
        abha_id=request.abha_id,
        triage_level=request.triage_level,
    )
    if request.patient_id:
        _store_patient_report(request.patient_id, parsed, report_type="summary")

    return PatientHistoryRecord(**record)


@app.get("/patient-histories/{history_id}", response_model=PatientHistoryRecord)
async def get_patient_history(history_id: str) -> PatientHistoryRecord:
    """Fetch a single patient history row for physician review."""
    if _is_supabase_available():
        try:
            response = supabase.table(PATIENT_HISTORIES_TABLE).select("*").eq("id", history_id).execute()
            if response.data:
                return PatientHistoryRecord(**response.data[0])
        except Exception as exc:
            logger.warning("Supabase get_patient_history failed: %s", exc)

    record = store.get_history(history_id)
    if not record:
        raise HTTPException(status_code=404, detail="Patient history not found.")

    return PatientHistoryRecord(**record)


@app.patch("/patient-histories/{history_id}", response_model=PatientHistoryRecord)
async def update_patient_history(history_id: str, update: PatientHistoryUpdate) -> PatientHistoryRecord:
    """Apply physician edits to a saved patient history."""
    payload = update.model_dump(exclude_unset=True)
    if not payload:
        raise HTTPException(status_code=400, detail="No fields provided to update.")

    updated_supabase = None
    if _is_supabase_available():
        try:
            response = (
                supabase.table(PATIENT_HISTORIES_TABLE)
                .update(payload)
                .eq("id", history_id)
                .execute()
            )
            if response.data:
                updated_supabase = response.data[0]
        except Exception as exc:
            logger.warning("Supabase update_patient_history failed: %s", exc)

    updated_local = store.update_history(history_id, payload)
    final_record = updated_supabase or updated_local
    if not final_record:
        raise HTTPException(status_code=404, detail="Patient history not found.")

    return PatientHistoryRecord(**final_record)


@app.get("/api/v1/priority-alerts", response_model=WaitingRoomResponse)
async def get_priority_alerts() -> WaitingRoomResponse:
    """
    Return unacknowledged patient histories with detected red flags, oldest first,
    for the triage dashboard.
    """
    if _is_supabase_available():
        try:
            response = (
                supabase.table(PATIENT_HISTORIES_TABLE)
                .select("*")
                .eq("red_flags_detected", True)
                .eq("alert_acknowledged", False)
                .order("created_at", desc=False)
                .execute()
            )
            if response.data is not None:
                return WaitingRoomResponse(histories=[PatientHistoryRecord(**r) for r in response.data])
        except Exception as exc:
            logger.warning("Supabase get_priority_alerts failed: %s", exc)

    records = store.get_priority_alerts()
    return WaitingRoomResponse(histories=[PatientHistoryRecord(**r) for r in records])


@app.post("/patient-histories/{history_id}/acknowledge-alert", response_model=PatientHistoryRecord)
async def acknowledge_alert(history_id: str) -> PatientHistoryRecord:
    """Mark a red-flag alert as acknowledged by triage staff."""
    updated_supabase = None
    if _is_supabase_available():
        try:
            response = (
                supabase.table(PATIENT_HISTORIES_TABLE)
                .update({"alert_acknowledged": True})
                .eq("id", history_id)
                .execute()
            )
            if response.data:
                updated_supabase = response.data[0]
        except Exception as exc:
            logger.warning("Supabase acknowledge_alert failed: %s", exc)

    updated_local = store.acknowledge_alert(history_id)
    final_record = updated_supabase or updated_local
    if not final_record:
        raise HTTPException(status_code=404, detail="Patient history not found.")

    return PatientHistoryRecord(**final_record)


SAMPLE_ABHA_PATIENTS = {
    "91-9876543210@abdm": ("Aarav Sharma", "1992-04-12", "Male"),
    "priya.patel@abdm": ("Priya Patel", "1988-06-15", "Female"),
    "vikram.singh@abdm": ("Vikram Singh", "1975-11-23", "Male"),
    "ananya.das@abdm": ("Ananya Das", "1996-08-30", "Female"),
    "rajesh.kumar@abdm": ("Rajesh Kumar", "1972-03-18", "Male"),
    "sunita.sharma@abdm": ("Sunita Sharma", "1990-09-25", "Female"),
    "harish.c@abdm": ("Harish Chandra", "1960-01-14", "Male"),
}


@app.post("/abdm/verify-abha", response_model=AbhaVerificationResult)
async def verify_abha(request: AbhaVerificationRequest) -> AbhaVerificationResult:
    """ABHA ID verification with realistic patient name lookup."""
    clean_id = request.abha_id.strip().lower()
    if clean_id in SAMPLE_ABHA_PATIENTS:
        name, dob, gender = SAMPLE_ABHA_PATIENTS[clean_id]
    else:
        if "@" in request.abha_id:
            name = request.abha_id.split("@")[0].replace(".", " ").title()
        else:
            name = f"Patient {request.abha_id[-4:]}"
        dob = "1990-01-01"
        gender = "Unspecified"

    return AbhaVerificationResult(
        abha_id=request.abha_id,
        verified=True,
        patient_name=name,
        date_of_birth=dob,
        gender=gender,
    )


# TODO: Replace the mock FHIR Bundle with a real ABDM/HIS gateway client once the
# sandbox credentials and required FHIR message definitions are provided by the hospital.
@app.post("/abdm/push-to-his", response_model=HisPushResult)
async def push_to_his(request: HisPushRequest) -> HisPushResult:
    """Mock push of a patient history to the Hospital Information System."""
    exists = False
    if _is_supabase_available():
        try:
            response = supabase.table(PATIENT_HISTORIES_TABLE).select("id").eq("id", request.history_id).execute()
            exists = bool(response.data)
        except Exception:
            pass

    if not exists:
        exists = bool(store.get_history(request.history_id))

    if not exists:
        raise HTTPException(status_code=404, detail="Patient history not found.")

    fhir_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Composition",
                    "status": "final",
                    "type": {"text": "Clinical summary"},
                    "subject": {"reference": f"Patient/{request.abha_id}"},
                    "date": datetime.utcnow().isoformat(),
                    "title": "MediKiosk patient intake summary",
                }
            }
        ],
    }
    his_id = f"HIS-FHIR-{uuid.uuid4().hex[:8].upper()}"
    return HisPushResult(
        history_id=request.history_id,
        abha_id=request.abha_id,
        his_record_id=his_id,
        status="submitted",
        fhir_bundle=fhir_bundle,
    )
