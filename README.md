# MediKiosk

AI Clinical History Software Platform — backend for an AI-assisted clinical history intake kiosk.

Full architecture rules, tech stack, and the clinical data contracts live in [`claude.md.md`](claude.md.md) — read that first.

## Tech Stack
- **Backend:** Python 3.10+, FastAPI, Pydantic v2
- **AI/ML:** Local Qwen2.5-1.5B-Instruct LoRA adapter for bilingual intake; OpenAI-compatible API for document/history extraction
- **Database & Storage:** Supabase (PostgreSQL & Storage)

This repository currently provides the backend API; there is no web frontend in this checkout.

## Getting Started

1. **Clone and enter the project**
   ```bash
   git clone <this-repo-url>
   cd Webapp
   ```

2. **Create and activate a virtual environment**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   pip install -r training\requirements.txt
   ```

4. **Configure environment variables**
   ```powershell
   Copy-Item .env.example .env
   ```
   Fill in `.env` with real values if using document/history extraction:
   - `OPENAI_API_KEY` — from OpenAI, or set `AI_PROVIDER=xai` and `GROK_API_KEY` for xAI
   - `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` — from the shared Supabase project (ask a teammate for access, or see below). The backend requires the **service_role** key, not the anon key — patient data is locked down to service_role-only access (see `supabase_schema.sql`). Never share this key outside the team or commit it.

5. **Run the API**
   ```powershell
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```
   API docs available at `http://127.0.0.1:8000/docs`.

   The AI provider defaults to OpenAI. To use Grok for the existing extraction
   endpoints, set `AI_PROVIDER=xai`,
   `GROK_API_KEY`, and optionally `AI_MODEL` (for example,
   `grok-4.6`) in `.env`. Both providers use the same
   OpenAI-compatible client; no API key is committed to the repository.

   The `/ask-clinical-question` endpoint uses the local bilingual Qwen adapter
   from `training/outputs/qwen2.5-1.5b-bilingual-lora/` and does not require an
   API key. The adapter is loaded lazily on its first request.

## API usage

Open Swagger UI at `http://127.0.0.1:8000/docs`.

- `POST /extract-history` — body `{"transcript": "..."}`, extracts structured clinical history from a text transcript.
- `POST /ask-clinical-question` — body `{"transcript": "..."}`, uses the local Qwen adapter to ask one safe follow-up question in English, Hindi, or Romanized Hinglish. It does not prescribe medicine.
- `POST /extract-from-image` — multipart file upload, extracts structured clinical history from a prescription/document image (uploaded to Supabase Storage first, then read by the OpenAI Vision model).
- `GET /health` — liveness check.

Example:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/ask-clinical-question `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"transcript":"mere pet mae dard hae"}'
```

The first local-model request loads Qwen and may take a few seconds. The adapter is trained for question phrasing, not diagnosis, treatment, or clinical decision-making.

## Voice assistant support

The backend includes voice helpers for patient and doctor workflows:

- `POST /voice/transcribe` — upload an audio file and get text transcription
- `POST /voice/speak` — convert text into spoken audio
- `POST /voice/patient-assistant` — speak the patient's problem, transcribe it, and return the next question
- `POST /doctor/voice-note` — transcribe a doctor's voice note or dictation

Use `VOICE_PROVIDER=openai` for OpenAI transcription and TTS, or set `VOICE_PROVIDER=bhashini` with the Bhashini ASR/TTS URLs and API key for Indian-language voice integration.

This repository also includes a safer, controlled profile and review flow:

- `POST /patient/profile` — store allergies, chronic conditions, and medicine history for a patient
- `GET /patient/profile/{patient_id}` — fetch the known profile to avoid repeating allergy questions
- `POST /doctor/approved-case` — store a doctor-reviewed case for a curated training dataset
- `python training/build_doctor_case_dataset.py` — generate a small JSONL dataset from approved cases for future model improvement

Important: automatic self-training from every live patient interaction is not enabled in this repo. Approved, sanitized cases are stored for controlled review and retraining, which is the safer healthcare pattern.

## Learn how training works

For a beginner-friendly, step-by-step explanation of the dataset, tokenizer,
fine-tuning, LoRA adapter, training settings, inference flow, and limitations,
see [`training/README.md`](training/README.md#how-the-model-was-trained-class-10-explanation).

## Database (Supabase)

The database schema is defined in [`supabase_schema.sql`](supabase_schema.sql):
- `patient_histories` table — stores every extracted clinical history.
- `medical_documents` Storage bucket — stores uploaded prescription/document images.
- RLS is enabled on both with no anon/authenticated policies — only the backend's `service_role` key can read or write. The frontend must go through the FastAPI backend, never call Supabase directly.

To stand up a Supabase project from scratch: create a project at [supabase.com](https://supabase.com), open the SQL Editor, and run the contents of `supabase_schema.sql`. Then put that project's URL and **service_role** key (Dashboard -> Settings -> API) into your `.env` as `SUPABASE_SERVICE_ROLE_KEY`.

If the team is sharing a single Supabase project, ask whoever created it for the `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` instead of creating your own — this key is sensitive, so share it privately (not in Slack/GitHub), not by committing it anywhere.

## Project Structure

```
claude.md.md          Architecture rules, tech stack, and JSON data contracts
main.py                FastAPI app and API routes
local_bilingual_model.py Lazy loader for the local Qwen bilingual adapter
bilingual_clinical_conversation_questions.xlsx Training question dataset
training/               Dataset preparation, training scripts, adapters, and metrics
supabase_schema.sql     Database schema, storage bucket, and RLS policies
requirements.txt        Python dependencies
.env.example            Required environment variables (copy to .env)
```
