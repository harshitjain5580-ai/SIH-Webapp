# MediKiosk

AI Clinical History Software Platform — backend for an AI-assisted clinical history intake kiosk.

Full architecture rules, tech stack, and the clinical data contracts live in [`claude.md.md`](claude.md.md) — read that first.

## Tech Stack
- **Backend:** Python 3.10+, FastAPI, Pydantic v2
- **Frontend:** React/Next.js, TypeScript, Tailwind CSS
- **AI/ML:** OpenAI API (Structured Outputs + Vision)
- **Database & Storage:** Supabase (PostgreSQL & Storage)

## Getting Started (Backend)

1. **Clone and enter the project**
   ```bash
   git clone <this-repo-url>
   cd Webapp
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Fill in `.env` with real values:
   - `OPENAI_API_KEY` — from the OpenAI dashboard
   - `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` — from the shared Supabase project (ask a teammate for access, or see below). The backend requires the **service_role** key, not the anon key — patient data is locked down to service_role-only access (see `supabase_schema.sql`). Never share this key outside the team or commit it.

5. **Run the API**
   ```bash
   uvicorn main:app --reload
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

## For the Frontend Team

The backend exposes two endpoints, both returning the `ClinicalHistorySummary` JSON contract defined in `claude.md.md` §4 and implemented in `main.py`:

- `POST /extract-history` — body `{"transcript": "..."}`, extracts structured clinical history from a text transcript.
- `POST /ask-clinical-question` — body `{"transcript": "..."}`, asks the next question in Hindi, English, or Hinglish without diagnosing or prescribing medicine.
- `POST /extract-from-image` — multipart file upload, extracts structured clinical history from a prescription/document image (uploaded to Supabase Storage first, then read by the OpenAI Vision model).
- `GET /health` — liveness check.

The full response schema (with field descriptions) is browsable live at `/docs` once the server is running, or read directly off the Pydantic models at the top of `main.py`.

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
main.py                FastAPI app: Pydantic models + /extract-history + /extract-from-image
supabase_schema.sql     Database schema, storage bucket, and RLS policies
requirements.txt        Python dependencies
.env.example            Required environment variables (copy to .env)
```
