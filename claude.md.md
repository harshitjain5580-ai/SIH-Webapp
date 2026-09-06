# MediKiosk: AI Clinical History Software Platform

## 1. Tech Stack & Architecture
- **Backend:** Python 3.10+, FastAPI, Pydantic (v2).
- **Frontend:** React/Next.js, TypeScript, Tailwind CSS.
- **AI/ML Engine:** OpenAI API.
- **Voice/OCR:** Web Speech API (Frontend native), Multimodal LLM (for Document OCR).
- **Database & Storage:** Supabase (PostgreSQL & Storage).

## 2. Strict Development Rules
### Python Backend (FastAPI)
- Use standard 4-space indentation for all Python code.
- Always use `pydantic` `BaseModel` with explicit `Field(description="...")` for AI data contracts.
- NEVER parse LLM string outputs manually. Strictly use OpenAI's native Structured Outputs (`client.beta.chat.completions.parse`) to guarantee JSON enforcement.
- Use `async def` for all routing endpoints. 
- Implement mock endpoints for ABDM/Hospital integrations. Do not build real database connections during the hackathon.

### TypeScript Frontend (React)
- Use strict TypeScript. The use of `any` is forbidden.
- **Dual-Mode UI:** Every interaction must support both a large touchscreen button and a Web Speech API microphone input.
- **Accessibility:** Use high-contrast Tailwind classes and large font bases (`text-lg` or higher) for elderly/low-literacy users.

## 3. Core Medical Domain Logic
Claude MUST adhere to these clinical frameworks when generating dialogue or summarizing:
- **SOCRATES Framework:** When a patient reports pain, the LLM must attempt to extract: Site, Onset, Character, Radiation, Associated symptoms, Time course, Exacerbating/relieving factors, and Severity.
- **AYUSH Dashavidha Pariksha:** Ensure the conversational engine includes logic to capture Ayurvedic parameters (Prakriti, Vikriti, Agni, Ahara-Vihara).
- **Emergency Red Flags:** If the transcript contains markers for acute cardiac events (chest pain, dyspnoea) or neurological deficits (stroke symptoms), the system MUST instantly set `red_flags_detected = True`.

## 4. The Primary JSON Data Contracts
All backend-to-frontend communication regarding the clinical history must conform to this schema:

```json
{
  "chief_complaint": "string (1-2 sentences)",
  "hpi_socrates": "string (detailed narrative)",
  "past_medical_history": ["string"],
  "current_medications": [
    {
      "name": "string",
      "dosage": "string",
      "frequency": "string"
    }
  ],
  "ayush_parameters": {
    "prakriti": "string",
    "vikriti": "string"
  },
  "red_flags_detected": "boolean"
}
```

## 5. Database Rules
- Use the official `supabase-py` Python client for all database operations.
- Store all extracted clinical histories in a `patient_histories` table.
- Upload all prescription images to a Supabase Storage bucket called `medical_documents` before sending the public URL to the OpenAI Vision model.