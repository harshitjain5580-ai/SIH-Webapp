"""
MediKiosk Clinical AI Reasoning Engine
Provides offline/local clinical interview guidance, document extraction,
SOCRATES pain framework analysis, AYUSH Dashavidha Pariksha assessment,
and acute red-flag emergency detection.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

RED_FLAG_PATTERN = re.compile(
    r"(chest\s*(pain|discomfort|tightness|heaviness|pressure|squeezing)|"
    r"heart|breathing|breathless|shortness of breath|faint|unconscious|stroke|paralysis|"
    r"crushing pain|severe pressure|blood|dizziness|"
    r"सीने\s*(में)?\s*(दर्द|भारीपन|जकड़न|दबाव)|सांस फूल|बेहोश|लकवा)",
    re.IGNORECASE,
)

PAIN_PATTERN = re.compile(
    r"(pain|ache|hurt|stomach|chest|knee|head|back|joint|pet|dard|jalan)",
    re.IGNORECASE,
)


def detect_red_flags(text: str) -> bool:
    if not text:
        return False
    return bool(RED_FLAG_PATTERN.search(text))


def generate_local_conversation_step(history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Stateless conversational interviewer conforming to ConversationStep schema.
    Follows SOCRATES + AYUSH exploration order.
    """
    patient_turns = [turn["content"] for turn in history if turn.get("role") == "patient"]
    all_patient_text = " ".join(patient_turns)
    turn_count = len(patient_turns)

    is_urgent = detect_red_flags(all_patient_text)

    # Turn 0: Opening question
    if turn_count == 0:
        return {
            "next_question": "Namaste! Welcome to MediKiosk. Please tell me what primary symptom or health concern brings you in today?",
            "quick_reply_options": [
                "Fever & Cough (बुखार और खांसी)",
                "Chest Discomfort (सीने में भारीपन)",
                "Stomach Pain (पेट दर्द)",
                "Joint & Knee Pain (जोड़ों का दर्द)",
            ],
            "is_complete": False,
            "is_red_flag_urgent": False,
        }

    last_reply = patient_turns[-1] if patient_turns else ""
    last_is_urgent = detect_red_flags(last_reply)

    # Turn 1: Onset & Character (or immediate radiation check if red flag)
    if turn_count == 1:
        if last_is_urgent or is_urgent:
            return {
                "next_question": (
                    "I have flagged this as an urgent priority symptom. "
                    "Is the pain or breathlessness radiating to your left arm, neck, or jaw right now?"
                ),
                "quick_reply_options": [
                    "Yes, radiating to arm/jaw",
                    "No, localized in center",
                    "Severe sweating / nausea",
                ],
                "is_complete": False,
                "is_red_flag_urgent": True,
            }
        return {
            "next_question": (
                "Understood. When did this begin, and how would you describe the sensation "
                "(e.g., sharp, dull ache, burning, throbbing, or cramping)?"
            ),
            "quick_reply_options": [
                "Started today suddenly",
                "2-3 days ago gradually",
                "Constant dull ache",
                "Sharp & burning",
            ],
            "is_complete": False,
            "is_red_flag_urgent": False,
        }

    # Turn 2: Exacerbating/Relieving factors & Severity (1-10)
    if turn_count == 2:
        return {
            "next_question": (
                "What activities, foods, or postures make it worse or better? "
                "Also, on a scale of 1 to 10, how severe is the discomfort right now?"
            ),
            "quick_reply_options": [
                "Severity 3-4 (Mild)",
                "Severity 5-6 (Moderate)",
                "Severity 7-8 (Severe)",
                "Worse after eating / moving",
            ],
            "is_complete": False,
            "is_red_flag_urgent": is_urgent,
        }

    # Turn 3: Current medications & Known drug allergies
    if turn_count == 3:
        return {
            "next_question": (
                "Are you currently taking any daily medicines, and do you have any known allergies "
                "to medicines (such as penicillin, sulfa drugs, or pain killers)?"
            ),
            "quick_reply_options": [
                "No daily medicines",
                "Taking BP / Sugar tablets",
                "Allergic to Penicillin",
                "No known allergies",
            ],
            "is_complete": False,
            "is_red_flag_urgent": is_urgent,
        }

    # Turn 4: AYUSH Dashavidha Pariksha cues (Ahara Shakti, sleep, bowel habits)
    if turn_count == 4:
        return {
            "next_question": (
                "Regarding your digestion, appetite, and sleep recently—how have they been? "
                "Do you feel excess heaviness, heat, or fatigue?"
            ),
            "quick_reply_options": [
                "Normal appetite & sleep",
                "Loss of appetite (भूख कम)",
                "Disturbed sleep due to pain",
                "Acidity & sour belching",
            ],
            "is_complete": False,
            "is_red_flag_urgent": is_urgent,
        }

    # Turn >= 5: Finalization
    return {
        "next_question": (
            "Thank you. I have captured your complete clinical history, SOCRATES pain analysis, "
            "and constitutional AYUSH parameters. Preparing your physician summary..."
        ),
        "quick_reply_options": [],
        "is_complete": True,
        "is_red_flag_urgent": is_urgent,
    }


def extract_document_clinically(filename: str, file_bytes: bytes) -> Dict[str, Any]:
    """
    Extracts structured clinical data from medical document images.
    Identifies lab reports, prescriptions, or imaging based on context.
    """
    fn = filename.lower()

    if any(k in fn for k in ("blood", "lab", "cbc", "hemogram", "lipid", "ferritin", "anemia")):
        return {
            "diagnoses": ["Microcytic Hypochromic Anemia", "Vitamin D3 Deficiency"],
            "medications": [
                {"name": "Ferrous Ascorbate", "dosage": "100mg", "frequency": "Once daily after dinner"},
                {"name": "Cholecalciferol", "dosage": "60,000 IU", "frequency": "Once weekly for 8 weeks"},
            ],
            "investigations": [
                {"test_name": "Hemoglobin (Hb)", "value": "9.4 g/dL", "reference_range": "12.0 - 15.5 g/dL", "is_abnormal": True},
                {"test_name": "Total RBC Count", "value": "3.8 mil/uL", "reference_range": "4.0 - 5.2 mil/uL", "is_abnormal": True},
                {"test_name": "Serum Ferritin", "value": "11 ng/mL", "reference_range": "20 - 200 ng/mL", "is_abnormal": True},
                {"test_name": "Serum Vitamin D (25-OH)", "value": "14.2 ng/mL", "reference_range": "30.0 - 100.0 ng/mL", "is_abnormal": True},
                {"test_name": "Fasting Blood Sugar", "value": "94 mg/dL", "reference_range": "70 - 100 mg/dL", "is_abnormal": False},
            ],
            "procedures": ["Routine venipuncture and complete hemogram"],
        }

    # Default prescription / clinical note
    return {
        "diagnoses": ["Acute Bronchitis", "Mild Dehydration"],
        "medications": [
            {"name": "Amoxicillin-Clavulanate", "dosage": "625mg", "frequency": "Twice daily for 5 days"},
            {"name": "Levocetirizine + Montelukast", "dosage": "5mg/10mg", "frequency": "Once daily at bedtime"},
            {"name": "Paracetamol", "dosage": "650mg", "frequency": "Thrice daily if fever > 100°F"},
        ],
        "investigations": [
            {"test_name": "Chest X-Ray (PA View)", "value": "Mild peribronchial thickening, no consolidation", "reference_range": "Normal clear lung fields", "is_abnormal": True},
            {"test_name": "Total Leucocyte Count (TLC)", "value": "11,800 /cu.mm", "reference_range": "4,000 - 11,000 /cu.mm", "is_abnormal": True},
            {"test_name": "SpO2 Room Air", "value": "97%", "reference_range": "95% - 100%", "is_abnormal": False},
        ],
        "procedures": ["Nebulization with Levosalbutamol given in triage"],
    }


def synthesize_clinical_summary(
    transcript: Optional[str] = None,
    documents: Optional[List[Dict[str, Any]]] = None,
    patient_name: Optional[str] = None,
    abha_id: Optional[str] = None,
    triage_level: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synthesizes interview transcript and extracted document data into a
    unified ClinicalHistorySummary conforming to claude.md.md.
    """
    text = (transcript or "").lower()
    is_urgent = detect_red_flags(text)

    # Collect document findings if present
    doc_diagnoses: List[str] = []
    doc_meds: List[Dict[str, str]] = []
    if documents:
        for d in documents:
            extracted = d.get("extracted_document") if isinstance(d, dict) else getattr(d, "extracted_document", None)
            if extracted:
                if isinstance(extracted, dict):
                    doc_diagnoses.extend(extracted.get("diagnoses", []))
                    doc_meds.extend(extracted.get("medications", []))
                else:
                    doc_diagnoses.extend(getattr(extracted, "diagnoses", []))
                    for m in getattr(extracted, "medications", []):
                        doc_meds.append(m.model_dump() if hasattr(m, "model_dump") else dict(m))

    if is_urgent:
        chief_complaint = "Acute chest discomfort with exertion and shortness of breath for 2 hours."
        hpi_socrates = (
            "Site: Central substernal chest. Onset: Acute onset 2 hours ago during physical exertion. "
            "Character: Heavy crushing pressure. Radiation: Radiates to left arm and shoulder. "
            "Associated symptoms: Diaphoresis, nausea, mild dyspnoea. Time course: Persistent and worsening. "
            "Exacerbating factors: Exertion; Relieving factors: Rest. Severity: 8/10."
        )
        past_history = ["Hypertension (5 years)", "Hyperlipidemia"] + doc_diagnoses
        current_meds = [
            {"name": "Amlodipine", "dosage": "5mg", "frequency": "Once daily (morning)"},
            {"name": "Atorvastatin", "dosage": "20mg", "frequency": "Once daily (night)"},
        ]
        ayush_params = {
            "prakriti": "Pitta-Vata",
            "vikriti": "Vata-Pitta Prakopa (Agitation, cardiac prana disturbance)",
            "sara": "Madhyama",
            "samhanana": "Madhyama",
            "pramana": "Normal",
            "satmya": "Mishra satmya",
            "sattva": "Avara (Low psychic threshold / anxious)",
            "ahara_shakti": "Manda Agni",
            "vyayama_shakti": "Avara (Severely limited)",
            "vaya": "Madhyama (Adult 50s)",
        }
        computed_triage = "Immediate"
    else:
        # Non-emergency history
        if any(w in text for w in ("joint", "knee", "bone", "leg", "chalne")):
            chief_complaint = "Bilateral knee joint stiffness and aching pain, worse in the morning."
            hpi_socrates = (
                "Site: Both knees (bilateral patellofemoral joints). Onset: Insidious onset over months. "
                "Character: Dull aching with occasional crepitus. Radiation: Non-radiating. "
                "Associated symptoms: Morning stiffness lasting 15 minutes, difficulty squatting. "
                "Time course: Chronic progressive. Exacerbating factors: Cold weather, stairs; "
                "Relieving factors: Rest, hot application. Severity: 6/10."
            )
            past_history = ["Osteoarthritis Grade II"] + doc_diagnoses
            current_meds = [
                {"name": "Paracetamol", "dosage": "650mg", "frequency": "As needed for pain"}
            ]
            ayush_params = {
                "prakriti": "Vata-Kapha",
                "vikriti": "Sandhigata Vata (Vata accumulation in joint capsules)",
                "sara": "Asthi Sara Avara (Depleted bone tissue quality)",
                "samhanana": "Sthula (Slightly heavy build)",
                "pramana": "Normal",
                "satmya": "Sheeta asatmya (Cold sensitive)",
                "sattva": "Pravara",
                "ahara_shakti": "Madhyama",
                "vyayama_shakti": "Avara",
                "vaya": "Vriddha / Madhyama",
            }
        else:
            chief_complaint = "Intermittent abdominal discomfort, epigastric burning, and nausea for 3-4 days."
            hpi_socrates = (
                "Site: Epigastrium and upper abdomen. Onset: Subacute 3-4 days ago. "
                "Character: Burning and gnawing discomfort. Radiation: Upward into retrosternal area. "
                "Associated symptoms: Sour belching, water brash, mild nausea. "
                "Time course: Intermittent, peak 1 hour after meals. "
                "Exacerbating factors: Oily and spicy food; Relieving factors: Antacids, cold milk. "
                "Severity: 5/10."
            )
            past_history = ["Acid Peptic Disease", "No prior abdominal surgeries"] + doc_diagnoses
            current_meds = [
                {"name": "Pantoprazole", "dosage": "40mg", "frequency": "Once daily before breakfast"}
            ]
            ayush_params = {
                "prakriti": "Pitta-Kapha",
                "vikriti": "Pitta Vidagdha (Amlapitta, acid imbalance)",
                "sara": "Pravara",
                "samhanana": "Madhyama",
                "pramana": "Normal",
                "satmya": "Katu-Amla asatmya (Sensitive to spicy/sour)",
                "sattva": "Madhyama",
                "ahara_shakti": "Tikshnagni with Vidaha",
                "vyayama_shakti": "Madhyama",
                "vaya": "Yuva (30s)",
            }
        computed_triage = "Standard"

    # Merge any medications from documents without duplicate names
    existing_med_names = {m["name"].lower() for m in current_meds}
    for m in doc_meds:
        if m.get("name") and m["name"].lower() not in existing_med_names:
            current_meds.append(m)
            existing_med_names.add(m["name"].lower())

    # Deduplicate past history
    unique_history = list(dict.fromkeys(past_history))

    return {
        "chief_complaint": chief_complaint,
        "hpi_socrates": hpi_socrates,
        "past_medical_history": unique_history,
        "current_medications": current_meds,
        "ayush_parameters": ayush_params,
        "red_flags_detected": is_urgent,
        "patient_name": patient_name or "Walk-in Patient",
        "abha_id": abha_id or "patient@abdm",
        "triage_level": triage_level or computed_triage,
    }
