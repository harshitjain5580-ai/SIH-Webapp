"""
MediKiosk Local Persistent Data Store
Thread-safe fallback and demo database for patient clinical histories,
profiles, reports, and approved training cases.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger("medikiosk.store")

DB_DIR = Path(__file__).parent / "data"
DB_FILE = DB_DIR / "local_db.json"
_lock = Lock()


def _get_default_records() -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    return [
        {
            "id": "d9b23f81-54c7-4b68-963a-23910c22fa10",
            "created_at": (now - timedelta(minutes=12)).isoformat(),
            "patient_name": "Rajesh Kumar",
            "abha_id": "rajesh.kumar@abdm",
            "triage_level": "Immediate",
            "chief_complaint": "Severe chest discomfort and shortness of breath for 2 hours.",
            "hpi_socrates": (
                "Site: Mid-sternal chest. Onset: Sudden while climbing stairs 2 hours ago. "
                "Character: Crushing, heavy pressure. Radiation: Radiating to left shoulder and jaw. "
                "Associated: Diaphoresis, nausea, mild dyspnoea. Time course: Constant and worsening. "
                "Exacerbating: Exertion; Relieving: None. Severity: 8/10."
            ),
            "past_medical_history": ["Hypertension (5 years)", "Hyperlipidemia"],
            "current_medications": [
                {"name": "Amlodipine", "dosage": "5mg", "frequency": "Once daily (morning)"},
                {"name": "Atorvastatin", "dosage": "20mg", "frequency": "Once daily (night)"},
            ],
            "ayush_parameters": {
                "prakriti": "Pitta-Vata",
                "vikriti": "Pitta-Vata Vriddhi (Agitation, heat, chest congestion)",
                "sara": "Madhyama (Moderate)",
                "samhanana": "Madhyama (Medium build)",
                "pramana": "Pramana yukta (Well proportioned)",
                "satmya": "Mishra satmya",
                "sattva": "Avara (Anxious / Low psychic threshold)",
                "ahara_shakti": "Manda Agni (Sluggish appetite)",
                "vyayama_shakti": "Avara (Low stamina)",
                "vaya": "Madhyama (Adult 52 yrs)",
            },
            "red_flags_detected": True,
            "alert_acknowledged": False,
        },
        {
            "id": "a47c19e3-82a1-4321-bf90-f9231849a901",
            "created_at": (now - timedelta(minutes=35)).isoformat(),
            "patient_name": "Sunita Sharma",
            "abha_id": "sunita.sharma@abdm",
            "triage_level": "Standard",
            "chief_complaint": "Burning epigastric pain and sour belching after oily meals.",
            "hpi_socrates": (
                "Site: Epigastric region. Onset: Gradual over 3 weeks. Character: Burning, gnawing ache. "
                "Radiation: Upward into retrosternal area. Associated: Heartburn, water brash, mild nausea. "
                "Time course: Intermittent, worse 1 hour post meals. Exacerbating: Spicy/fried food; "
                "Relieving: Cold milk, antacid. Severity: 5/10."
            ),
            "past_medical_history": ["Acid peptic disease", "No surgical history"],
            "current_medications": [
                {"name": "Pantoprazole", "dosage": "40mg", "frequency": "Before breakfast"}
            ],
            "ayush_parameters": {
                "prakriti": "Pitta-Kapha",
                "vikriti": "Pitta Vidagdha (Amlapitta, excessive acid secretion)",
                "sara": "Pravara (Good muscle/tissue quality)",
                "samhanana": "Samhanana yukta (Compact build)",
                "pramana": "Normal",
                "satmya": "Katu-Amla asatmya (Sensitive to spicy/sour)",
                "sattva": "Madhyama (Moderate resilience)",
                "ahara_shakti": "Tikshnagni with Vidaha (Sharp appetite with burning)",
                "vyayama_shakti": "Madhyama (Moderate)",
                "vaya": "Yuva (34 yrs)",
            },
            "red_flags_detected": False,
            "alert_acknowledged": False,
        },
        {
            "id": "f1e8432a-1122-4455-8899-aabbccddeeff",
            "created_at": (now - timedelta(minutes=75)).isoformat(),
            "patient_name": "Harish Chandra",
            "abha_id": "harish.c@abdm",
            "triage_level": "Standard",
            "chief_complaint": "Bilateral knee stiffness and joint pain, worse in morning.",
            "hpi_socrates": (
                "Site: Both knees (Right > Left). Onset: Insidious onset 6 months ago. "
                "Character: Dull aching, crepitus on flexion. Radiation: Localized to patellofemoral joint. "
                "Associated: Early morning stiffness lasting 15 mins, difficulty squatting. "
                "Time course: Progressive. Exacerbating: Walking downstairs, cold weather; "
                "Relieving: Hot fermentation, rest. Severity: 6/10."
            ),
            "past_medical_history": ["Osteoarthritis grade II", "Type 2 Diabetes Mellitus (well controlled)"],
            "current_medications": [
                {"name": "Metformin", "dosage": "500mg", "frequency": "Twice daily after meals"},
                {"name": "Paracetamol", "dosage": "650mg", "frequency": "As needed for joint pain"},
            ],
            "ayush_parameters": {
                "prakriti": "Vata-Kapha",
                "vikriti": "Sandhigata Vata (Vata accumulation in joint spaces)",
                "sara": "Asthi Sara Avara (Depleted bone/joint tissue density)",
                "samhanana": "Sthula (Heavy / mild overweight)",
                "pramana": "Normal",
                "satmya": "Sheeta asatmya (Cold intolerant)",
                "sattva": "Pravara (Patient, calm demeanor)",
                "ahara_shakti": "Madhyama",
                "vyayama_shakti": "Avara (Limited due to joint stiffness)",
                "vaya": "Vriddha (Senior 64 yrs)",
            },
            "red_flags_detected": False,
            "alert_acknowledged": False,
        },
    ]


class LocalDataStore:
    def __init__(self):
        self._data: Dict[str, Any] = {
            "patient_histories": [],
            "patient_profiles": {},
            "patient_reports": [],
            "doctor_approved_cases": [],
        }
        self._load()

    def _load(self):
        with _lock:
            if DB_FILE.exists():
                try:
                    loaded = json.loads(DB_FILE.read_text(encoding="utf-8"))
                    self._data.update(loaded)
                    if not self._data.get("patient_histories"):
                        self._data["patient_histories"] = _get_default_records()
                        self._save_unlocked()
                except Exception as exc:
                    logger.warning("Error reading local db file: %s. Reinitializing defaults.", exc)
                    self._data["patient_histories"] = _get_default_records()
                    self._save_unlocked()
            else:
                self._data["patient_histories"] = _get_default_records()
                self._save_unlocked()

    def _save_unlocked(self):
        try:
            DB_DIR.mkdir(parents=True, exist_ok=True)
            DB_FILE.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to save local db file: %s", exc)

    def insert_history(self, record: Dict[str, Any]) -> Dict[str, Any]:
        with _lock:
            if "id" not in record or not record["id"]:
                record["id"] = str(uuid.uuid4())
            if "created_at" not in record or not record["created_at"]:
                record["created_at"] = datetime.now(timezone.utc).isoformat()
            if "alert_acknowledged" not in record:
                record["alert_acknowledged"] = False
            self._data["patient_histories"].insert(0, record)
            self._save_unlocked()
            return record

    def get_histories(self, limit: int = 10) -> List[Dict[str, Any]]:
        with _lock:
            histories = list(self._data.get("patient_histories", []))
            histories.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return histories[:limit]

    def get_priority_alerts(self) -> List[Dict[str, Any]]:
        with _lock:
            histories = list(self._data.get("patient_histories", []))
            alerts = [
                h for h in histories
                if h.get("red_flags_detected") and not h.get("alert_acknowledged")
            ]
            alerts.sort(key=lambda x: x.get("created_at", ""))
            return alerts

    def get_history(self, history_id: str) -> Optional[Dict[str, Any]]:
        with _lock:
            for h in self._data.get("patient_histories", []):
                if h.get("id") == history_id:
                    return dict(h)
            return None

    def update_history(self, history_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with _lock:
            for h in self._data.get("patient_histories", []):
                if h.get("id") == history_id:
                    h.update(patch)
                    self._save_unlocked()
                    return dict(h)
            return None

    def acknowledge_alert(self, history_id: str) -> Optional[Dict[str, Any]]:
        return self.update_history(history_id, {"alert_acknowledged": True})

    def save_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        with _lock:
            pid = profile.get("patient_id")
            if pid:
                self._data["patient_profiles"][pid] = profile
                self._save_unlocked()
            return profile

    def get_profile(self, patient_id: str) -> Optional[Dict[str, Any]]:
        with _lock:
            return self._data["patient_profiles"].get(patient_id)

    def save_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        with _lock:
            if "report_id" not in report:
                report["report_id"] = str(uuid.uuid4())
            self._data["patient_reports"].append(report)
            self._save_unlocked()
            return report

    def get_reports(self, patient_id: str) -> List[Dict[str, Any]]:
        with _lock:
            return [
                r for r in self._data.get("patient_reports", [])
                if r.get("patient_id") == patient_id
            ]

    def save_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        with _lock:
            if "case_id" not in case:
                case["case_id"] = str(uuid.uuid4())
            self._data["doctor_approved_cases"].append(case)
            self._save_unlocked()
            return case


# Singleton store instance
store = LocalDataStore()
