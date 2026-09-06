from fastapi.testclient import TestClient

from main import app, _local_clinical_summary, normalize_multilingual_voice_text


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_patient_login_demo_auth():
    response = client.post(
        "/auth/patient/login",
        json={"patient_id": "P-101", "abha_id": "ABHA-123"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["patient_id"] == "P-101"
    assert payload["status"] == "ok"


def test_voice_normalization_handles_hinglish():
    text, detected = normalize_multilingual_voice_text("mere pet me dard hai", "hi")
    assert "dard" in text.lower()
    assert detected in {"hi", "hi-IN", "Hindi", "Hinglish", "en-IN"}


def test_local_fallback_summary_is_valid():
    summary = _local_clinical_summary("mere pet me dard hai", [])
    assert summary.chief_complaint
    assert isinstance(summary.red_flags_detected, bool)


def test_document_upload_offline_fallback():
    response = client.post(
        "/upload-document",
        files={"file": ("report.jpg", b"not really image", "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["storage_path"].endswith(".jpg")
    assert "extracted_document" in payload
    assert isinstance(payload["extracted_document"]["diagnoses"], list)
