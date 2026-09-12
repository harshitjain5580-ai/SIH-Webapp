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


def test_extract_history_offline_fallback_returns_record():
    response = client.post(
        "/extract-history",
        json={"transcript": "mere pet me dard hai"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"]
    assert payload["chief_complaint"]
    assert payload["alert_acknowledged"] is False


def test_conversation_fallback_adapts_to_patient_symptom():
    response = client.post(
        "/converse",
        json={"history": [{"role": "patient", "content": "mere pet me dard hai"}]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "dard" in payload["next_question"].lower()
    assert payload["is_complete"] is False


def test_gender_metadata_does_not_replace_first_symptom_question():
    response = client.post(
        "/converse",
        json={"history": [{"role": "patient", "content": "Patient gender selected: Male"}]},
    )
    assert response.status_code == 200
    assert "main symptom" in response.json()["next_question"].lower()


def test_conversation_fallback_uses_previous_answer_to_choose_next_detail():
    response = client.post(
        "/converse",
        json={
            "history": [
                {"role": "patient", "content": "mere pet me dard hai"},
                {"role": "assistant", "content": "Dard kis jagah hai?"},
                {"role": "patient", "content": "pet me jalne wala dard hai"},
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "kab se" in payload["next_question"].lower() or "when did" in payload["next_question"].lower()
    assert "kis jagah" not in payload["next_question"].lower()


def test_conversation_fallback_asks_respiratory_specific_question():
    response = client.post(
        "/converse",
        json={
            "history": [
                {"role": "patient", "content": "I have cough and fever for two days"},
                {"role": "assistant", "content": "How long have you had the cough, cold, or fever?"},
                {"role": "patient", "content": "Two days"},
            ]
        },
    )
    assert response.status_code == 200
    assert "mucus" in response.json()["next_question"].lower()


def test_conversation_fallback_asks_urinary_red_flag_question():
    response = client.post(
        "/converse",
        json={
            "history": [
                {"role": "patient", "content": "It burns when I pass urine and I go often"},
                {"role": "assistant", "content": "Do you have burning while passing urine, frequent urination, or urgency?"},
                {"role": "patient", "content": "Burning and frequent"},
            ]
        },
    )
    assert response.status_code == 200
    question = response.json()["next_question"].lower()
    assert "fever" in question or "back" in question or "blood" in question
