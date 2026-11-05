"""
API tests. The real LLM call is replaced with a fake one,
so these run without an API key or internet.
"""

from fastapi.testclient import TestClient

from be_prac_agnetic_ai import main

client = TestClient(main.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask(monkeypatch):
    # swap the real ask() with a fake that just echoes the question
    monkeypatch.setattr(main, "ask", lambda question: f"echo: {question}")

    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 200
    assert response.json() == {"answer": "echo: hi"}


# ---------- /health-report ----------

from be_prac_agnetic_ai.health_report import HealthReport

FAKE_REPORT = HealthReport(
    is_medical_report=True,
    summary="Vitamin D is low, rest is normal.",
    key_findings=[],
    health_concerns=["Low Vitamin D"],
    diet_plan=[{"time": "Breakfast (8:00 AM)", "options": ["2 moong dal chilla"]}],
    foods_to_include=["Mushrooms"],
    foods_to_avoid=["Fried snacks"],
    lifestyle_tips=["15 min morning sunlight"],
    see_doctor_for=[],
)


def test_health_report(monkeypatch):
    calls = {}

    def fake_analyze(file_bytes, mime_type, diet_preference, region):
        calls.update(mime_type=mime_type, diet_preference=diet_preference, region=region)
        return FAKE_REPORT

    monkeypatch.setattr(main, "analyze_report", fake_analyze)

    response = client.post(
        "/health-report",
        files={"file": ("report.pdf", b"%PDF-fake", "application/pdf")},
        data={"diet_preference": "jain", "region": "west"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == FAKE_REPORT.summary
    assert "disclaimer" in body
    assert calls == {"mime_type": "application/pdf", "diet_preference": "jain", "region": "west"}


def test_health_report_guesses_type_from_extension(monkeypatch):
    monkeypatch.setattr(main, "analyze_report", lambda *args: FAKE_REPORT)
    response = client.post(
        "/health-report",
        files={"file": ("report.pdf", b"%PDF-fake", "application/octet-stream")},
    )
    assert response.status_code == 200


def test_health_report_rejects_unsupported_type():
    response = client.post("/health-report", files={"file": ("report.docx", b"x", "application/msword")})
    assert response.status_code == 415


def test_health_report_rejects_empty_file():
    response = client.post("/health-report", files={"file": ("report.txt", b"", "text/plain")})
    assert response.status_code == 400


def test_health_report_not_medical(monkeypatch):
    not_medical = FAKE_REPORT.model_copy(update={"is_medical_report": False})
    monkeypatch.setattr(main, "analyze_report", lambda *args: not_medical)
    response = client.post("/health-report", files={"file": ("notes.txt", b"shopping list", "text/plain")})
    assert response.status_code == 422
