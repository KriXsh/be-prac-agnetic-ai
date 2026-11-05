"""
FastAPI app that exposes the LLM as an API.

Run:  poetry run uvicorn be_prac_agnetic_ai.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import mimetypes

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from be_prac_agnetic_ai.health_report import (
    SUPPORTED_TYPES,
    DietPreference,
    HealthReport,
    Region,
    analyze_report,
)
from be_prac_agnetic_ai.llm import ask

app = FastAPI(title="Agentic AI Practice")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

DISCLAIMER = (
    "This is AI-generated information, not medical advice. "
    "Please consult a doctor or registered dietitian before making changes."
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


class HealthReportResponse(HealthReport):
    disclaimer: str = DISCLAIMER


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    answer = ask(request.question)
    return AskResponse(answer=answer)


@app.post("/health-report", response_model=HealthReportResponse)
def health_report_endpoint(
    file: UploadFile = File(..., description="Medical/lab report: PDF, PNG, JPG, WEBP or TXT"),
    diet_preference: DietPreference = Form("vegetarian"),
    region: Region = Form("any"),
):
    # some clients send a generic type, so fall back to the file extension
    mime_type = file.content_type
    if mime_type not in SUPPORTED_TYPES:
        mime_type = mimetypes.guess_type(file.filename or "")[0]
    if mime_type not in SUPPORTED_TYPES:
        raise HTTPException(415, f"Unsupported file type. Use one of: {sorted(SUPPORTED_TYPES)}")

    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(400, "File is empty")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(413, "File is larger than 10 MB")

    try:
        report = analyze_report(file_bytes, mime_type, diet_preference, region)
    except Exception as e:
        # Gemini errors (overloaded, timeout, bad key) -> 502 instead of a bare 500
        raise HTTPException(502, f"LLM call failed: {e}")

    if not report.is_medical_report:
        raise HTTPException(422, "This file does not look like a medical or lab report")

    return HealthReportResponse(**report.model_dump())
