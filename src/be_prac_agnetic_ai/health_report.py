"""
Health report analyzer.
Takes a medical/lab report (PDF, image or text), and asks Gemini for:
  - a simple health summary
  - key findings from the report
  - an Indian diet plan based on those findings
"""

import base64
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from be_prac_agnetic_ai.llm import llm

# file types Gemini can read directly
SUPPORTED_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
    "text/plain",
}

DietPreference = Literal["vegetarian", "non-vegetarian", "eggetarian", "vegan", "jain"]
Region = Literal["any", "north", "south", "east", "west"]


# ---------- output schema (Gemini fills this in) ----------

class LabFinding(BaseModel):
    test_name: str = Field(description="Name of the test, e.g. HbA1c, Vitamin D, LDL Cholesterol")
    value: str = Field(description="Value with unit as written in the report")
    normal_range: str = Field(description="Reference range from the report, or empty if not given")
    status: Literal["low", "normal", "high", "borderline"]
    explanation: str = Field(description="One simple sentence on what this means")


class Meal(BaseModel):
    time: str = Field(description="Meal slot with time, e.g. 'Breakfast (8:00 AM)'")
    options: list[str] = Field(description="2-3 Indian food options for this slot, with portion size")


class HealthReport(BaseModel):
    is_medical_report: bool = Field(description="False if the file is not a medical/lab report")
    summary: str = Field(description="Overall health summary in simple, non-technical language")
    key_findings: list[LabFinding] = Field(description="Important values from the report, abnormal ones first")
    health_concerns: list[str] = Field(description="Main concerns based on the findings")
    diet_plan: list[Meal] = Field(description="Full-day Indian diet plan from early morning to bedtime")
    foods_to_include: list[str]
    foods_to_avoid: list[str]
    lifestyle_tips: list[str] = Field(description="Exercise, sleep, water intake and similar tips")
    see_doctor_for: list[str] = Field(description="Findings that need a doctor's attention")


SYSTEM_PROMPT = """You are a careful clinical nutritionist based in India.
You read medical and lab reports and explain them in simple language.

Rules:
- Only use values that are actually in the report. Do not invent numbers.
- Diet plan must use common Indian foods (dal, roti, rice, sabzi, curd, poha, idli, etc.)
  that match the user's diet preference and region.
- Respect the diet preference strictly (e.g. no egg for vegetarian, no onion/garlic/root vegetables for jain).
- Keep portions realistic and mention them (1 katori, 2 rotis, etc.).
- Do not diagnose diseases or prescribe medicines. Flag serious values under see_doctor_for.
- If the file is not a medical report, set is_medical_report to false and leave other lists empty.
"""

# structured output: Gemini returns a HealthReport object instead of plain text
analyzer = llm.with_structured_output(HealthReport)


def _file_block(file_bytes: bytes, mime_type: str) -> dict:
    """Turn the uploaded file into a message block Gemini understands."""
    if mime_type == "text/plain":
        return {"type": "text", "text": file_bytes.decode("utf-8", errors="ignore")}
    return {
        "type": "file",
        "source_type": "base64",
        "mime_type": mime_type,
        "data": base64.b64encode(file_bytes).decode("utf-8"),
    }


def analyze_report(
    file_bytes: bytes,
    mime_type: str,
    diet_preference: DietPreference = "vegetarian",
    region: Region = "any",
) -> HealthReport:
    """Analyze a health report and return a summary + Indian diet plan."""
    instructions = (
        f"Analyze this health report.\n"
        f"Diet preference: {diet_preference}\n"
        f"Region of India for food choices: {region}"
    )
    message = HumanMessage(content=[
        {"type": "text", "text": instructions},
        _file_block(file_bytes, mime_type),
    ])
    return analyzer.invoke([SystemMessage(content=SYSTEM_PROMPT), message])
