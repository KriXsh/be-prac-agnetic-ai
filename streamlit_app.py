"""
Streamlit UI for the health report API.

Run the API first:  poetry run uvicorn be_prac_agnetic_ai.main:app --reload
Then the UI:        poetry run streamlit run streamlit_app.py
"""

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

STATUS_ICON = {"low": "🔻", "high": "🔺", "borderline": "⚠️", "normal": "✅"}

st.set_page_config(page_title="Health Report & Diet Plan", page_icon="🥗", layout="wide")

st.title("🥗 Health Report Analyzer")
st.caption("Upload your lab report to get a simple health summary and an Indian diet plan.")

# ---------- inputs ----------
with st.sidebar:
    st.header("Your preferences")
    diet_preference = st.selectbox(
        "Diet preference",
        ["vegetarian", "non-vegetarian", "eggetarian", "vegan", "jain"],
        format_func=str.title,
    )
    region = st.selectbox(
        "Regional food style",
        ["any", "north", "south", "east", "west"],
        format_func=lambda r: "Any" if r == "any" else f"{r.title()} Indian",
    )
    st.divider()
    st.caption(f"API: `{API_URL}`")

uploaded = st.file_uploader(
    "Upload report (PDF, image or text)",
    type=["pdf", "png", "jpg", "jpeg", "webp", "txt"],
)

if st.button("Analyze report", type="primary", disabled=uploaded is None):
    with st.spinner("Reading your report... this can take up to a minute"):
        try:
            response = httpx.post(
                f"{API_URL}/health-report",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                data={"diet_preference": diet_preference, "region": region},
                timeout=180,
            )
        except httpx.ConnectError:
            st.error(f"Cannot reach the API at {API_URL}. Is uvicorn running?")
            st.stop()
        except httpx.TimeoutException:
            st.error("The API took too long to respond. Please try again.")
            st.stop()

    if response.status_code != 200:
        st.error(f"Error {response.status_code}: {response.json().get('detail', response.text)}")
        st.stop()

    # keep result so it survives Streamlit reruns
    st.session_state["report"] = response.json()

report = st.session_state.get("report")
if not report:
    st.stop()

# ---------- results ----------
st.subheader("🩺 Health summary")
st.info(report["summary"])

if report["key_findings"]:
    st.subheader("🔬 Key findings")
    st.dataframe(
        [
            {
                "": STATUS_ICON.get(f["status"], ""),
                "Test": f["test_name"],
                "Value": f["value"],
                "Normal range": f["normal_range"],
                "Status": f["status"].title(),
                "What it means": f["explanation"],
            }
            for f in report["key_findings"]
        ],
        hide_index=True,
        width="stretch",
    )

if report["health_concerns"]:
    st.subheader("⚠️ Health concerns")
    for concern in report["health_concerns"]:
        st.markdown(f"- {concern}")

st.subheader("🍛 Your Indian diet plan")
cols = st.columns(2)
for i, meal in enumerate(report["diet_plan"]):
    with cols[i % 2].container(border=True):
        st.markdown(f"**{meal['time']}**")
        for option in meal["options"]:
            st.markdown(f"- {option}")

include_col, avoid_col = st.columns(2)
with include_col:
    st.subheader("✅ Foods to include")
    for food in report["foods_to_include"]:
        st.markdown(f"- {food}")
with avoid_col:
    st.subheader("❌ Foods to avoid")
    for food in report["foods_to_avoid"]:
        st.markdown(f"- {food}")

if report["lifestyle_tips"]:
    st.subheader("🏃 Lifestyle tips")
    for tip in report["lifestyle_tips"]:
        st.markdown(f"- {tip}")

if report["see_doctor_for"]:
    st.subheader("👨‍⚕️ Talk to your doctor about")
    for item in report["see_doctor_for"]:
        st.warning(item)

st.divider()
st.caption(report["disclaimer"])
