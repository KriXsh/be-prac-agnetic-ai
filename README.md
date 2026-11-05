# Agentic AI Practice

Hands-on code for an **Agentic AI series**, going from a single LLM call up to RAG, tool-using agents and multi-agent systems.
Each topic is learned in a Jupyter notebook first, then turned into reusable Python functions, a FastAPI endpoint and, where it helps, a Streamlit UI.

**Stack:** Python 3.12 · Poetry · LangChain · Google Gemini · FastAPI · Streamlit · pytest

---

## Series roadmap

| # | Topic | What you build | Status |
|---|---|---|---|
| 1 | Calling an LLM | `ask()` function + `POST /ask` | ✅ Done |
| 2 | Multimodal input + structured output | Health report analyzer (PDF/image → summary + Indian diet plan) + Streamlit app | ✅ Done |
| 3 | Prompt templates & chains | Reusable prompts, LCEL chains | ⏳ Planned |
| 4 | Embeddings & vector stores | Turn documents into searchable vectors | ⏳ Planned |
| 5 | RAG (Retrieval-Augmented Generation) | Chat with your own documents | ⏳ Planned |
| 6 | Tools / function calling | LLM that calls Python functions and APIs | ⏳ Planned |
| 7 | Agents | Agent that decides which tools to use, in a loop | ⏳ Planned |
| 8 | Memory | Chat history and long-term memory | ⏳ Planned |
| 9 | Multi-agent systems / LangGraph | Several agents working together on one task | ⏳ Planned |
| 10 | Evaluation & deployment | Testing LLM output, shipping the app | ⏳ Planned |

Update the status column as each part is finished.

---

## Project structure

```
be-prac-agnetic-ai/
├── README.md
├── pyproject.toml              # project config + dependencies (Poetry)
├── poetry.lock                 # exact installed versions, do not edit by hand
├── .env                        # API keys (not committed)
├── .gitignore
│
├── src/be_prac_agnetic_ai/     # all reusable application code
│   ├── llm.py                  # shared Gemini client + ask()
│   ├── health_report.py        # report analyzer: prompt, output schema, analyze_report()
│   └── main.py                 # FastAPI app: all API endpoints
│
├── streamlit_app.py            # Streamlit UI, calls the FastAPI endpoints
├── callLLM.ipynb               # notebook for experimenting
├── samples/
│   └── sample_blood_report.txt # fictional lab report for testing
└── tests/
    └── test_main.py            # API tests (LLM is faked, no key needed)
```

### How the code fits together

```
Streamlit UI  ──HTTP──▶  FastAPI (main.py)  ──▶  feature module (health_report.py)  ──▶  llm.py  ──▶  Gemini
 streamlit_app.py         endpoints, validation      prompt + output schema              one shared client
```

- **`llm.py`** creates the Gemini client once. Every feature imports `llm` from here, so the model name, timeout and retries live in one place.
- **Feature modules** (like `health_report.py`) hold the prompt, the Pydantic output schema and one main function. They know nothing about HTTP.
- **`main.py`** only handles HTTP: reading the request, validating input, calling a feature function and returning errors with the right status code.
- **`streamlit_app.py`** only talks to the API over HTTP. It never imports the feature code directly, so the UI and API can run separately.

### Workflow for each new topic

1. Experiment in a notebook until it works.
2. Move the working code into a new module in `src/be_prac_agnetic_ai/` (e.g. `rag.py`).
3. Add an endpoint for it in `main.py`.
4. Add tests in `tests/` that fake the LLM call.
5. (Optional) Add a page or section to the Streamlit app.
6. Update the roadmap table above.

---

## Setup

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation) 2.x
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### 1. Install dependencies

```bash
cd be-prac-agnetic-ai
poetry config virtualenvs.in-project true   # keep the virtualenv in ./.venv
poetry install
```

### 2. Add your API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your-key-here
```

### 3. Activate the environment (optional)

```bash
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

If you don't activate it, put `poetry run` in front of every command instead.

### 4. VS Code / Jupyter

Select `.venv/bin/python` as the interpreter (Cmd+Shift+P → **Python: Select Interpreter**) and as the notebook kernel (**Select Kernel** in the top right of the notebook).
If you see `No module named ...`, the notebook is using the wrong Python.

---

## Running the apps

Run each in its own terminal from the project root.

```bash
# API → http://127.0.0.1:8000  (interactive docs at /docs)
poetry run uvicorn be_prac_agnetic_ai.main:app --reload

# Streamlit UI → http://localhost:8501  (needs the API running)
poetry run streamlit run streamlit_app.py
```

The Streamlit app calls `http://127.0.0.1:8000` by default. To point it at a different API, set `API_URL`:

```bash
API_URL=http://my-server:8000 poetry run streamlit run streamlit_app.py
```

---

## API reference

### `GET /health`

Checks that the server is up.

```json
{"status": "ok"}
```

### `POST /ask`

Ask the LLM a question.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is an AI agent?"}'
```

```json
{"answer": "An AI agent is ..."}
```

### `POST /health-report`

Upload a medical or lab report and get a health summary and an Indian diet plan.

| Field | Type | Values | Default |
|---|---|---|---|
| `file` | file | PDF, PNG, JPG, WEBP or TXT, max 10 MB | required |
| `diet_preference` | form | `vegetarian`, `non-vegetarian`, `eggetarian`, `vegan`, `jain` | `vegetarian` |
| `region` | form | `any`, `north`, `south`, `east`, `west` | `any` |

```bash
curl -X POST http://127.0.0.1:8000/health-report \
  -F "file=@samples/sample_blood_report.txt" \
  -F diet_preference=vegetarian \
  -F region=north
```

Response fields: `summary`, `key_findings` (test, value, normal range, status, explanation), `health_concerns`, `diet_plan` (meal time + options), `foods_to_include`, `foods_to_avoid`, `lifestyle_tips`, `see_doctor_for`, `disclaimer`.

| Status | Meaning |
|---|---|
| 200 | Success |
| 400 | File is empty |
| 413 | File is larger than 10 MB |
| 415 | Unsupported file type |
| 422 | File is not a medical report, or invalid form values |
| 502 | Gemini call failed (overloaded, timeout, bad API key) |

> ⚠️ The output is AI-generated and is not medical advice.

---

## Apps

### Health Report Analyzer (`health_report.py` + `streamlit_app.py`)

**Concepts covered:** multimodal input (PDF/images sent to the model as base64), structured output with Pydantic (`llm.with_structured_output(...)`), system prompts with rules, and input validation in the API.

**How it works:**
1. The user uploads a report and chooses a diet preference and region.
2. The API checks the file type and size.
3. `analyze_report()` sends the file and the instructions to Gemini.
4. Gemini returns a `HealthReport` object that matches the Pydantic schema, so there's no text parsing.
5. Streamlit shows the findings table, the meal plan cards and the lists.

### RAG app (planned)

Will go in `src/be_prac_agnetic_ai/rag.py` with its own endpoints, roughly:

- `POST /rag/ingest`: upload documents, split them into chunks, embed them and store them in a vector store
- `POST /rag/query`: retrieve the relevant chunks and answer the question using only them, with sources

---

## Testing

```bash
poetry run pytest
```

Tests use FastAPI's `TestClient` and replace LLM calls with fakes via `monkeypatch`, so they are fast, free and work without an API key or internet.

---

## Configuration

| Setting | Where | Current value |
|---|---|---|
| Model | `MODEL_NAME` in `llm.py` | `gemini-3-flash-preview` |
| LLM timeout / retries | `llm.py` | 120 s / 1 retry |
| Max upload size | `MAX_FILE_SIZE` in `main.py` | 10 MB |
| API URL for Streamlit | `API_URL` env var | `http://127.0.0.1:8000` |
| Gemini API key | `GOOGLE_API_KEY` in `.env` | — |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'dotenv'` (or any package) | You're using the system Python. Select `.venv/bin/python` as the interpreter or kernel, or use `poetry run`. |
| `pyproject.toml changed significantly since poetry.lock was last generated` | Run `poetry lock`, then `poetry install`. Or add packages with `poetry add <pkg>`, which does both. |
| `404 NOT_FOUND ... model is no longer available` | Google retired that model. Change `MODEL_NAME` in `llm.py`. |
| `503 UNAVAILABLE ... high demand` | Gemini is overloaded. Wait and retry, or switch to another model. |
| Request hangs, no log line in the uvicorn terminal | Uvicorn logs only after a response is sent. The LLM call is still running or retrying. |
| `source .venv/Scripts/activate: no such file` | That's the Windows path. On macOS use `source .venv/bin/activate`. |
| Streamlit says it cannot reach the API | Start uvicorn first, and check `API_URL`. |

---

## Useful Poetry commands

```bash
poetry add <package>            # add a dependency (updates pyproject + lock + installs)
poetry add --group dev <pkg>    # add a dev-only dependency
poetry remove <package>
poetry install                  # install everything from poetry.lock
poetry lock                     # re-resolve after editing pyproject.toml by hand
poetry env info --path          # where the virtualenv is
poetry show --tree              # dependency tree
```
