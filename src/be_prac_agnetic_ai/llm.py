"""
LLM helper functions.
Reads GOOGLE_API_KEY from .env and talks to Gemini through LangChain.
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MODEL_NAME = "gemini-3-flash-preview"

# create the client once and reuse it for every call
# timeout/max_retries: fail fast with an error instead of hanging when Gemini is overloaded
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, timeout=120, max_retries=1)


def ask(question: str) -> str:
    """Send a question to the LLM and return the answer text."""
    response = llm.invoke(question)
    # newer Gemini models return content as a list of blocks, .text joins them into one string
    return response.text
