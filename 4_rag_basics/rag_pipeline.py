"""
Steps 1-5 of rag_demo.ipynb in one function.
Lets you run Step 6 onwards without running the earlier cells first.
Keep the settings here in sync with the notebook.
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = Path(__file__).parent / "telecom_guide.pdf"

SYSTEM_PROMPT = """\
You are a helpful telecom assistant.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.

Context:
{context}
"""


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag():
    """Run Steps 1-5 and return (retriever, chain)."""
    load_dotenv()

    # Step 1: load the PDF
    pages = PyPDFLoader(str(PDF_PATH)).load()

    # Step 2: split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)

    # Step 3: embed and store in Chroma
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embeddings)

    # Step 4: retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Step 5: RAG chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0,
        reasoning_format="parsed",
    )
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"RAG pipeline built: {len(pages)} pages, {len(chunks)} chunks.")
    return retriever, chain
