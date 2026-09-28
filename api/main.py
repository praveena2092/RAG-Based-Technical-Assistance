"""FastAPI entrypoint (Vercel serverless function). POST /api {question, image?}."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `tds_rag` importable on Vercel

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tds_rag.generation.generator import generate_answer
from tds_rag.retrieval.retriever import retrieve

app = FastAPI(title="TDS Virtual TA", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to your frontend origin(s) in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryPayload(BaseModel):
    question: str
    image: Optional[str] = None  # base64-encoded user image


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api")
async def query_api(payload: QueryPayload):
    question = payload.question.strip()
    if not question:
        return {"answer": "Invalid input: Question cannot be empty.", "links": []}

    try:
        retrieved = await retrieve(question)
    except Exception as e:  # noqa: BLE001
        return {"answer": f"Retrieval failed: {e}", "links": []}

    if not retrieved:
        return {"answer": "No relevant results found.", "links": []}

    images = list(retrieved.images)
    if payload.image:
        images.insert(0, payload.image)

    try:
        answer = await generate_answer(question, retrieved.context, images)
    except Exception as e:  # noqa: BLE001
        return {"answer": f"GPT generation failed: {e}", "links": retrieved.links}

    return {"answer": answer, "links": retrieved.links}
