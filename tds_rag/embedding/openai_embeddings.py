"""Thin client for the OpenAI-compatible /embeddings endpoint (AIPipe by default)."""
from __future__ import annotations

import httpx

from tds_rag import config

EMBEDDING_URL = f"{config.OPENAI_BASE_URL}/embeddings"


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.get_openai_key()}",
        "Content-Type": "application/json",
    }


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed many texts in one request (sync; used by the offline pipeline)."""
    payload = {"model": config.EMBEDDING_MODEL, "input": texts}
    resp = httpx.post(EMBEDDING_URL, headers=_headers(), json=payload, timeout=60.0)
    resp.raise_for_status()
    data = sorted(resp.json()["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data]


async def aembed(text: str) -> list[float]:
    """Embed one text (async; used by the API at query time)."""
    payload = {"model": config.EMBEDDING_MODEL, "input": text}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(EMBEDDING_URL, headers=_headers(), json=payload)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
