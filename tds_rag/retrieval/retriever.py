"""Step 5 - Retrieve: embed the question, run vector search, build context + source links."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from textwrap import shorten

from qdrant_client import QdrantClient

from tds_rag import config
from tds_rag.embedding.openai_embeddings import aembed


@dataclass
class Retrieved:
    context: str
    images: list[str] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.context)


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(
        url=config.require_env("QDRANT_URL"),
        api_key=config.require_env("QDRANT_API_KEY"),
    )


def _search(vector: list[float], limit: int):
    client = get_client()
    if hasattr(client, "query_points"):  # newer qdrant-client
        return client.query_points(
            collection_name=config.QDRANT_COLLECTION, query=vector, limit=limit
        ).points
    return client.search(  # older clients
        collection_name=config.QDRANT_COLLECTION, query_vector=vector, limit=limit
    )


def summarize(text: str, max_len: int = 120) -> str:
    return shorten(text.strip().replace("\n", " "), width=max_len, placeholder="...")


def build_retrieved(hits) -> Retrieved:
    blocks, images, links = [], [], []
    for hit in hits:
        payload = hit.payload or {}
        text = payload.get("text", "").strip()
        blocks.append(text)
        snippet = summarize(text)

        if payload.get("source") == "thread":
            imgs = payload.get("images_base64", [])
            if isinstance(imgs, list):
                images.extend(imgs)
            if "post_url" in payload:
                links.append({"url": payload["post_url"], "text": snippet})
        else:
            links.append({"url": payload.get("url", ""), "text": snippet})
    return Retrieved(context="\n\n".join(blocks), images=images, links=links)


async def retrieve(question: str, top_k: int = config.TOP_K) -> Retrieved:
    vector = await aembed(question)
    return build_retrieved(_search(vector, top_k))
