"""Step 3 - Embed course chunks + Discourse posts and write data/embeddings/vectors.json."""
from __future__ import annotations

import json
import uuid

from tds_rag import config
from tds_rag.embedding.openai_embeddings import embed_batch


def _embed_all(texts: list[str]) -> list[list[float] | None]:
    """Embed in batches; if a batch fails, retry item-by-item so one bad text doesn't drop the batch."""
    size = config.EMBED_BATCH_SIZE
    out: list[list[float] | None] = []
    for i in range(0, len(texts), size):
        batch = texts[i : i + size]
        try:
            out.extend(embed_batch(batch))
        except Exception as e:  # noqa: BLE001
            print(f"Batch {i // size} failed ({e}); retrying one by one")
            for t in batch:
                try:
                    out.append(embed_batch([t])[0])
                except Exception as e2:  # noqa: BLE001
                    print(f"  skipped one text: {e2}")
                    out.append(None)
        print(f"Embedded {min(i + size, len(texts))}/{len(texts)}")
    return out


def _chunk_records() -> tuple[list[str], list[dict]]:
    with open(config.CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    texts, metas = [], []
    for c in chunks:
        text = c.get("content", "").strip()
        if text:
            texts.append(text)
            metas.append(
                {"text": text, "source": "chunk", "original_id": c.get("id", ""), "url": c.get("url", "")}
            )
    return texts, metas


def _thread_records() -> tuple[list[str], list[dict]]:
    with open(config.THREADS_FILE, encoding="utf-8") as f:
        threads = json.load(f)
    texts, metas = [], []
    for t in threads:
        for post in t.get("posts", []):
            text = post.get("text", "").strip()
            if text:
                texts.append(text)
                metas.append(
                    {
                        "text": text,
                        "source": "thread",
                        "thread_title": t.get("thread_title", ""),
                        "post_url": post.get("url"),
                        "created_by": post.get("created_by"),
                        "type": post.get("type"),
                        "images_base64": post.get("images_base64", []),
                    }
                )
    return texts, metas


def main() -> None:
    for f in (config.CHUNKS_FILE, config.THREADS_FILE):
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")

    texts, metas = [], []
    for t, m in (_chunk_records(), _thread_records()):
        texts += t
        metas += m

    embeddings = _embed_all(texts)
    vectors = [
        {"id": str(uuid.uuid4()), "embedding": emb, "metadata": meta}
        for emb, meta in zip(embeddings, metas)
        if emb is not None
    ]

    config.VECTORS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.VECTORS_FILE, "w", encoding="utf-8") as f:
        json.dump(vectors, f)
    print(f"{len(vectors)} vectors saved to {config.VECTORS_FILE}")


if __name__ == "__main__":
    main()
