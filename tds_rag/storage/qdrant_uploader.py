"""Step 4 - Create the Qdrant collection (if needed) and upsert vectors."""
from __future__ import annotations

import json

import httpx

from tds_rag import config


def _base() -> tuple[str, dict]:
    url = config.require_env("QDRANT_URL").rstrip("/")
    headers = {"api-key": config.require_env("QDRANT_API_KEY"), "Content-Type": "application/json"}
    return url, headers


def ensure_collection() -> None:
    url, headers = _base()
    coll = f"{url}/collections/{config.QDRANT_COLLECTION}"
    r = httpx.get(coll, headers=headers, timeout=30.0)
    if r.status_code == 200:
        print(f"Collection '{config.QDRANT_COLLECTION}' already exists")
        return
    body = {"vectors": {"size": config.EMBEDDING_DIM, "distance": "Cosine"}}
    httpx.put(coll, headers=headers, json=body, timeout=30.0).raise_for_status()
    print(f"Created collection '{config.QDRANT_COLLECTION}' ({config.EMBEDDING_DIM}d, Cosine)")


def is_valid_point(point: dict) -> bool:
    emb = point.get("embedding")
    return (
        isinstance(point.get("id"), (str, int))
        and isinstance(emb, list)
        and len(emb) == config.EMBEDDING_DIM
        and all(isinstance(x, float) for x in emb)
    )


def upload_batch(points: list[dict]) -> None:
    url, headers = _base()
    r = httpx.put(
        f"{url}/collections/{config.QDRANT_COLLECTION}/points?wait=true",
        headers=headers, json={"points": points}, timeout=60.0,
    )
    r.raise_for_status()


def main() -> None:
    if not config.VECTORS_FILE.exists():
        raise FileNotFoundError(f"{config.VECTORS_FILE} not found - run `make embed` first.")
    ensure_collection()

    with open(config.VECTORS_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    points = []
    for p in raw:
        if not is_valid_point(p):
            print(f"Skipped invalid vector: {p.get('id')}")
            continue
        points.append({"id": p["id"], "vector": p["embedding"], "payload": p.get("metadata", {})})
    print(f"Valid vectors to upload: {len(points)}")

    size = config.UPLOAD_BATCH_SIZE  # small batches avoid HTTP 413 with image payloads
    for i in range(0, len(points), size):
        upload_batch(points[i : i + size])
        print(f"Uploaded {min(i + size, len(points))}/{len(points)}")
    print("Upload complete")


if __name__ == "__main__":
    main()
