"""Step 2 - Split course markdown into token-bounded chunks (paragraph-aware)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from tds_rag import config

TOKENIZER_MODEL = "gpt-4o"
_MD_SUFFIX = re.compile(r"\.md.*$")


def _default_token_counter() -> Callable[[str], int]:
    import tiktoken  # imported lazily so tests / API don't need it

    enc = tiktoken.encoding_for_model(TOKENIZER_MODEL)
    return lambda text: len(enc.encode(text))


def split_markdown(
    text: str,
    max_tokens: int = config.CHUNK_TOKEN_LIMIT,
    count_tokens: Callable[[str], int] | None = None,
) -> list[str]:
    """Greedily pack paragraphs into chunks of at most `max_tokens` tokens.

    A single paragraph larger than `max_tokens` becomes its own chunk.
    """
    count_tokens = count_tokens or _default_token_counter()
    chunks: list[str] = []
    current, current_tokens = "", 0

    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        tokens = count_tokens(para)
        if current and current_tokens + tokens > max_tokens:
            chunks.append(current.strip())
            current, current_tokens = para, tokens
        else:
            current = f"{current}\n\n{para}" if current else para
            current_tokens += tokens

    if current.strip():
        chunks.append(current.strip())
    return chunks


def page_url(path: Path) -> str:
    """docs/embeddings.md -> https://tds.s-anand.net/#/embeddings"""
    return f"{config.COURSE_SITE_URL}{_MD_SUFFIX.sub('', path.name)}"


def build_chunks(input_dir: Path = config.COURSE_REPO_DIR) -> list[dict]:
    counter = _default_token_counter()
    all_chunks: list[dict] = []

    for path in sorted(input_dir.rglob("*.md")):
        if ".git" in path.parts:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"Skipped {path}: {e}")
            continue

        rel = path.relative_to(input_dir).as_posix()
        for idx, chunk in enumerate(split_markdown(content, count_tokens=counter)):
            all_chunks.append({"id": f"{rel}#{idx}", "content": chunk, "url": page_url(path)})
    return all_chunks


def main() -> None:
    if not config.COURSE_REPO_DIR.exists():
        raise FileNotFoundError(
            f"{config.COURSE_REPO_DIR} not found - run `make scrape-course` first."
        )
    chunks = build_chunks()
    config.CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"{len(chunks)} chunks written to {config.CHUNKS_FILE}")


if __name__ == "__main__":
    main()
