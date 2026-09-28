"""Step 6 - Generate: answer the question from retrieved context (+ optional images)."""
from __future__ import annotations

import httpx

from tds_rag import config

CHAT_URL = f"{config.OPENAI_BASE_URL}/chat/completions"

SYSTEM_PROMPT = """
You are a helpful assistant that answers academic and technical questions using the context provided.
Guidelines:
- "answer" should contain a concise and accurate response to the user's question.
- If an image is provided, use it along with the text context to answer.
- If a tool or model is mentioned, explain its usage and whether it's supported.
- If one or more tools or models are mentioned, explain whether it can be used or not.
- If the user asks for a specific model, always clarify whether that model is available or not.
- If the user asks about a model, clarify its intended use and explain whether it is supported in this setup.
""".strip()


def build_messages(question: str, context: str, images: list[str]) -> list[dict]:
    content = [{"type": "text", "text": f"Context:\n{context}\n\nQuestion: {question}"}]
    for img in images:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


async def generate_answer(question: str, context: str, images: list[str]) -> str:
    headers = {
        "Authorization": f"Bearer {config.get_openai_key()}",
        "Content-Type": "application/json",
    }
    payload = {"model": config.CHAT_MODEL, "messages": build_messages(question, context, images)}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(CHAT_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
