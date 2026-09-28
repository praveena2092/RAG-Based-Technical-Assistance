"""Step 1b - Ingest TDS Discourse threads (requires your own session cookies).

Set DISCOURSE_SESSION_COOKIE (_forum_session) and DISCOURSE_T_COOKIE (_t) in `.env`.
Optional: --start / --end (YYYY-MM-DD) to change the date window.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from tds_rag import config

HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_cookies() -> dict[str, str]:
    return {
        "_forum_session": config.require_env("DISCOURSE_SESSION_COOKIE"),
        "_t": config.require_env("DISCOURSE_T_COOKIE"),
    }


def get_post_type(post: dict, idx: int) -> str:
    if idx == 0:
        return "question"
    if post.get("reply_to_post_number") == 1:
        return "answer"
    return "follow-up"


def clean_html(content: str, cookies: dict) -> tuple[str, list[str]]:
    """Strip HTML to text; download inline images and return them base64-encoded."""
    soup = BeautifulSoup(content, "html.parser")
    images_base64: list[str] = []
    config.DISCOURSE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for img in soup.find_all("img"):
        classes = img.get("class", [])
        if "emoji" in classes or "avatar" in classes:
            img.decompose()
            continue
        src = img.get("src")
        try:
            res = requests.get(src, headers=HEADERS, cookies=cookies, timeout=30)
            if res.status_code == 200:
                img_name = os.path.basename(src.split("?")[0])
                (config.DISCOURSE_IMAGES_DIR / img_name).write_bytes(res.content)
                images_base64.append(base64.b64encode(res.content).decode("utf-8"))
                img.replace_with(f"[Image: {img_name}]")
        except Exception:  # noqa: BLE001
            img.decompose()

    return soup.get_text(separator="\n").strip(), images_base64


def fetch_thread(topic_id: int, slug: str, cookies: dict) -> dict | None:
    base = config.DISCOURSE_BASE_URL
    r = requests.get(f"{base}/t/{slug}/{topic_id}.json", headers=HEADERS, cookies=cookies, timeout=30)
    if r.status_code != 200:
        return None

    data = r.json()
    thread = {
        "thread_title": data["title"],
        "thread_url": f"{base}/t/{slug}/{topic_id}",
        "posts": [],
    }
    for idx, post in enumerate(data["post_stream"]["posts"]):
        text, images_b64 = clean_html(post["cooked"], cookies)
        thread["posts"].append(
            {
                "type": get_post_type(post, idx),
                "text": text,
                "url": f"{base}/t/{slug}/{topic_id}/{post['post_number']}",
                "created_by": post["username"],
                "created_at": post["created_at"],
                "images_base64": images_b64,
            }
        )
    return thread


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-04-14")
    args = parser.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    cookies = get_cookies()
    output, page = [], 0

    while True:
        r = requests.get(
            f"{config.DISCOURSE_CATEGORY_URL}.json?page={page}",
            headers=HEADERS, cookies=cookies, timeout=30,
        )
        if r.status_code != 200:
            break
        topics = r.json().get("topic_list", {}).get("topics", [])
        if not topics:
            break

        for topic in topics:
            created = datetime.strptime(topic["created_at"][:10], "%Y-%m-%d")
            if not (start <= created <= end):
                continue
            print(f"Fetching thread: {topic['slug']}")
            thread = fetch_thread(topic["id"], topic["slug"], cookies)
            if thread:
                output.append(thread)
                time.sleep(1)  # be polite to the forum
        page += 1

    config.DISCOURSE_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.THREADS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(output)} threads to {config.THREADS_FILE}")


if __name__ == "__main__":
    main()
