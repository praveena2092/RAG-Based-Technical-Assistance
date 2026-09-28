from types import SimpleNamespace

from tds_rag.retrieval.retriever import build_retrieved


def hit(payload):
    return SimpleNamespace(payload=payload)


def test_build_retrieved_mixes_course_chunks_and_threads():
    r = build_retrieved([
        hit({"text": "course text", "source": "chunk", "url": "https://tds.s-anand.net/#/docker"}),
        hit({"text": "thread text", "source": "thread", "post_url": "https://discourse/x/1", "images_base64": ["AAA"]}),
    ])
    assert "course text" in r.context and "thread text" in r.context
    assert r.images == ["AAA"]
    assert [l["url"] for l in r.links] == ["https://tds.s-anand.net/#/docker", "https://discourse/x/1"]


def test_empty_is_falsy():
    assert not build_retrieved([])
