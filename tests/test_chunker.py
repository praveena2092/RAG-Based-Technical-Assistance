from pathlib import Path

from tds_rag.chunking.chunker import page_url, split_markdown

words = lambda s: len(s.split())  # noqa: E731


def test_packs_paragraphs_within_limit():
    text = "one two\n\nthree four\n\nfive six"
    assert split_markdown(text, max_tokens=4, count_tokens=words) == ["one two\n\nthree four", "five six"]


def test_no_empty_chunk_when_first_paragraph_is_oversized():
    chunks = split_markdown("a b c d e f\n\ng h", max_tokens=3, count_tokens=words)
    assert chunks == ["a b c d e f", "g h"]
    assert all(chunks)


def test_page_url():
    assert page_url(Path("docs/embeddings.md")) == "https://tds.s-anand.net/#/embeddings"
