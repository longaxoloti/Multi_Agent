from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    content: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(content or "")


def build_document_chunks(
    content: str,
    source_url: str,
    title: str,
    topic: str,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    chunks = chunk_text(content)
    base_meta = {
        "source_url": source_url,
        "title": title,
        "topic": topic,
        "chunk_count": len(chunks),
    }
    if metadata:
        base_meta.update(metadata)

    documents: list[dict[str, Any]] = []
    for idx, text in enumerate(chunks):
        meta = dict(base_meta)
        meta["chunk_index"] = idx
        documents.append({"content": text, "metadata": meta})

    return documents


def build_document_chunks_from_crawl_result(
    crawl_result: dict[str, Any],
    topic: str = "general",
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Convenience wrapper for article extraction results.

    Accepts the dict returned by ``article_extractor.extract_article()`` or each
    element of ``extract_articles()`` and builds RAG document chunks from the
    already-clean markdown content.

    Args:
        crawl_result: Dict with at least 'content', 'url', 'title', 'success'.
        topic: Topic label for metadata.
        metadata: Extra metadata to attach to each chunk.

    Returns:
        List of chunk dicts ready for vector-store insertion (empty if the
        crawl was unsuccessful or content is blank).
    """
    if not crawl_result.get("success") or not crawl_result.get("content"):
        return []

    return build_document_chunks(
        content=crawl_result["content"],
        source_url=crawl_result.get("url", ""),
        title=crawl_result.get("title", ""),
        topic=topic,
        metadata=metadata,
    )
