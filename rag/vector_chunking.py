from __future__ import annotations

from math import sqrt
from typing import Any, Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text_for_embedding(
    content: str,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    max_chunks: int = 128,
) -> list[str]:
    """Split long text into embedding-safe chunks.

    This splitter is dedicated to vector embedding/storage workflows and keeps
    chunk sizes conservative to reduce model context overflow.
    """
    text = (content or "").strip()
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [chunk.strip() for chunk in splitter.split_text(text) if chunk and chunk.strip()]
    if max_chunks > 0:
        return chunks[:max_chunks]
    return chunks


def build_vector_chunks(
    content: str,
    *,
    source_url: str,
    title: str,
    topic: str,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    max_chunks: int = 128,
) -> list[dict[str, Any]]:
    """Build structured chunks ready for embedding/vector persistence."""
    chunks = chunk_text_for_embedding(
        content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunks=max_chunks,
    )
    if not chunks:
        return []

    base_meta = {
        "source_url": source_url,
        "title": title,
        "topic": topic,
        "chunk_count": len(chunks),
        "pipeline": "vector_chunking",
    }
    if metadata:
        base_meta.update(metadata)

    docs: list[dict[str, Any]] = []
    for idx, text in enumerate(chunks):
        item_meta = dict(base_meta)
        item_meta["chunk_index"] = idx
        item_meta["chunk_chars"] = len(text)
        docs.append({"content": text, "metadata": item_meta})
    return docs


def aggregate_chunk_embeddings(
    vectors: Iterable[Iterable[float]],
    *,
    weights: Iterable[float] | None = None,
    l2_normalize: bool = True,
) -> list[float]:
    """Aggregate chunk vectors into one vector via weighted average.

    Useful when you still want one representative vector per document after
    embedding chunks independently.
    """
    vector_list = [list(map(float, vec)) for vec in vectors if vec is not None]
    if not vector_list:
        return []

    dim = len(vector_list[0])
    if dim == 0:
        return []
    if any(len(vec) != dim for vec in vector_list):
        raise ValueError("all vectors must share the same dimension")

    if weights is None:
        weight_list = [1.0] * len(vector_list)
    else:
        weight_list = [float(w) for w in weights]
        if len(weight_list) != len(vector_list):
            raise ValueError("weights length must match vectors length")

    total_weight = sum(weight_list)
    if total_weight <= 0:
        raise ValueError("sum(weights) must be > 0")

    merged = [0.0] * dim
    for vec, w in zip(vector_list, weight_list):
        for i in range(dim):
            merged[i] += vec[i] * w

    merged = [value / total_weight for value in merged]

    if not l2_normalize:
        return merged

    norm = sqrt(sum(value * value for value in merged))
    if norm <= 0.0:
        return merged
    return [value / norm for value in merged]
