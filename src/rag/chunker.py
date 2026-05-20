"""
src/rag/chunker.py

Stage 2 of the RAG pipeline: splitting documents into retrieval-sized chunks.

Strategy: markdown-aware recursive splitting.
  1. Split each document on H2-H4 headers so each piece is a logical
     section that keeps its heading context.
  2. If a section is still larger than the target size, recursively
     split it on progressively smaller boundaries
     (paragraphs -> lines -> sentences -> words).
  3. Add a small overlap between consecutive chunks so ideas that
     straddle a boundary are not lost.
  4. Prefix every chunk with its title + header so the embedding
     carries that context and citations are meaningful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.loader import Document


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_CONTENT = 80
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """A retrieval unit: a slice of a Document plus inherited metadata."""
    text: str
    chunk_id: str
    source: str
    title: str
    section: str
    metadata: dict = field(default_factory=dict)


def _split_on_headers(text: str) -> list[tuple[str, str]]:
    """Split markdown into (header, body) sections on H2-H4 headings."""
    header_re = re.compile(r"^(#{2,4})\s+(.+)$", re.MULTILINE)

    sections: list[tuple[str, str]] = []
    last_index = 0
    last_header = ""

    for match in header_re.finditer(text):
        body = text[last_index:match.start()].strip()
        if body:
            sections.append((last_header, body))

        raw_heading = match.group(2).strip()
        clean_heading = re.sub(r"\s*\{[^}]*\}\s*$", "", raw_heading).strip()
        last_header = clean_heading
        last_index = match.end()

    tail = text[last_index:].strip()
    if tail:
        sections.append((last_header, tail))

    if not sections:
        sections.append(("", text.strip()))

    return sections


def _recursive_split(text: str, separators: list[str]) -> list[str]:
    """Split text into pieces <= CHUNK_SIZE, trying separators in order."""
    if len(text) <= CHUNK_SIZE:
        return [text] if text.strip() else []

    if not separators:
        return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    separator = separators[0]
    remaining = separators[1:]

    if separator == "":
        parts = list(text)
    else:
        parts = text.split(separator)

    chunks: list[str] = []
    current = ""

    for part in parts:
        piece = part if separator == "" else part + separator

        if len(current) + len(piece) <= CHUNK_SIZE:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            if len(piece) > CHUNK_SIZE:
                chunks.extend(_recursive_split(piece, remaining))
                current = ""
            else:
                current = piece

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _add_overlap(chunks: list[str]) -> list[str]:
    """Prepend the tail of each chunk to the next for cross-boundary context."""
    if len(chunks) <= 1:
        return chunks

    overlapped: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-CHUNK_OVERLAP:]
        overlapped.append(prev_tail + " " + chunks[i])

    return overlapped


def chunk_document(doc: Document) -> list[Chunk]:
    """Turn one Document into a list of Chunk objects."""
    chunks: list[Chunk] = []
    chunk_index = 0

    for header, body in _split_on_headers(doc.content):
        pieces = _recursive_split(body, SEPARATORS)
        pieces = _add_overlap(pieces)

        for piece in pieces:
            
            # Skip near-empty pieces: they make weak embeddings and
            # pollute retrieval. We measure the PIECE, not the prefixed
            # text, so the title prefix can't inflate a thin chunk.
            
            if len(piece.strip()) < MIN_CHUNK_CONTENT:
                continue
            if header:
                prefix = f"{doc.title} > {header}\n\n"
            else:
                prefix = f"{doc.title}\n\n"

            text = prefix + piece
            chunk_id = f"{doc.source}::{chunk_index}"

            chunks.append(
                Chunk(
                    text=text,
                    chunk_id=chunk_id,
                    source=doc.source,
                    title=doc.title,
                    section=doc.section,
                    metadata={
                        "header": header,
                        "char_count": len(text),
                        "doc_section": doc.section,
                    },
                )
            )
            chunk_index += 1

    return chunks


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    """Chunk a whole corpus of Documents."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


if __name__ == "__main__":
    from rag.loader import load_documents

    docs = load_documents("data/raw/fastapi/docs/en/docs")
    chunks = chunk_documents(docs)

    sizes = [c.metadata["char_count"] for c in chunks]
    print(f"Documents: {len(docs)}")
    print(f"Chunks:    {len(chunks)}")
    print(f"Avg chunk: {sum(sizes) // len(sizes)} chars")
    print(f"Min/Max:   {min(sizes)} / {max(sizes)} chars")
    print(f"Ratio:     {len(chunks) / len(docs):.1f} chunks per doc\n")

    print("--- Sample chunk (first one) ---")
    print(chunks[0].chunk_id)
    print(chunks[0].text[:400])
    print("\n--- Sample chunk (a middle one) ---")
    mid = chunks[len(chunks) // 2]
    print(mid.chunk_id, "|", mid.metadata["header"])
    print(mid.text[:400])
