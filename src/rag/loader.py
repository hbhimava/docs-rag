"""
src/rag/loader.py

Stage 1 of the RAG pipeline: loading raw documents.

Walk a directory of markdown files, read each one, clean it, and return
a list of Document objects carrying content + metadata.

Cleaning is pragmatic, not exhaustive: we strip the high-frequency
noise in FastAPI's docs (frontmatter, code-includes, the H1 line,
MkDocs admonition delimiters, rendered-CLI box-drawing art). Deeper
markdown normalization (tables, tabbed content, HTML) is a documented
known limitation, deferred by design under the time budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


EXCLUDED_FILENAMES = {
    "release-notes.md",
    "fastapi-people.md",
    "_llm-test.md",
    "external-links.md",
    "newsletter.md",
    "management.md",
    "management-tasks.md",
    "contributing.md",
}

MIN_CONTENT_CHARS = 50

# Unicode ranges for box-drawing glyphs (U+2500-U+257F) and block
# elements (U+2580-U+259F). FastAPI docs embed rendered CLI output
# that uses these; they carry no semantic meaning for retrieval.
_BOX = r"\u2500-\u257F\u2580-\u259F"


@dataclass
class Document:
    """A single source document with its text content and metadata."""
    content: str
    source: str
    title: str
    section: str
    metadata: dict = field(default_factory=dict)


def _derive_title(text: str, path: Path) -> str:
    """Pull the title from the first markdown H1 (operates on RAW text)."""
    match = re.search(r"^\#\s+(.+)$", text, flags=re.MULTILINE)
    if match:
        title = match.group(1).strip()
        title = re.sub(r"\s*\{[^}]*\}\s*$", "", title).strip()
        return title

    stem = path.stem.replace("-", " ").replace("_", " ")
    return stem.title()


def _clean_markdown(text: str) -> str:
    """Strip the high-frequency noise from FastAPI markdown."""
    # 1. YAML frontmatter (only if the file starts with '---').
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL)

    # 2. FastAPI code-include directives {!..!}.
    text = re.sub(r"\{!.*?!\}", "", text)

    # 3. The first H1 line (its text is captured as the title and
    #    prepended by the chunker; keeping it here duplicates it).
    text = re.sub(r"^\#\s+.+$", "", text, count=1, flags=re.MULTILINE)

    # 4. MkDocs admonition delimiters: lines that start with '///'
    #    (e.g. '/// warning', '/// info', or a bare closing '///').
    #    We delete the delimiter LINES but keep the content between them.
    text = re.sub(r"^/{3}.*$", "", text, flags=re.MULTILINE)

    # 5. Rendered-CLI box-drawing art:
    #    5a. Any line that is essentially only box/block chars (plus
    #        whitespace) and whatever trails on that line -> drop it.
    text = re.sub(
        rf"^[\s{_BOX}]*[{_BOX}][\s{_BOX}]*.*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    #    5b. Any box/block chars left inline -> replace with a space.
    text = re.sub(rf"[{_BOX}]+", " ", text)

    # 6. Collapse long runs of spaces/tabs (CLI art leaves big gaps).
    text = re.sub(r"[ \t]{3,}", " ", text)

    # 7. Collapse runs of 3+ newlines into exactly 2.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_documents(docs_dir: str | Path) -> list[Document]:
    """Walk docs_dir recursively, load every .md file (except excluded),
    clean it, and return a list of Document objects.
    """
    docs_dir = Path(docs_dir)

    if not docs_dir.exists():
        raise FileNotFoundError(
            f"docs_dir does not exist: {docs_dir.resolve()}"
        )

    documents: list[Document] = []

    for file_path in sorted(docs_dir.rglob("*.md")):
        if file_path.name in EXCLUDED_FILENAMES:
            continue

        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")

        # Title FIRST (raw text, H1 still present), THEN clean.
        title = _derive_title(raw_text, file_path)
        cleaned = _clean_markdown(raw_text)

        if len(cleaned) < MIN_CONTENT_CHARS:
            continue

        relative_path = file_path.relative_to(docs_dir)
        source = relative_path.as_posix()

        if len(relative_path.parts) > 1:
            section = relative_path.parts[0]
        else:
            section = "root"

        documents.append(
            Document(
                content=cleaned,
                source=source,
                title=title,
                section=section,
                metadata={
                    "char_count": len(cleaned),
                    "abs_path": str(file_path.resolve()),
                },
            )
        )

    return documents


if __name__ == "__main__":
    docs = load_documents("data/raw/fastapi/docs/en/docs")
    print(f"Loaded {len(docs)} documents\n")

    for d in docs[:8]:
        print(f"  [{d.section:<10}] {d.title:<45} "
              f"({d.source})  {d.metadata['char_count']} chars")

    total_chars = sum(d.metadata["char_count"] for d in docs)
    sections = sorted({d.section for d in docs})
    print(f"\nTotal: {len(docs)} docs, {total_chars:,} chars")
    print(f"Sections: {', '.join(sections)}")
