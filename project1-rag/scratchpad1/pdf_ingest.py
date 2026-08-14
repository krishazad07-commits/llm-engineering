import re
from pathlib import Path

import pymupdf
from chunker import chunk_text

PDF_PATH = Path(
    "C:\\Users\\Krish\\Documents\\code\\llm-engineering\\project1-rag\\data\\berkshire_2023.pdf"
)


def clean_text(text: str) -> str:
    """Remove dot leaders, page-number footers, and normalize whitespace."""
    # collapse 3+ consecutive dots (with optional spaces) into a single space
    text = re.sub(r"(\s*\.\s*){3,}", " ", text)
    # remove trailing standalone page numbers (e.g. lonely "17" at end)
    text = re.sub(r"\n\s*\d{1,4}\s*$", "", text)
    # collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # collapse triple+ newlines to double
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text page by page, with page number metadata."""
    doc = pymupdf.open(pdf_path)
    pages = []
    dropped = []
    for page_num, page in enumerate(doc, start=1):
        raw = page.get_text()
        cleaned = clean_text(raw)
        if cleaned:
            pages.append({"page": page_num, "text": cleaned})
        else:
            dropped.append(page_num)
    doc.close()
    if dropped:
        print(f"[info] Dropped {len(dropped)} blank page(s): {dropped}")
    return pages


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}.")

    pages = extract_pages(PDF_PATH)

    print(f"Extracted {len(pages)} non-empty pages.")
    print(f"Total characters: {sum(len(p['text']) for p in pages):,}")

    # NEW: chunk every page
    all_chunks = []
    for page in pages:
        page_chunks = chunk_text(page["text"], max_size=1000, overlap=150)
        for chunk_idx, chunk_text_content in enumerate(page_chunks):
            all_chunks.append(
                {
                    "page": page["page"],
                    "chunk_id": chunk_idx,
                    "text": chunk_text_content,
                }
            )

    print(f"\nProduced {len(all_chunks)} chunks total.")
    print(
        f"Avg chunk size: {sum(len(c['text']) for c in all_chunks) // len(all_chunks)} chars"
    )
    print(f"Min chunk size: {min(len(c['text']) for c in all_chunks)}")
    print(f"Max chunk size: {max(len(c['text']) for c in all_chunks)}")

    # show a few sample chunks
    print("\n" + "=" * 60)
    print("FIRST CHUNK (page 1, chunk 0):")
    print("=" * 60)
    print(all_chunks[0]["text"])

    print("\n" + "=" * 60)
    print(
        f"MIDDLE CHUNK (page {all_chunks[len(all_chunks) // 2]['page']}, chunk {all_chunks[len(all_chunks) // 2]['chunk_id']}):"
    )
    print("=" * 60)
    print(all_chunks[len(all_chunks) // 2]["text"])


if __name__ == "__main__":
    main()
