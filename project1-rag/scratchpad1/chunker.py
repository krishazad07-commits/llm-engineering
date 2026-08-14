def _split_recursive(text: str, max_size: int, separators: list[str]) -> list[str]:
    """Split text using separators in order of preference until every piece fits."""
    # base case: already small enough
    if len(text) <= max_size:
        return [text]

    # no separators left — hard slice (last resort, only reached for pathological input)
    if not separators:
        return [text[i : i + max_size] for i in range(0, len(text), max_size)]

    sep = separators[0]
    rest = separators[1:]

    # empty separator = split into characters
    if sep == "":
        return [text[i : i + max_size] for i in range(0, len(text), max_size)]

    pieces = text.split(sep)
    result = []
    for piece in pieces:
        if len(piece) <= max_size:
            result.append(piece)
        else:
            # this piece is still too big — recurse with the next-best separator
            result.extend(_split_recursive(piece, max_size, rest))
    return result


def _merge_pieces(
    pieces: list[str], max_size: int, overlap: int, joiner: str = " "
) -> list[str]:
    """Merge small pieces greedily up to max_size, with overlap between chunks."""
    if overlap >= max_size:
        raise ValueError(f"overlap ({overlap}) must be less than max_size ({max_size})")

    chunks: list[str] = []
    current = ""

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        # would adding this piece overflow?
        candidate_len = (
            len(current) + len(joiner) + len(piece) if current else len(piece)
        )

        if candidate_len > max_size and current:
            # save current chunk, start a new one with overlap
            chunks.append(current)
            # take the last `overlap` characters of current as the head of the new chunk
            tail = current[-overlap:] if overlap > 0 else ""
            # if the new piece alone is bigger than (max_size - overlap), skip overlap entirely
            if len(piece) + len(tail) + len(joiner) > max_size:
                current = piece
            else:
                current = tail + joiner + piece if tail else piece
        else:
            current = current + joiner + piece if current else piece

    if current:
        chunks.append(current)

    return chunks


def chunk_text(
    text: str,
    max_size: int = 1000,
    overlap: int = 150,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Recursively split text into chunks of approximately max_size characters,
    with overlap between adjacent chunks.

    Preference order for split boundaries: paragraph > line > sentence > word > character.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if overlap >= max_size:
        raise ValueError(f"overlap ({overlap}) must be less than max_size ({max_size})")

    pieces = _split_recursive(text, max_size, separators)
    chunks = _merge_pieces(pieces, max_size, overlap)
    return chunks


# ------------------------------------------------------------------
# quick smoke test — runs only when this file is executed directly
# ------------------------------------------------------------------


def _demo():
    sample = """
    First paragraph. It has two sentences. This is the second sentence.
    
    Second paragraph is a bit longer. It contains multiple sentences too. This is meant to test the sentence splitter. And another sentence for good measure.
    
    Third paragraph. Short one.
    
    Fourth paragraph is again a bit longer to force multiple chunks when max_size is small. It should demonstrate how the recursive splitter handles chunking across paragraph boundaries when combined with overlap between chunks.
    """.strip()

    for size, overlap in [(100, 20), (200, 40), (500, 75)]:
        print(f"\n{'=' * 60}")
        print(f"max_size={size}, overlap={overlap}")
        print("=" * 60)
        chunks = chunk_text(sample, max_size=size, overlap=overlap)
        print(f"Produced {len(chunks)} chunks")
        for i, c in enumerate(chunks, start=1):
            print(f"\n--- chunk {i} (len={len(c)}) ---")
            print(c)


if __name__ == "__main__":
    _demo()
