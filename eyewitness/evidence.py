"""Restore whitespace-only quote drift to a real, contiguous source substring."""

import re


def source_quote(text: str, quote: str) -> str | None:
    if not quote or not quote.strip():
        return None
    if quote in text:
        return quote
    # Keep every non-whitespace character and every word boundary. Never remove
    # punctuation, merge messages, or accept paraphrases/fuzzy string matches.
    parts = re.split(r"\s+", quote.strip())
    pattern = r"\s+".join(re.escape(part) for part in parts)
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        return None
    restored = matches[0].group()
    return restored if len(restored) <= 1000 else None
