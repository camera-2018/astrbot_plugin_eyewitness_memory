"""Provider-reported usage; missing usage is never represented as zero."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelReply:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None

    @classmethod
    def from_response(cls, response):
        usage = getattr(response, "usage", None)

        def count(name):
            value = getattr(usage, name, None)
            return value if type(value) is int and value >= 0 else None

        return cls(
            response.completion_text or "", count("input"), count("output"), count("input_cached")
        )
