"""Tolerate only extra response fields, without relaxing the protocol schema."""

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def parse_output(schema: type[T], raw: str) -> tuple[T, int]:
    """Validate first; strip only the exact keys flagged as extra by Pydantic.

    The schema remains strict (including nested evidence and configuration).
    Missing fields, wrong types/enums and oversized lists still fail. Do not
    turn a completely unrelated object into the protocol's default empty list.
    Returned values never include the discarded fields or their contents.
    """
    try:
        return schema.model_validate_json(raw), 0
    except ValidationError as exc:
        errors = exc.errors(include_input=False)
        if not errors or any(error["type"] != "extra_forbidden" for error in errors):
            raise
        value = json.loads(raw)
        if not isinstance(value, dict) or not schema.model_fields.keys() & value.keys():
            raise
        for error in errors:
            target = value
            location = error["loc"]
            for part in location[:-1]:
                if isinstance(target, dict) and part in target:
                    target = target[part]
                elif isinstance(target, list) and isinstance(part, int) and 0 <= part < len(target):
                    target = target[part]
                else:
                    # Refuse ambiguous paths (e.g. a future union branch), not
                    # arbitrary recursively named keys in otherwise valid data.
                    raise exc
            if not location or not isinstance(target, dict) or location[-1] not in target:
                raise exc
            del target[location[-1]]
        return schema.model_validate_json(json.dumps(value, ensure_ascii=False)), len(errors)
