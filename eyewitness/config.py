"""AstrBot owns configuration; SQLite keeps a migration/rollback snapshot only."""

from __future__ import annotations

import copy
import hashlib
import json

from .models import Settings


class ConfigSettings:
    def __init__(self, config):
        self.config = config

    def read(self) -> Settings:
        values = dict(self.config)
        return Settings.model_validate({k: values[k] for k in Settings.model_fields if k in values})

    def write(self, settings: Settings, qdrant_api_key: str | None = None):
        patch = {**settings.model_dump(), "config_version": 1}
        if qdrant_api_key is not None:
            patch["qdrant_api_key"] = qdrant_api_key
        before = copy.deepcopy(dict(self.config))
        try:
            self.config.save_config(patch)
        except Exception:
            # AstrBot mutates its dict before committing the file; restore on failure.
            self.config.clear()
            self.config.update(before)
            raise

    def migrate(self, legacy: Settings | None) -> Settings:
        current = self.read()
        if self.config.get("config_version", 0) < 1:
            if legacy is not None:
                # Schema defaults are added by AstrBot before plugin initialization.
                # Preserve existing SQLite choices; explicit non-default native choices win.
                defaults = Settings().model_dump()
                overrides = {k: v for k, v in current.model_dump().items() if v != defaults[k]}
                current = Settings.model_validate({**legacy.model_dump(), **overrides})
            self.write(current)
        return current


def settings_revision(settings: Settings) -> str:
    return hashlib.sha256(json.dumps(settings.model_dump(), sort_keys=True).encode()).hexdigest()
