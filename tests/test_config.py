import copy
import json
from pathlib import Path

import pytest

from eyewitness.admin import AdminAPI
from eyewitness.engine import Engine
from eyewitness.models import Settings
from tests.conftest import SCOPE, seed

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text())


class NativeConfig(dict):
    """Mimic AstrBot's dict mutation followed by an atomic file commit."""

    def __init__(self, **values):
        super().__init__({k: copy.deepcopy(v["default"]) for k, v in SCHEMA.items()})
        self.update(values)
        self.persisted = copy.deepcopy(dict(self))
        self.fail = False

    def save_config(self, patch):
        self.update(patch)
        if self.fail:
            raise OSError("synthetic write failure")
        self.persisted = copy.deepcopy(dict(self))


def test_registered_fields_defaults_and_native_provider_picker():
    assert set(SCHEMA) == set(Settings.model_fields) | {"qdrant_api_key", "config_version"}
    defaults = {k: SCHEMA[k]["default"] for k in Settings.model_fields}
    assert Settings.model_validate(defaults) == Settings()
    assert SCHEMA["provider_id"]["_special"] == "select_provider"
    assert SCHEMA["embedding_provider_id"]["_special"] == "select_provider_embedding"
    assert SCHEMA["embedding_provider_id"]["options"] == [""]
    assert SCHEMA["qdrant_api_key"]["secret"]
    for key, field in Settings.model_fields.items():
        bounds = {
            attr: getattr(m, attr)
            for m in field.metadata
            for attr in ("ge", "le")
            if hasattr(m, attr)
        }
        if bounds:
            assert SCHEMA[key]["slider"]["min"] == bounds["ge"]
            assert SCHEMA[key]["slider"]["max"] == bounds["le"]


async def test_one_time_migration_preserves_old_choices_and_key(store):
    memory, _ = await seed(store)
    old = await store.call("get_settings")
    config = NativeConfig(qdrant_api_key="synthetic-test-key")
    await store.call("bind_config", config)
    assert config["config_version"] == 1
    assert config["qdrant_api_key"] == "synthetic-test-key"
    assert (await store.call("get_settings")) == old
    assert all(config.persisted[k] == v for k, v in old.model_dump().items())
    assert (await store.call("detail", memory))["id"] == memory
    # A later native reset-to-default must not resurrect SQLite's old values.
    config.save_config({"mode": "off", "provider_id": ""})
    await store.call("bind_config", config)
    assert (await store.call("get_settings")).mode == "off"
    assert (await store.call("get_settings")).provider_id == ""


async def test_native_non_default_choices_win_first_migration(store):
    config = NativeConfig(provider_id="native-choice")
    await store.call("bind_config", config)
    assert (await store.call("get_settings")).provider_id == "native-choice"
    assert config["allowed_scopes"] == [SCOPE, "default:GroupMessage:200"]


async def test_empty_install_uses_native_defaults(tmp_path):
    from eyewitness.store import Store

    store = Store(tmp_path / "new.sqlite3")
    await store.call("open")
    try:
        config = NativeConfig()
        await store.call("bind_config", config)
        assert (await store.call("get_settings")) == Settings()
        assert config.persisted["config_version"] == 1
    finally:
        await store.call("close")


async def noop(*args):
    return "{}"


async def test_panel_and_native_use_same_config_without_exposing_key(store):
    config = NativeConfig(qdrant_api_key="synthetic-secret")
    await store.call("bind_config", config)
    api = AdminAPI(store, Engine(store, noop))
    doc, status = await api.handle({"path": "settings"})
    assert status == 200 and "synthetic-secret" not in json.dumps(doc)
    doc["provider_id"] = "chosen-in-panel"
    doc["qdrant_api_key"] = "synthetic-new-secret"
    saved, status = await api.handle({"path": "settings", "method": "PUT", "body": doc})
    assert status == 200 and "qdrant_api_key" not in saved
    assert config.persisted["provider_id"] == "chosen-in-panel"
    assert config.persisted["qdrant_api_key"] == "synthetic-new-secret"
    # Native save triggers plugin reload in AstrBot; bind the new config snapshot.
    config2 = NativeConfig(**{**config.persisted, "provider_id": "chosen-in-native"})
    await store.call("bind_config", config2)
    read, status = await api.handle({"path": "settings"})
    assert status == 200 and read["provider_id"] == "chosen-in-native"
    assert read["revision"] != saved["revision"]
    # An already-open panel cannot overwrite newer native edits.
    assert (await api.handle({"path": "settings", "method": "PUT", "body": saved}))[1] == 400
    assert config2["provider_id"] == "chosen-in-native"
    # Leaving the key out preserves it; an explicit empty string clears it.
    assert (await api.handle({"path": "settings", "method": "PUT", "body": read}))[1] == 200
    assert config2["qdrant_api_key"] == "synthetic-new-secret"
    assert (
        await api.handle(
            {"path": "settings", "method": "PUT", "body": {**read, "qdrant_api_key": ""}}
        )
    )[1] == 200
    assert config2.persisted["qdrant_api_key"] == ""


async def test_native_index_change_on_reload_queues_reindex(store):
    memory, _ = await seed(store)
    config = NativeConfig()
    await store.call("bind_config", config)
    store.db.execute("DELETE FROM outbox")
    store.db.commit()
    config.save_config({"embedding_provider_id": "new-embedding"})
    await store.call("bind_config", config)
    assert store.db.execute("SELECT memory_id FROM outbox").fetchone()[0] == memory


async def test_failed_native_save_leaves_runtime_and_database_unchanged(store):
    config = NativeConfig()
    await store.call("bind_config", config)
    old = await store.call("get_settings")
    snapshot = store.db.execute("SELECT data FROM settings").fetchone()[0]
    config.fail = True
    changed = old.model_copy(update={"mode": "off"})
    with pytest.raises(OSError):
        await store.call("save_settings", changed)
    assert await store.call("get_settings") == old
    assert store.db.execute("SELECT data FROM settings").fetchone()[0] == snapshot
