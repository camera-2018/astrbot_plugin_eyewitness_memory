import ast
import json
from pathlib import Path

import pytest

from eyewitness.models import Settings

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = "astrbot_plugin_eyewitness_memory"


def test_runtime_identity_is_consistent():
    tree = ast.parse((ROOT / "main.py").read_text())
    identity = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PLUGIN" for target in node.targets)
    )
    assert identity == PLUGIN
    assert f"name: {PLUGIN}\n" in (ROOT / "metadata.yaml").read_text()
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "EyewitnessMemoryPlugin"
        for node in tree.body
    )
    assert 'name = "astrbot-plugin-eyewitness-memory"' in (ROOT / "pyproject.toml").read_text()
    assert (
        json.loads((ROOT / "frontend/package.json").read_text())["name"]
        == "eyewitness-memory-console"
    )


def test_collection_namespace_is_new_and_isolated():
    assert Settings().collection == "eyewitness_memory_v1"
    assert Settings(collection="eyewitness_memory_custom").collection == "eyewitness_memory_custom"
    for name in ("evidence_memory_v1", "repeat_memory", "eyewitness_memory_"):
        with pytest.raises(ValueError):
            Settings(collection=name)
