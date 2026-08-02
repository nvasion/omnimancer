"""Migration must not eat config blocks it doesn't know about.

Found 2026-07-31: `_migrate_from_v1` rebuilds the config from a fixed
dict, silently dropping newer optional blocks (`enhancement`,
`custom_models`, `fallback`, `permissions`, `hooks`, `subagents`) from
any v1-shaped file. The bug was masked while Config defaulted an
enhancement block with the same values; it surfaced when the field
became opt-in (None).
"""

import json

import pytest

from omnimancer.core.config_migration import ConfigMigration

V1_CONFIG = {
    "default_provider": "gateway",
    "storage_path": "/tmp/omni-migration-test",
    "providers": {
        "gateway": {
            "model": "qwen3-coder-30b",
            "provider_type": "openai-compatible",
            "base_url": "http://alpha:8888/v1",
            "auth_type": "none",
        }
    },
    "enhancement": {
        "provider": "gateway",
        "model": "qwen3-8b",
        "temperature": 0.4,
        "default_profile": "code",
    },
    "custom_models": [
        {
            "name": "qwen3-coder-30b",
            "provider": "gateway",
            "context_window": 262144,
            "max_tokens": 262144,
            "supports_tools": True,
        }
    ],
    "fallback": {"fallback_order": ["gateway"], "auto_fallback": False},
    "events": {"enabled": False, "max_file_mb": 5},
}


@pytest.fixture
def migrated(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(V1_CONFIG))
    migration = ConfigMigration(config_path)
    assert migration.needs_migration()
    ok, messages = migration.migrate_config()
    assert ok, messages
    return json.loads(config_path.read_text())


class TestMigrationPassthrough:
    def test_enhancement_block_survives(self, migrated):
        assert migrated.get("enhancement", {}).get("model") == "qwen3-8b"

    def test_custom_models_survive(self, migrated):
        names = [m.get("name") for m in migrated.get("custom_models", [])]
        assert names == ["qwen3-coder-30b"]

    def test_fallback_block_survives(self, migrated):
        assert migrated.get("fallback", {}).get("fallback_order") == ["gateway"]

    def test_events_block_survives(self, migrated):
        # An explicit opt-out must not be silently reverted to the
        # default-on EventsConfig by a migration.
        assert migrated.get("events", {}).get("enabled") is False
        assert migrated.get("events", {}).get("max_file_mb") == 5


class TestProviderMigrationPreservesUserValues:
    """Provider defaults may only fill gaps — `update(provider_defaults)`
    clobbered explicit user values (provider_type 'openai-compatible' became
    the self-referential entry name; auth_type 'none' became 'api_key')."""

    def test_provider_type_survives(self, migrated):
        provider = migrated["providers"]["gateway"]
        assert provider.get("provider_type") == "openai-compatible"

    def test_auth_type_none_survives(self, migrated):
        provider = migrated["providers"]["gateway"]
        assert provider.get("auth_type") == "none"
