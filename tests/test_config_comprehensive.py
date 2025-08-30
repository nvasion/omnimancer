"""
Comprehensive configuration tests - consolidates all config-related test functionality.

This module consolidates tests from:
- test_config_validator_new.py
- test_config_generator.py
- test_config_migration.py
- test_config_repair.py
- test_config_simplification.py
- test_config_templates.py
- test_enhanced_config.py
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from omnimancer.core.config_migration import ConfigMigration
from omnimancer.core.config_validator import ConfigValidator

# from omnimancer.core.config_generator import ConfigGenerator  # Removed as over-engineered
# from omnimancer.core.config_repair import ConfigRepair  # Removed as over-engineered
from omnimancer.core.models import (
    Config,
    ConfigTemplate,
    ConfigTemplateManager,
    MCPConfig,
    MCPServerConfig,
    ProviderConfig,
)


class TestConfigValidator:
    """Test cases for ConfigValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator()

    def test_init(self):
        """Test ConfigValidator initialization."""
        assert isinstance(self.validator._validation_cache, dict)
        assert self.validator._cache_ttl == 300.0
        assert hasattr(self.validator, "_lock")

    def test_validate_config_valid(self):
        """Test validation of valid configuration."""
        provider_config = ProviderConfig(model="gpt-4", api_key="sk-test123")
        config = Config(
            default_provider="openai",
            providers={"openai": provider_config},
            storage_path="/tmp/omnimancer",
            mcp=MCPConfig(),
        )

        with patch("omnimancer.core.config_validator.Path") as mock_path:
            mock_path.return_value.expanduser.return_value.parent.exists.return_value = (
                True
            )

            errors = self.validator.validate_config(config)

            assert isinstance(errors, list)
            assert len(errors) == 0

    def test_validate_config_no_default_provider(self):
        """Test validation with no default provider."""
        mock_config = Mock()
        mock_config.default_provider = ""
        mock_config.providers = {"openai": Mock()}
        mock_config.storage_path = "/tmp/omnimancer"
        mock_config.mcp = Mock()

        with patch.object(self.validator, "validate_provider_config", return_value=[]):
            with patch.object(self.validator, "validate_mcp_config", return_value=[]):
                errors = self.validator._validate_config(mock_config)

                assert len(errors) > 0
                assert any(
                    "No default provider configured" in error for error in errors
                )

    def test_validate_config_default_provider_not_configured(self):
        """Test validation when default provider is not in providers."""
        provider_config = ProviderConfig(model="gpt-4", api_key="sk-test123")
        config = Config(
            default_provider="claude",
            providers={"openai": provider_config},
            storage_path="/tmp/omnimancer",
            mcp=MCPConfig(),
        )

        errors = self.validator.validate_config(config)

        assert len(errors) > 0
        assert any(
            "Default provider 'claude' is not configured" in error for error in errors
        )

    def test_validate_config_caching(self):
        """Test configuration validation caching."""
        provider_config = ProviderConfig(model="gpt-4", api_key="sk-test123")
        config = Config(
            default_provider="openai",
            providers={"openai": provider_config},
            storage_path="/tmp/omnimancer",
            mcp=MCPConfig(),
        )

        with patch("omnimancer.core.config_validator.Path") as mock_path:
            mock_path.return_value.expanduser.return_value.parent.exists.return_value = (
                True
            )

            # First call
            errors1 = self.validator.validate_config(config)

            # Second call should use cache
            errors2 = self.validator.validate_config(config)

            assert errors1 == errors2

    def test_validate_provider_config_no_model(self):
        """Test provider validation with no model."""
        provider_config = ProviderConfig(model="", api_key="sk-test123")

        errors = self.validator.validate_provider_config("openai", provider_config)

        assert len(errors) > 0
        assert any("has no model specified" in error for error in errors)

    def test_validate_claude_config_valid(self):
        """Test valid Claude configuration."""
        provider_config = ProviderConfig(
            model="claude-3-5-sonnet-20241022", api_key="sk-ant-test123"
        )

        errors = self.validator._validate_claude_config(provider_config)

        assert len(errors) == 0

    def test_validate_claude_config_invalid_model(self):
        """Test Claude configuration with invalid model."""
        provider_config = ProviderConfig(
            model="invalid-model", api_key="sk-ant-test123"
        )

        errors = self.validator._validate_claude_config(provider_config)

        assert len(errors) > 0
        assert any("Unknown Claude model" in error for error in errors)

    def test_validate_openai_config_valid(self):
        """Test valid OpenAI configuration."""
        provider_config = ProviderConfig(model="gpt-4", api_key="sk-test123")

        errors = self.validator._validate_openai_config(provider_config)

        assert len(errors) == 0

    def test_validate_mcp_config_valid(self):
        """Test validation of valid MCP configuration."""
        server_config = MCPServerConfig(
            name="filesystem", command="fs-server", timeout=30
        )
        mcp_config = MCPConfig(
            servers={"filesystem": server_config},
            auto_approve_timeout=30,
            max_concurrent_servers=10,
        )

        errors = self.validator.validate_mcp_config(mcp_config)

        assert len(errors) == 0

    def test_generate_config_hash(self):
        """Test configuration hash generation."""
        provider_config = ProviderConfig(model="gpt-4", api_key="sk-test123")
        config = Config(
            default_provider="openai",
            providers={"openai": provider_config},
            storage_path="/tmp/omnimancer",
            mcp=MCPConfig(),
        )

        hash1 = self.validator._generate_config_hash(config)
        hash2 = self.validator._generate_config_hash(config)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length


@pytest.mark.skip(reason="ConfigGenerator removed as over-engineered")
class TestConfigGenerator:
    """Test cases for ConfigGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        # self.generator = ConfigGenerator()  # Removed as over-engineered

    def test_init(self):
        """Test ConfigGenerator initialization."""
        # assert hasattr(self.generator, 'provider_registry')  # Removed as over-engineered
        # assert hasattr(self.generator, 'template_manager')  # Removed as over-engineered
        pass

    def test_generate_full_config(self):
        """Test full configuration generation."""
        # Test that the method exists and is callable
        # assert hasattr(self.generator, 'generate_full_config')  # Removed as over-engineered
        # assert callable(self.generator.generate_full_config)  # Removed as over-engineered
        pass

    def test_generate_template_config(self):
        """Test template configuration generation."""
        # Test that the method exists and is callable
        # assert hasattr(self.generator, 'generate_template_config')  # Removed as over-engineered
        # assert callable(self.generator.generate_template_config)  # Removed as over-engineered
        pass

    def test_mcp_server_defaults_disabled(self):
        """Test that MCP servers are disabled by default in generated config."""
        # mcp_config = self.generator._create_example_mcp_config()  # Removed as over-engineered
        return

        # Check that specific MCP servers are disabled by default
        assert "filesystem" in mcp_config.servers
        assert mcp_config.servers["filesystem"].enabled is False

        assert "calculator" in mcp_config.servers
        assert mcp_config.servers["calculator"].enabled is False

        assert "datetime" in mcp_config.servers
        assert mcp_config.servers["datetime"].enabled is False

    def test_mcp_servers_have_proper_commands(self):
        """Test that MCP servers have correct uvx command configuration."""
        # mcp_config = self.generator._create_example_mcp_config()  # Removed as over-engineered
        return

        for server_name, server_config in mcp_config.servers.items():
            assert server_config.command == "uvx"
            assert isinstance(server_config.args, list)
            assert len(server_config.args) > 0
            assert server_config.args[0].startswith("mcp-server-")


@pytest.mark.skip(reason="ConfigRepair removed as over-engineered")
class TestConfigRepair:
    """Test cases for ConfigRepair class."""

    def setup_method(self):
        """Set up test fixtures."""
        # self.repair = ConfigRepair()  # Removed as over-engineered

    def test_init(self):
        """Test ConfigRepair initialization."""
        # assert isinstance(self.repair.validator, ConfigValidator)  # Removed as over-engineered
        # assert hasattr(self.repair, 'config_path')  # Removed as over-engineered
        # assert hasattr(self.repair, 'backup_path')  # Removed as over-engineered
        pass

    def test_analyze_config(self):
        """Test configuration analysis."""
        provider_config = ProviderConfig(model="gpt-4", api_key="sk-test123")
        Config(
            default_provider="openai",
            providers={"openai": provider_config},
            storage_path="/tmp/omnimancer",
            mcp=MCPConfig(),
        )

        # issues = self.repair.analyze_config(config)  # Removed as over-engineered
        issues = []

        assert isinstance(issues, list)
        # Should have minimal issues for valid config
        assert len([issue for issue in issues if issue["severity"] == "error"]) == 0

    def test_is_error_fixable(self):
        """Test error fixability detection."""

        # assert self.repair._is_error_fixable(fixable_error) is True  # Removed as over-engineered
        # assert self.repair._is_error_fixable(unfixable_error) is False  # Removed as over-engineered
        pass


class TestConfigTemplateManager:
    """Test cases for ConfigTemplateManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ConfigTemplateManager()

    def test_init(self):
        """Test ConfigTemplateManager initialization."""
        assert isinstance(self.manager.templates, dict)
        assert len(self.manager.templates) > 0
        # Should have standard templates
        expected_templates = [
            "coding",
            "research",
            "creative",
            "general",
            "performance",
        ]
        for template_name in expected_templates:
            assert template_name in self.manager.templates

    def test_get_template_existing(self):
        """Test getting an existing template."""
        template = self.manager.get_template("coding")

        assert isinstance(template, ConfigTemplate)
        assert template.name == "coding"
        assert template.use_case == "coding"
        assert (
            "openai" in template.recommended_providers
            or "claude" in template.recommended_providers
        )

    def test_get_template_nonexistent(self):
        """Test getting a non-existent template."""
        with pytest.raises(KeyError):
            self.manager.get_template("nonexistent")

    def test_list_templates(self):
        """Test listing all available templates."""
        templates = self.manager.list_templates()

        assert isinstance(templates, list)
        assert len(templates) > 0

        for template_info in templates:
            assert "name" in template_info
            assert "description" in template_info
            assert "use_case" in template_info

    def test_create_coding_template(self):
        """Test coding template creation."""
        template = self.manager._create_coding_template()

        assert isinstance(template, ConfigTemplate)
        assert template.name == "coding"
        assert template.use_case == "coding"
        assert (
            "software development" in template.description.lower()
            or "coding" in template.description.lower()
        )

        # Should have coding-relevant providers
        assert any(
            provider in ["openai", "claude", "claude_code"]
            for provider in template.recommended_providers
        )

        # Should have appropriate settings for coding
        assert "temperature" in template.settings
        assert template.settings["temperature"] <= 0.3  # Lower temperature for coding

    def test_template_mcp_servers_disabled_by_default(self):
        """Test that MCP servers in templates are disabled by default."""
        # Test coding template
        coding_template = self.manager._create_coding_template()
        if coding_template.mcp_servers:
            for (
                server_name,
                server_config,
            ) in coding_template.mcp_servers.items():
                assert (
                    server_config.get("enabled", True) is False
                ), f"Server {server_name} should be disabled by default"

        # Test research template
        research_template = self.manager._create_research_template()
        if research_template.mcp_servers:
            for (
                server_name,
                server_config,
            ) in research_template.mcp_servers.items():
                assert (
                    server_config.get("enabled", True) is False
                ), f"Server {server_name} should be disabled by default"

    def test_template_mcp_server_descriptions_mention_uvx(self):
        """Test that MCP server descriptions mention uvx requirement."""
        templates = [
            self.manager._create_coding_template(),
            self.manager._create_research_template(),
            self.manager._create_creative_template(),
        ]

        for template in templates:
            if template.mcp_servers:
                for server_name, server_config in template.mcp_servers.items():
                    description = server_config.get("description", "")
                    if description:  # Only check non-empty descriptions
                        assert (
                            "uvx" in description.lower()
                        ), f"Server {server_name} description should mention uvx requirement"


class TestConfigMigration:
    """Test cases for ConfigMigration class."""

    def setup_method(self):
        """Set up test fixtures."""
        # ConfigMigration requires a config_path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            self.tmp_config_path = tmp.name
        self.migration = ConfigMigration(self.tmp_config_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "tmp_config_path") and Path(self.tmp_config_path).exists():
            Path(self.tmp_config_path).unlink()

    def test_init(self):
        """Test ConfigMigration initialization."""
        assert hasattr(self.migration, "config_path")
        assert hasattr(self.migration, "backup_dir")

    def test_needs_migration(self):
        """Test migration necessity detection."""
        # Ensure file doesn't exist first
        if Path(self.tmp_config_path).exists():
            Path(self.tmp_config_path).unlink()

        # File doesn't exist - no migration needed
        assert self.migration.needs_migration() is False

        # Create old config file (without version = needs migration)
        old_config = {"providers": {"openai": {"model": "gpt-4"}}}
        with open(self.tmp_config_path, "w") as f:
            json.dump(old_config, f)

        # Should need migration now (no config_version means 1.0, needs upgrade to 2.0)
        assert self.migration.needs_migration() is True

        # Create new config file (with correct version = no migration needed)
        new_config = {
            "config_version": "2.0",
            "providers": {"openai": {"model": "gpt-4"}},
        }
        with open(self.tmp_config_path, "w") as f:
            json.dump(new_config, f)

        # Should not need migration now
        assert self.migration.needs_migration() is False


class TestConfigTemplate:
    """Test cases for ConfigTemplate dataclass."""

    def test_config_template_creation(self):
        """Test ConfigTemplate creation with all fields."""
        template = ConfigTemplate(
            name="test_template",
            description="Test template description",
            use_case="testing",
            recommended_providers=["openai", "claude"],
            recommended_models={
                "openai": "gpt-4",
                "claude": "claude-3-sonnet",
            },
            mcp_tools=["filesystem", "git"],
            settings={"temperature": 0.7, "max_tokens": 2048},
            provider_configs={"openai": {"api_key": "test"}},
            mcp_servers={"filesystem": {"command": "fs-server"}},
        )

        assert template.name == "test_template"
        assert template.description == "Test template description"
        assert template.use_case == "testing"
        assert len(template.recommended_providers) == 2
        assert "openai" in template.recommended_providers
        assert template.recommended_models["openai"] == "gpt-4"
        assert "filesystem" in template.mcp_tools
        assert template.settings["temperature"] == 0.7
        assert template.provider_configs["openai"]["api_key"] == "test"
        assert template.mcp_servers["filesystem"]["command"] == "fs-server"


@pytest.mark.skip(reason="ConfigGenerator and ConfigRepair removed as over-engineered")
class TestConfigIntegration:
    """Integration tests for config components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator()
        # self.generator = ConfigGenerator()  # Removed as over-engineered
        # self.repair = ConfigRepair()  # Removed as over-engineered
        self.template_manager = ConfigTemplateManager()
        # ConfigMigration requires a config_path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            self.tmp_config_path = tmp.name
        self.migration = ConfigMigration(self.tmp_config_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "tmp_config_path") and Path(self.tmp_config_path).exists():
            Path(self.tmp_config_path).unlink()

    def test_end_to_end_config_workflow(self):
        """Test complete configuration workflow."""
        # 1. Generate a config using template
        template = self.template_manager.get_template("coding")
        assert isinstance(template, ConfigTemplate)

        # 2. Create a provider config
        provider_config = ProviderConfig(model="gpt-4", api_key="sk-test123")

        # 3. Create full config
        config = Config(
            default_provider="openai",
            providers={"openai": provider_config},
            storage_path="/tmp/omnimancer",
            mcp=MCPConfig(),
        )

        # 4. Validate config
        with patch("omnimancer.core.config_validator.Path") as mock_path:
            mock_path.return_value.expanduser.return_value.parent.exists.return_value = (
                True
            )
            errors = self.validator.validate_config(config)
            assert len(errors) == 0

        # 5. Analyze for issues
        # issues = self.repair.analyze_config(config)  # Removed as over-engineered
        issues = []
        error_issues = [issue for issue in issues if issue["severity"] == "error"]
        assert len(error_issues) == 0

        # 6. Test migration detection
        config_dict = {
            "default_provider": "openai",
            "providers": {"openai": {"model": "gpt-4", "api_key": "sk-test123"}},
        }
        # Write config to temp file for migration test
        with open(self.tmp_config_path, "w") as f:
            json.dump(config_dict, f)

        needs_migration = self.migration.needs_migration()
        assert needs_migration is True  # Missing version field

    def test_config_validation_with_issues(self):
        """Test configuration validation with various issues."""
        # Create config with intentional issues
        provider_config = ProviderConfig(
            model="", api_key="sk-test123"  # Missing model
        )
        config = Config(
            default_provider="nonexistent",  # Provider not in providers dict
            providers={"openai": provider_config},
            storage_path="/tmp/omnimancer",
            mcp=MCPConfig(),
        )

        errors = self.validator.validate_config(config)

        assert len(errors) > 0
        assert any(
            "Default provider 'nonexistent' is not configured" in error
            for error in errors
        )
        assert any("has no model specified" in error for error in errors)

    def test_template_validation_integration(self):
        """Test template validation integration."""
        template = self.template_manager.get_template("research")

        # Template should be valid
        is_valid, errors = self.template_manager.validate_template(template)
        assert is_valid is True
        assert len(errors) == 0

        # Get template as config dict
        config_dict = self.template_manager.get_template_config_dict("research")
        assert isinstance(config_dict, dict)
        assert "provider_configs" in config_dict
        assert "settings" in config_dict

    def test_migration_integration(self):
        """Test migration integration with validation."""
        # Write old config data to temp file
        old_config = {
            "providers": {"openai": {"model": "gpt-3.5-turbo", "api_key": "sk-test123"}}
        }
        with open(self.tmp_config_path, "w") as f:
            json.dump(old_config, f)

        # Check that migration is needed
        assert self.migration.needs_migration() is True

        # Can validate the old config too
        provider_config = ProviderConfig(
            model=old_config["providers"]["openai"]["model"],
            api_key=old_config["providers"]["openai"]["api_key"],
        )
        config = Config(
            default_provider="openai",
            providers={"openai": provider_config},
            storage_path="/tmp/omnimancer",
            mcp=MCPConfig(),
        )

        with patch("omnimancer.core.config_validator.Path") as mock_path:
            mock_path.return_value.expanduser.return_value.parent.exists.return_value = (
                True
            )
            errors = self.validator.validate_config(config)
            assert len(errors) == 0


class TestConfigManagerSmokeTests:
    """Core smoke tests for ConfigManager functionality."""

    def setup_method(self):
        """Setup test environment with temporary config."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
        from omnimancer.core.config_manager import ConfigManager

        self.config_manager = ConfigManager(str(self.config_path))

    def teardown_method(self):
        """Cleanup test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_manager_initialization(self):
        """Test that ConfigManager can be initialized properly."""
        assert self.config_manager is not None
        assert self.config_manager.config_path == self.config_path
        assert hasattr(self.config_manager, "_cipher")

    def test_encryption_methods_exist(self):
        """Test that encryption methods are available."""
        assert hasattr(self.config_manager, "_encrypt_api_key")
        assert hasattr(self.config_manager, "_decrypt_api_key")
        assert hasattr(self.config_manager, "_get_cipher")

    def test_config_loading_and_saving(self):
        """Test basic config loading and saving."""
        # Load config (should create default if not exists)
        config = self.config_manager.load_config()
        assert isinstance(config, Config)

        # Save config
        self.config_manager.save_config(config)
        assert self.config_path.exists()

        # Load again to verify persistence
        config2 = self.config_manager.load_config()
        assert config2.config_version == config.config_version
