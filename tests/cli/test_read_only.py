"""Tests for process-only read-only agent permissions."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from omnimancer.cli.agent_loop import AgentLoopMixin
from omnimancer.core.agent.types import Operation, OperationResult, OperationType
from omnimancer.core.agent_engine import AgentEngine
from omnimancer.core.config_manager import ConfigManager
from omnimancer.core.models import ProviderConfig


def _operation(operation_type: OperationType) -> Operation:
    data = {"path": "example.txt"}
    if operation_type == OperationType.COMMAND_EXECUTE:
        data = {"command": "echo test"}
    return Operation(
        type=operation_type,
        description="test operation",
        data=data,
        requires_approval=False,
    )


class MarkerHarness(AgentLoopMixin):
    """Minimal marker parser host for read-only permission tests."""

    def __init__(self, agent_engine: AgentEngine) -> None:
        self.engine = SimpleNamespace(agent_engine=agent_engine)
        self.errors: list[str] = []

    def _show_error(self, message: str) -> None:
        self.errors.append(message)


def _read_only_engine(config_manager: ConfigManager) -> AgentEngine:
    engine = AgentEngine.__new__(AgentEngine)
    engine.config_manager = config_manager
    engine.operation_history = []
    engine._generate_preview = AsyncMock(return_value="preview")
    engine._execute_operation = AsyncMock(
        return_value=OperationResult(success=True, data="read contents")
    )
    engine.set_read_only(True)
    return engine


async def test_read_only_denies_mutation_and_execution_without_config_write(
    tmp_path,
) -> None:
    config_path = tmp_path / "config.json"
    config_manager = ConfigManager(str(config_path))
    config = config_manager.get_config()
    config.default_provider = "test"
    config.providers = {"test": ProviderConfig(api_key="k", model="m")}
    config.storage_path = str(tmp_path)
    config_manager.save_config(config)
    before = config_path.read_bytes()

    engine = _read_only_engine(config_manager)

    for operation_type in (
        OperationType.FILE_WRITE,
        OperationType.FILE_DELETE,
        OperationType.DIRECTORY_CREATE,
        OperationType.DIRECTORY_DELETE,
        OperationType.COMMAND_EXECUTE,
    ):
        result = await engine.execute_with_approval(_operation(operation_type))
        assert result.success is False
        assert "permission rule" in (result.error or "")

    read_result = await engine.execute_with_approval(
        _operation(OperationType.FILE_READ)
    )
    assert read_result.success is True
    assert config_path.read_bytes() == before


async def test_marker_write_and_execution_are_denied(tmp_path) -> None:
    config_manager = ConfigManager(str(tmp_path / "config.json"))
    config_manager.get_config()
    engine = _read_only_engine(config_manager)
    harness = MarkerHarness(engine)
    target = tmp_path / "owned.txt"
    response = (
        f"[FILE_WRITE:{target}]owned[/FILE_WRITE]\n"
        f"[SAFE_EXEC]echo owned > {target}[/SAFE_EXEC]\n"
        f"[COMMAND_EXEC]echo owned > {target}[/COMMAND_EXEC]"
    )

    rendered = await harness._parse_and_execute_operations(response)

    assert not target.exists()
    assert rendered.count("permission rule") == 3
    engine._execute_operation.assert_not_awaited()


async def test_marker_find_and_search_use_command_permission_gate(tmp_path) -> None:
    config_manager = ConfigManager(str(tmp_path / "config.json"))
    config_manager.get_config()
    engine = _read_only_engine(config_manager)
    harness = MarkerHarness(engine)

    rendered = await harness._parse_and_execute_operations(
        "[FIND:*.py]\n[SEARCH:needle]"
    )

    assert "Find failed" in rendered
    assert "Search failed" in rendered
    assert rendered.count("permission rule") == 2
    engine._execute_operation.assert_not_awaited()


async def test_marker_file_read_remains_allowed(tmp_path) -> None:
    config_manager = ConfigManager(str(tmp_path / "config.json"))
    config_manager.get_config()
    engine = _read_only_engine(config_manager)
    harness = MarkerHarness(engine)
    source = tmp_path / "source.txt"
    source.write_text("source")

    rendered = await harness._parse_and_execute_operations(f"[FILE_READ:{source}]")

    assert "read contents" in rendered
    engine._execute_operation.assert_awaited_once()
