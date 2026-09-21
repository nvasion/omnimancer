"""H2L data models: refs, config, plan validation (PRD-h2l §1, §2)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from omnimancer.core.models import Config, ProviderConfig
from omnimancer.h2l.models import (
    H2LConfig,
    H2LModelRef,
    Plan,
    PlanValidationError,
    Story,
    parse_model_ref,
    validate_plan,
)


class TestModelRef:
    def test_two_part_form(self):
        ref = parse_model_ref("gateway:qwen3-8b", default_provider="claude")
        assert ref == H2LModelRef(provider="gateway", model="qwen3-8b")

    def test_bare_model_uses_default_provider(self):
        ref = parse_model_ref("claude-opus-5", default_provider="claude")
        assert ref == H2LModelRef(provider="claude", model="claude-opus-5")

    def test_model_id_may_contain_colons(self):
        # Only the first colon separates the entry from the model id.
        ref = parse_model_ref("ollama-oai:qwen2.5-coder:14b", default_provider="x")
        assert ref.provider == "ollama-oai"
        assert ref.model == "qwen2.5-coder:14b"

    @pytest.mark.parametrize("text", ["", "   ", ":model", "gateway:", ":"])
    def test_malformed_refs_rejected(self, text):
        with pytest.raises(ValueError):
            parse_model_ref(text, default_provider="claude")

    def test_label_round_trips(self):
        ref = H2LModelRef(provider="gateway", model="qwen3-8b")
        assert ref.label == "gateway:qwen3-8b"
        assert parse_model_ref(ref.label, default_provider="z") == ref


class TestH2LConfig:
    def _cfg(self, **overrides):
        base = dict(
            high=H2LModelRef(provider="claude", model="claude-opus-5"),
            low=[H2LModelRef(provider="gateway", model="qwen3-8b")],
        )
        base.update(overrides)
        return H2LConfig(**base)

    def test_defaults_match_prd(self):
        cfg = self._cfg()
        assert cfg.enabled is True
        assert cfg.max_parallel == 2
        assert cfg.max_retries == 2
        assert cfg.pass_threshold == 80
        assert cfg.worker_tools == ["Edit", "Write", "Bash"]
        assert cfg.allow_worker_read is True
        assert cfg.escalation == "ask"
        assert cfg.plan_approval is True
        assert cfg.worker_max_iterations == 15
        assert cfg.planner_max_iterations is None  # unbounded unless opted in
        assert cfg.high_max_tokens == 16384
        assert cfg.low_max_tokens == 8192

    def test_low_pool_must_not_be_empty(self):
        with pytest.raises(ValidationError):
            self._cfg(low=[])

    @pytest.mark.parametrize("threshold", [-1, 101])
    def test_threshold_range(self, threshold):
        with pytest.raises(ValidationError):
            self._cfg(pass_threshold=threshold)

    def test_max_parallel_at_least_one(self):
        with pytest.raises(ValidationError):
            self._cfg(max_parallel=0)

    def test_refs_accept_string_form(self):
        # Config files write refs as "entry:model" strings.
        cfg = H2LConfig(high="claude:claude-opus-5", low=["gateway:qwen3-8b"])
        assert cfg.high.provider == "claude"
        assert cfg.low[0].model == "qwen3-8b"

    def test_escalation_choices(self):
        with pytest.raises(ValidationError):
            self._cfg(escalation="retry-forever")


class TestConfigIntegration:
    def _config(self, **kw):
        return Config(
            default_provider="claude",
            providers={"claude": ProviderConfig(api_key="k", model="m")},
            storage_path="/tmp/omn-h2l-test",
            **kw,
        )

    def test_h2l_block_is_optional_and_off_by_default(self):
        assert self._config().h2l is None

    def test_h2l_block_parses_from_dict(self):
        cfg = self._config(
            h2l={"high": "claude:claude-opus-5", "low": ["gateway:qwen3-8b"]}
        )
        assert isinstance(cfg.h2l, H2LConfig)
        assert cfg.h2l.high.label == "claude:claude-opus-5"

    def test_h2l_block_survives_v1_migration(self):
        from omnimancer.core.config_migration import ConfigMigration

        migration = ConfigMigration(config_path="/tmp/omn-h2l-migration.json")
        old = {
            "default_provider": "claude",
            "providers": {"claude": {"api_key": "k", "model": "m"}},
            "h2l": {"high": "claude:x", "low": ["gateway:y"]},
        }
        new = migration._migrate_from_v1(old)
        assert new["h2l"] == old["h2l"]


def _story(id, files, depends_on=None, **kw):
    return Story(
        id=id,
        title=id,
        instructions="do it",
        files=files,
        acceptance=["it is done"],
        depends_on=depends_on or [],
        **kw,
    )


class TestValidatePlan:
    def test_valid_plan_passes_through(self, tmp_path: Path):
        plan = Plan(
            goal="g",
            design="d",
            stories=[_story("S1", [str(tmp_path / "a.py")])],
        )
        assert validate_plan(plan, tmp_path).stories[0].id == "S1"

    def test_duplicate_ids_rejected(self, tmp_path: Path):
        plan = Plan(
            goal="g",
            design="d",
            stories=[
                _story("S1", [str(tmp_path / "a.py")]),
                _story("S1", [str(tmp_path / "b.py")]),
            ],
        )
        with pytest.raises(PlanValidationError, match="S1"):
            validate_plan(plan, tmp_path)

    def test_unresolved_dependency_rejected(self, tmp_path: Path):
        plan = Plan(
            goal="g",
            design="d",
            stories=[_story("S1", [str(tmp_path / "a.py")], depends_on=["S9"])],
        )
        with pytest.raises(PlanValidationError, match="S9"):
            validate_plan(plan, tmp_path)

    def test_cycle_rejected(self, tmp_path: Path):
        plan = Plan(
            goal="g",
            design="d",
            stories=[
                _story("S1", [str(tmp_path / "a.py")], depends_on=["S2"]),
                _story("S2", [str(tmp_path / "b.py")], depends_on=["S1"]),
            ],
        )
        with pytest.raises(PlanValidationError, match="cycle"):
            validate_plan(plan, tmp_path)

    def test_relative_path_rejected(self, tmp_path: Path):
        plan = Plan(goal="g", design="d", stories=[_story("S1", ["a.py"])])
        with pytest.raises(PlanValidationError, match="absolute"):
            validate_plan(plan, tmp_path)

    def test_path_outside_cwd_rejected(self, tmp_path: Path):
        plan = Plan(goal="g", design="d", stories=[_story("S1", ["/etc/hosts"])])
        with pytest.raises(PlanValidationError, match="outside the working directory"):
            validate_plan(plan, tmp_path)

    def test_shared_file_gets_implicit_edge_in_plan_order(self, tmp_path: Path):
        shared = str(tmp_path / "shared.py")
        plan = Plan(
            goal="g",
            design="d",
            stories=[
                _story("S1", [shared]),
                _story("S2", [str(tmp_path / "other.py")]),
                _story("S3", [shared]),
            ],
        )
        validated = validate_plan(plan, tmp_path)
        by_id = {s.id: s for s in validated.stories}
        assert by_id["S3"].depends_on == ["S1"]
        assert by_id["S1"].depends_on == []
        assert by_id["S2"].depends_on == []

    def test_declared_edge_not_duplicated(self, tmp_path: Path):
        shared = str(tmp_path / "shared.py")
        plan = Plan(
            goal="g",
            design="d",
            stories=[_story("S1", [shared]), _story("S2", [shared], ["S1"])],
        )
        validated = validate_plan(plan, tmp_path)
        assert validated.stories[1].depends_on == ["S1"]

    def test_declared_reverse_order_on_shared_file_is_not_a_cycle(self, tmp_path):
        # Live failure: S1 (earlier in plan order) declares it depends on S2,
        # and both touch the same file. The declared edge already serializes
        # them; a blind implicit edge S2 -> S1 used to close a cycle and
        # reject a perfectly good plan.
        shared = str(tmp_path / "models.py")
        plan = Plan(
            goal="g",
            design="d",
            stories=[_story("S1", [shared], ["S2"]), _story("S2", [shared])],
        )
        validated = validate_plan(plan, tmp_path)
        by_id = {s.id: s for s in validated.stories}
        assert by_id["S1"].depends_on == ["S2"]
        assert by_id["S2"].depends_on == []

    def test_transitively_ordered_stories_get_no_extra_edge(self, tmp_path):
        shared = str(tmp_path / "engine.py")
        plan = Plan(
            goal="g",
            design="d",
            stories=[
                _story("S1", [shared]),
                _story("S2", [str(tmp_path / "x.py")], ["S1"]),
                _story("S3", [shared], ["S2"]),
            ],
        )
        validated = validate_plan(plan, tmp_path)
        assert validated.stories[2].depends_on == ["S2"]  # S1 reached via S2

    def test_all_problems_reported_at_once(self, tmp_path):
        # Each resubmission costs the planner a full plan generation, so the
        # rejection must list every problem, not just the first.
        plan = Plan(
            goal="g",
            design="d",
            stories=[
                _story("S1", ["relative.py"], ["S9"]),
                _story("S2", ["/etc/hosts"]),
            ],
        )
        with pytest.raises(PlanValidationError) as exc:
            validate_plan(plan, tmp_path)
        message = str(exc.value)
        assert "not an absolute path" in message
        assert "unknown story S9" in message
        assert "outside the working directory" in message

    def test_empty_plan_rejected(self, tmp_path: Path):
        with pytest.raises(PlanValidationError, match="no stories"):
            validate_plan(Plan(goal="g", design="d", stories=[]), tmp_path)

    def test_paths_are_normalized(self, tmp_path: Path):
        messy = str(tmp_path / "sub" / ".." / "a.py")
        plan = Plan(goal="g", design="d", stories=[_story("S1", [messy])])
        validated = validate_plan(plan, tmp_path)
        assert validated.stories[0].files == [str(tmp_path / "a.py")]
