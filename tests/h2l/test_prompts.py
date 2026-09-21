"""H2L prompt contracts (PRD-h2l §3–§5, NFR-6)."""

from omnimancer.h2l import prompts


class TestPlannerPrompt:
    def test_states_worker_constraints(self):
        text = prompts.PLANNER_SYSTEM
        assert "cannot see the repository" in text
        assert "will not explore" in text
        assert "absolute path" in text.lower()
        assert "old_string" in text and "new_string" in text
        assert "submit_plan" in text
        assert "depends_on" in text
        assert "verify" in text
        # Stories go one per call so no output cap can truncate the plan.
        assert "add_story" in text
        assert "once per story" in text


class TestWorkerPrompt:
    def test_forbids_exploration_and_handles_edit_mismatch(self):
        text = prompts.WORKER_SYSTEM
        assert "exactly" in text
        assert "do not explore" in text.lower()
        assert "old_string" in text
        assert "stop" in text.lower()
        assert "verify" in text.lower()


class TestJudgePrompt:
    def test_grades_diff_only(self):
        text = prompts.JUDGE_SYSTEM
        assert "acceptance criteria" in text
        assert "diff" in text
        assert "do not reward length" in text.lower()
        assert "submit_verdict" in text


class TestPins:
    """sha256 pins: a prompt edit must be deliberate (mirrors PromptFoundry)."""

    def test_pins_match(self):
        for name, expected in prompts.PROMPT_SHA256.items():
            assert prompts.sha256_of(getattr(prompts, name)) == expected, name
