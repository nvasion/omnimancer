"""Runtime provenance is derived from loaded source, never the caller's cwd."""

import subprocess

import pytest


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def repository(root):
    root.mkdir()
    package = root / "omnimancer"
    (package / "decisions").mkdir(parents=True)
    (package / "cli").mkdir()
    (package / "__init__.py").write_text("")
    (package / "decisions" / "routing.py").write_text('MODEL = "one"\n')
    (package / "decisions" / "integration.py").write_text("def gate(): return True\n")
    (package / "decisions" / "replay.py").write_text("LIMIT = 8\n")
    (package / "cli" / "headless.py").write_text("ENABLED = False\n")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Synthetic Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "synthetic fixture",
    )
    return package


@pytest.mark.parametrize(
    "file",
    [
        "decisions/routing.py",
        "decisions/integration.py",
        "decisions/replay.py",
        "cli/headless.py",
    ],
)
def test_digest_tracks_validators_constants_and_runtime_logic(tmp_path, file):
    from omnimancer.decisions.provenance import implementation_digest

    package = repository(tmp_path / "repo")
    before = implementation_digest(package)
    p = package / file
    p.write_text(p.read_text() + "CHANGED = True\n")
    assert implementation_digest(package) != before


def test_commit_and_dirty_state_belong_to_loaded_package(tmp_path, monkeypatch):
    from omnimancer.decisions.provenance import source_provenance

    package = repository(tmp_path / "source")
    other = repository(tmp_path / "caller")
    git(
        other.parent,
        "-c",
        "user.name=Synthetic Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "different commit",
    )
    monkeypatch.chdir(other.parent)
    expected = git(package.parent, "rev-parse", "HEAD")
    assert source_provenance(package) == {
        "source_commit": expected,
        "source_dirty": False,
    }
    (package / "decisions" / "routing.py").write_text("CHANGED = True\n")
    assert source_provenance(package) == {
        "source_commit": expected,
        "source_dirty": True,
    }


def test_untracked_installed_package_does_not_claim_surrounding_git_head(tmp_path):
    from omnimancer.decisions.provenance import source_provenance

    tracked = repository(tmp_path / "repo")
    installed = tracked.parent / "venv" / "omnimancer"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("")
    assert source_provenance(installed) == {"source_commit": None, "source_dirty": None}


@pytest.mark.asyncio
async def test_evaluation_manifest_uses_loaded_package_provenance(
    tmp_path, monkeypatch
):
    from omnimancer.decisions import provenance
    from omnimancer.decisions.evaluation import Case, evaluate
    from omnimancer.decisions.routing import RoutingPolicy

    package = repository(tmp_path / "loaded")
    other = repository(tmp_path / "caller")
    monkeypatch.chdir(other.parent)
    monkeypatch.setattr(provenance, "_package", lambda: package)
    policy = RoutingPolicy(
        targets={
            "fast": {"provider": "fast", "model": "small", "criteria": "simple"},
            "deep": {"provider": "deep", "model": "large", "criteria": "complex"},
        }
    )
    case = Case(
        id="one", split="test", category="simple", prompt="Fix label", expected="fast"
    )
    report = await evaluate([case], policy, repetitions=1)
    assert report.manifest.source_commit == git(package.parent, "rev-parse", "HEAD")
    assert report.manifest.source_dirty is False
    assert report.manifest.implementation_sha256 == provenance.implementation_digest()
    assert report.manifest.policy_sha256 == provenance.policy_digest(policy)
