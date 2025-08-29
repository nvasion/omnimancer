#!/usr/bin/env python3
"""
Version management script for Omnimancer CLI.
Handles version bumping and changelog generation.
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def get_current_version() -> str:
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")

    content = pyproject_path.read_text()
    match = re.search(r'version = "([^"]+)"', content)
    if not match:
        raise ValueError("Version not found in pyproject.toml")

    return match.group(1)


def update_version(new_version: str) -> None:
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    # Update version
    updated_content = re.sub(
        r'version = "[^"]+"', f'version = "{new_version}"', content
    )

    pyproject_path.write_text(updated_content)
    print(f"Updated version to {new_version} in pyproject.toml")


def bump_version(current_version: str, bump_type: str) -> str:
    """Bump version based on type (major, minor, patch)."""
    parts = current_version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {current_version}")

    major, minor, patch = map(int, parts)

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")

    return f"{major}.{minor}.{patch}"


def get_git_commits_since_tag(tag: Optional[str] = None) -> List[str]:
    """Get git commits since the last tag or all commits if no tag."""
    try:
        if tag:
            cmd = ["git", "log", f"{tag}..HEAD", "--oneline", "--no-merges"]
        else:
            # Get all commits if no previous tag
            cmd = ["git", "log", "--oneline", "--no-merges"]

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
        commits = result.stdout.strip().split("\n")
        return [commit for commit in commits if commit.strip()]
    except subprocess.CalledProcessError:
        return []


def get_last_tag() -> Optional[str]:
    """Get the last git tag."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def generate_changelog(version: str) -> str:
    """Generate changelog for the new version."""
    last_tag = get_last_tag()
    commits = get_git_commits_since_tag(last_tag)

    changelog = f"# Changelog\n\n## [{version}] - {datetime.now().strftime('%Y-%m-%d')}\n\n"

    if commits:
        # Categorize commits
        features = []
        fixes = []
        other = []

        for commit in commits:
            commit_msg = commit.split(" ", 1)[1] if " " in commit else commit
            if commit_msg.lower().startswith(("feat:", "feature:")):
                features.append(commit_msg)
            elif commit_msg.lower().startswith(("fix:", "bugfix:")):
                fixes.append(commit_msg)
            else:
                other.append(commit_msg)

        if features:
            changelog += "### Features\n"
            for feature in features:
                changelog += f"- {feature}\n"
            changelog += "\n"

        if fixes:
            changelog += "### Bug Fixes\n"
            for fix in fixes:
                changelog += f"- {fix}\n"
            changelog += "\n"

        if other:
            changelog += "### Other Changes\n"
            for change in other:
                changelog += f"- {change}\n"
            changelog += "\n"
    else:
        changelog += "- Initial release\n\n"

    return changelog


def update_changelog_file(new_changelog: str) -> None:
    """Update CHANGELOG.md file."""
    changelog_path = Path("CHANGELOG.md")

    if changelog_path.exists():
        existing_content = changelog_path.read_text()
        # Insert new changelog after the first line (title)
        lines = existing_content.split("\n")
        if lines and lines[0].startswith("# "):
            updated_content = (
                lines[0] + "\n\n" + new_changelog + "\n".join(lines[1:])
            )
        else:
            updated_content = new_changelog + existing_content
    else:
        updated_content = new_changelog

    changelog_path.write_text(updated_content)
    print(f"Updated CHANGELOG.md")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Manage Omnimancer CLI versions"
    )
    parser.add_argument(
        "action",
        choices=["bump", "set", "current", "changelog"],
        help="Action to perform",
    )
    parser.add_argument(
        "--type",
        choices=["major", "minor", "patch"],
        help="Type of version bump (for bump action)",
    )
    parser.add_argument(
        "--version", help="Specific version to set (for set action)"
    )
    parser.add_argument(
        "--update-changelog",
        action="store_true",
        help="Update CHANGELOG.md file",
    )

    args = parser.parse_args()

    try:
        if args.action == "current":
            print(get_current_version())

        elif args.action == "bump":
            if not args.type:
                print("Error: --type is required for bump action")
                sys.exit(1)

            current = get_current_version()
            new_version = bump_version(current, args.type)
            update_version(new_version)

            if args.update_changelog:
                changelog = generate_changelog(new_version)
                update_changelog_file(changelog)

            print(f"Bumped version from {current} to {new_version}")

        elif args.action == "set":
            if not args.version:
                print("Error: --version is required for set action")
                sys.exit(1)

            current = get_current_version()
            update_version(args.version)

            if args.update_changelog:
                changelog = generate_changelog(args.version)
                update_changelog_file(changelog)

            print(f"Set version from {current} to {args.version}")

        elif args.action == "changelog":
            current = get_current_version()
            changelog = generate_changelog(current)
            if args.update_changelog:
                update_changelog_file(changelog)
            else:
                print(changelog)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
