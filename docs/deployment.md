# Deployment Guide

This document explains the automated deployment pipeline for Omnimancer CLI.

## Overview

The project uses GitHub Actions for automated CI/CD with the following workflows:

- **CI/CD Pipeline** (`ci-cd.yml`) - Main testing, building, and publishing workflow
- **Auto Release** (`auto-release.yml`) - Automatic version bumping and release creation
- **Manual Release** (`release.yml`) - Manual release creation workflow
- **Code Quality** (`code-quality.yml`) - Additional code quality checks

## Automated Deployment Process

### 1. Continuous Integration

On every push to `main` or `develop` branches and pull requests:

1. **Testing** - Runs tests across Python 3.8-3.12
2. **Code Quality** - Black, isort, flake8, mypy checks
3. **Security** - Bandit and safety checks
4. **Build** - Creates wheel and source distributions
5. **Installation Test** - Tests package installation across platforms

### 2. Test PyPI Publishing

On pushes to `main` branch:
- Automatically publishes to Test PyPI for validation
- Uses `TEST_PYPI_API_TOKEN` secret

### 3. Production Release

On GitHub release publication:
- Publishes to PyPI using `PYPI_API_TOKEN` secret
- Uploads build artifacts to GitHub release

## Release Methods

### Automatic Releases

The `auto-release.yml` workflow automatically creates releases based on commit messages:

- **Major version bump**: Commits containing `BREAKING`, `breaking`, `feat!`, `fix!`, or `major:`
- **Minor version bump**: Commits containing `feat:`, `feature:`, or `minor:`
- **Patch version bump**: Commits containing `fix:`, `bugfix:`, or `patch:`

### Manual Releases

Use the GitHub Actions "Create Release" workflow:

1. Go to Actions → Create Release
2. Enter version (e.g., `v1.2.3`)
3. Choose if it's a pre-release or draft
4. The workflow will:
   - Update version in `pyproject.toml`
   - Update `CHANGELOG.md`
   - Create git tag
   - Create GitHub release

### Version Management Script

Use the `scripts/version_manager.py` script for local version management:

```bash
# Get current version
python3 scripts/version_manager.py current

# Bump version
python3 scripts/version_manager.py bump --type patch --update-changelog
python3 scripts/version_manager.py bump --type minor --update-changelog
python3 scripts/version_manager.py bump --type major --update-changelog

# Set specific version
python3 scripts/version_manager.py set --version 1.2.3 --update-changelog

# Generate changelog only
python3 scripts/version_manager.py changelog --update-changelog
```

## Required Secrets

Configure these secrets in your GitHub repository:

- `PYPI_API_TOKEN` - PyPI API token for production publishing
- `TEST_PYPI_API_TOKEN` - Test PyPI API token for testing
- `GITHUB_TOKEN` - Automatically provided by GitHub Actions

## Environment Protection

The workflows use GitHub environments for additional security:

- `pypi` environment - Protects production PyPI publishing
- `test-pypi` environment - Protects test PyPI publishing

## Changelog Management

The project maintains a `CHANGELOG.md` file following [Keep a Changelog](https://keepachangelog.com/) format:

- Automatically updated by version management script
- Categorizes changes as Features, Bug Fixes, and Other Changes
- Includes commit hashes and dates

## Best Practices

### Commit Message Format

Use conventional commit format for automatic release detection:

```
feat: add new provider support
fix: resolve authentication issue
docs: update README
chore: bump dependencies
BREAKING: remove deprecated API
```

### Release Process

1. **Development**: Work on feature branches
2. **Testing**: Create PR to `develop` branch
3. **Integration**: Merge to `main` branch
4. **Automatic Release**: Let auto-release workflow handle versioning
5. **Manual Release**: Use manual workflow for specific version control

### Version Strategy

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Troubleshooting

### Failed Deployments

1. Check GitHub Actions logs
2. Verify secrets are configured
3. Ensure PyPI tokens have correct permissions
4. Check package name availability on PyPI

### Version Conflicts

1. Ensure version in `pyproject.toml` is unique
2. Check existing tags: `git tag --list`
3. Use version manager script for consistency

### Build Issues

1. Verify `pyproject.toml` configuration
2. Test build locally: `python3 -m build`
3. Check dependencies in `requirements.txt`

## Monitoring

- **GitHub Actions**: Monitor workflow runs in the Actions tab
- **PyPI**: Check package status at https://pypi.org/project/omnimancer-cli/
- **Test PyPI**: Validate releases at https://test.pypi.org/project/omnimancer-cli/