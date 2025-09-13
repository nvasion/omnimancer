# Changelog

# Changelog

## [0.1.9] - 2025-09-13

### Bug Fixes
- fix: PROPERLY fix version extraction for tag pipelines

### Other Changes
- debug: Add more logging and clean old artifacts before build


# Changelog

## [0.1.8] - 2025-09-13

### Features
- feat: Add comprehensive version verification to CI pipeline

### Bug Fixes
- fix: CRITICAL - Use correct CI variable for tag pipeline detection
- fix: Simplify CI version handling for tag releases


# Changelog

## [0.1.7] - 2025-09-13

### Bug Fixes
- fix: Ensure build:package uses VERSION from extract_version for tag pipelines


# Changelog

## [0.1.6] - 2025-09-13

### Bug Fixes
- fix: GitLab CI version synchronization for tag releases


# Changelog

## [0.1.5] - 2025-09-12

### Bug Fixes
- fix: GitLab CI pipeline to properly handle version releases


# Changelog

## [0.1.4] - 2025-09-12

### Bug Fixes
- fix: GitLab CI pipeline to correctly deploy version from tag


# Changelog

## [0.1.3] - 2025-09-07

### Other Changes
- Edit README.md - kick off pipeline
- fixing issue with openrouter provider and others not propering going through /setup
- Edit README.md
- fixing unittests broken by black, ruff, and isort
- fixing tool escaping in pyproject
- more linting
- More black linting and formating changes
- removing dependencies on 3.8-9
- removing python 3.8-9 dependencies
- remove tag
- updating the gitlab yaml
- linting and updating the ci/cd workflow
- linting and updating the ci/cd workflow
- Update .gitlab-ci.yml file
- Apply 1 suggestion(s) to 1 file(s)
- Apply 1 suggestion(s) to 1 file(s)
- Apply 1 suggestion(s) to 1 file(s)
- Apply 1 suggestion(s) to 1 file(s)
- adding gitlab ci


# Changelog

## [0.1.2] - 2025-09-03

### Other Changes
- Update .gitlab-ci.yml file


# Changelog

## [0.1.1] - 2025-08-30

### Other Changes
- Standardize code formatting with black and ci/cd pipeline
- Initial Omnimancer release commit


All notable changes to the Omnimancer CLI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive GitLab CI/CD pipeline with multi-Python version testing
- Automated versioning with `scripts/version_manager.py`
- Security scanning (SAST, dependency scanning, license scanning)
- Code quality checks (black, isort, flake8, mypy)
- PyPI deployment automation (test and production)
- GitLab Package Registry deployment
- Automated release creation with changelog generation
- Performance testing and benchmarking support
- Documentation generation with Sphinx

### Changed
- Code formatting applied across entire codebase with black
- Enhanced test coverage reporting with XML and HTML formats
- Improved caching strategy for faster CI/CD builds

### Security
- Added SAST scanning with bandit
- Added dependency vulnerability scanning with safety and pip-audit
- Added license compliance scanning

## [0.1.0] - 2024-08-29

### Added
- Initial release of Omnimancer CLI
- Unified interface for multiple AI language models
- Support for Claude, OpenAI, Cohere, Azure, and other providers
- MCP (Model Context Protocol) integration
- Agent system with approval workflows
- Configuration management and security features
- Comprehensive test suite
- Rich CLI interface with progress indicators