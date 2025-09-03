# Changelog

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