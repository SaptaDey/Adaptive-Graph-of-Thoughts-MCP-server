# Changelog

All notable changes to the ASR-GoT Reimagined project will be documented in this file.

## [Unreleased]

### Changed
- Updated Docker base image to Python 3.13.3-slim-bookworm for improved performance and security

### Added
- Type hints for loguru in a utility module
- Script to add type ignore comments to logger calls
- Type stubs for loguru
- Mypy configuration file
- Pyright configuration file to control type checking behavior
- Script to fix indentation issues in Python files
- Scripts to add type annotations to stage_4_evidence.py
- Added missing type annotations (List[str], Dict[str, Any]) to various variables

### Fixed
- Docker configuration casing: `as` → `AS` for stages
- Syntax issues in `graph_elements.py`: indentation and newlines
- Type annotations in several classes and methods
- Unbound variable in `stage_4_evidence.py`
- EdgeMetadata usage in `stage_4_evidence.py`
- Missing imports in several modules
- CertaintyScore usage in `stage_4_evidence.py`
- Docker image security: updated to Alpine-based images with fewer vulnerabilities
- Added proper dependencies for Alpine Linux in Docker
- Indentation issues in multiple files
- Syntax errors in `stage_4_evidence.py`
- Added return value to abstract `execute` method

### Changed
- Docker base images to use more secure versions
- Updated Docker base images from Python 3.11.9 to Python 3.13.3
- Updated Python version references in configuration files
- Updated loguru imports with type ignore comments
- Removed unused imports

### Security
- Updated Docker images from Alpine to Bookworm/Slim-Bookworm to address critical and high vulnerabilities
- Updated Docker images from Python 3.11 to Python 3.13.3 to address remaining critical and high vulnerabilities
- Added proper dependency management for Debian-based Docker images
- Added security cleaning by removing /var/lib/apt/lists in Docker images
