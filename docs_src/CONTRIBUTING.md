# Contributing to NexusMind

We welcome contributions to NexusMind! Please follow these general guidelines when contributing:

## Getting Started

*   **Fork the Repository:** Start by forking the main NexusMind repository to your own GitHub account.
*   **Clone your fork:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/NexusMind.git
    cd NexusMind
    ```
*   **Install Dependencies:** Ensure you have [Poetry](https://python-poetry.org/) (version 1.2+) installed and run:
    ```bash
    poetry install --with dev
    ```
    This will create a virtual environment and install all necessary dependencies.

## Branching Strategy

1.  **Branch from `main`:** Always create your new feature or bugfix branches from the latest `main` branch of the upstream (original) NexusMind repository.
    ```bash
    # Ensure your main branch is up-to-date with upstream
    git checkout main
    git pull upstream main # Assuming 'upstream' remote is configured to git@github.com:CognitiveAreNa/NexusMind.git
    
    # Create your new branch
    git checkout -b feature/your-descriptive-feature-name
    ```
    Or for bug fixes:
    ```bash
    git checkout -b fix/issue-123-brief-description
    ```
2.  **Naming Conventions:**
    *   Feature branches: `feature/some-feature-description`
    *   Bugfix branches: `fix/issue-number-short-description` (e.g., `fix/123-mcp-auth-error`)
    *   Documentation branches: `docs/update-readme-styles`
    *   Refactor branches: `refactor/improve-stage-loading`

## Development Process

1.  **Make Changes:** Implement your feature or bug fix in your created branch.
2.  **Write Tests:** 
    *   Add appropriate tests for your changes. 
    *   Ensure all existing and new tests pass by running:
        ```bash
        poetry run pytest
        ```
    *   **Test Coverage:** We aim for a high test coverage of over 95%. Please ensure your contributions include relevant tests to maintain or increase this coverage.
3.  **Check Code Quality:** Run linters and type checkers:
    ```bash
    poetry run ruff check .
    poetry run ruff format . # Or use an IDE plugin for on-the-fly formatting
    poetry run mypy src/
    ```
4.  **Commit Your Changes:** 
    *   Follow [Conventional Commit](https://www.conventionalcommits.org/) guidelines if possible (e.g., `feat: add user authentication`, `fix: resolve issue with API response`). This is not strictly enforced but highly encouraged.
    *   Write clear and concise commit messages.
5.  **Push to Your Fork:**
    ```bash
    git push origin feature/your-descriptive-feature-name
    ```
6.  **Submit a Pull Request:** 
    *   Open a pull request from your feature branch in your fork to the `main` branch of the `CognitiveAreNa/NexusMind` repository.
    *   Provide a clear title and a detailed description of your changes in the pull request. Link any relevant issues.
    *   Ensure all automated checks (CI/CD pipeline) pass.

## Code Style

*   **PEP 8:** Follow PEP 8 style guidelines. `ruff` helps enforce this.
*   **Type Hints:** Use type hints for all function and method signatures, and for complex variable types. `mypy` is used for static type checking.
*   **Docstrings:** Write comprehensive docstrings for all modules, classes, functions, and methods using a standard format (e.g., Google style, reStructuredText). Explain what the code does, its arguments, and what it returns.
*   **Imports:** Organize imports according to PEP 8 (standard library, then third-party, then local application/library specific imports, each group separated by a blank line). `ruff` can help sort these.

## Release Strategy

The project aims to follow Semantic Versioning (SemVer - `MAJOR.MINOR.PATCH`). Releases are managed by the core maintainers. Branch protection rules are in place for the `main` branch to ensure stability.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on the GitHub repository.

Thank you for contributing!
