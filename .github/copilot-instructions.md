# Project Overview

`src_method` is a Python package, designed for quantum computing applications.

## General Instructions

- Always prioritize readability and clarity.
- For algorithm-related code, include explanations of the approach used.
- Write code with good maintainability practices, including comments on why certain design decisions were made.
- Handle edge cases and write clear exception handling.
- For libraries or external dependencies, mention their usage and purpose in comments.
- Use consistent naming conventions and follow language-specific best practices.
- Write concise, efficient, and idiomatic code that is also easily understandable.
- Use conventional commits (ideally with gitmojis) for commit messages.

## Python Coding Conventions

### Python Instructions

- Write code to support Python 3.12, 3.13, 3.14.
- Write clear and concise comments for each function.
- Ensure functions have descriptive names and include type hints.
- Provide docstrings following PEP 257 conventions.
- Use Google-style docstrings without type annotations (types handled by type hints).
- Use the `typing` module for type annotations (e.g., `list[str]`, `dict[str, int]`).
- Break down complex functions into smaller, more manageable functions.

### Code Style and Formatting

- Follow the PEP 8 style guide for Python.
- Maintain proper indentation (use 4 spaces for each level of indentation).
- Ensure lines do not exceed 79 characters.
- Place function and class docstrings immediately after the `def` or `class` keyword.
- Use blank lines to separate functions, classes, and code blocks where appropriate.

### Edge Cases and Testing

- Always include test cases for critical paths of the application.
- Account for common edge cases like empty inputs, invalid data types, and large datasets.
- Include comments for edge cases and the expected behavior in those cases.
- Write unit tests for functions and document them with docstrings explaining the test cases.
- Use `pytest` to run tests.

### Example of Proper Documentation

```python
def calculate_area(radius: float) -> float:
    """Calculate the area of a circle given the radius.

    Args:
        radius: The radius of the circle.

    Returns:
        The area of the circle, calculated as π * radius^2.
    """
    import math
    return math.pi * radius ** 2
```

## Development Commands

### Environment Setup

- Use `uv` for dependency management.
- Create the virtual environment with `uv sync --all-groups --all-extras`.
- Install `pre-commit` hooks to ensure code quality:
```bash
uv run pre-commit install --install-hooks
```

### Code Quality

- Use `uv run ruff check src/ tests/` for linting.
- Use `uv run ruff format src/ tests/` for formatting.
- Use `uv run pre-commit run --all-files` to ensure code quality.

### Testing

- Run the fast tests with `uv run pytest -m "not slow"`
- Run the slow tests with `uv run pytest -m "slow"`
- Run the whole test suite with `uv run pytest`
