# Agent Guidelines for Multi-Agent RAG System

## Build/Lint/Test Commands

### Testing
- Run all tests: `pytest`
- Run specific test file: `pytest test_quick_query.py`
- Run tests with coverage: `pytest --cov=. --cov-report=html`
- Run single test function: `pytest test_quick_query.py::test_function_name -v`

### Linting & Code Quality
- No specific linter configured - use your preferred Python linter (flake8, pylint, ruff)
- Type checking: `mypy .` (if mypy installed)

### Build/Run Commands
- Install dependencies: `pip install -r requirements.txt`
- Run main application: `python main.py`
- Run web UI: `python web_ui.py`
- Index codebase: `python index_academicon_lite.py`

## Code Style Guidelines

### Imports
- Standard library imports first
- Third-party imports second
- Local imports last
- Use absolute imports within the project
- Group imports with blank lines between groups

### Naming Conventions
- Classes: PascalCase (e.g., `AcademiconAssistant`, `GraphAnalystAgent`)
- Functions/methods: snake_case (e.g., `retrieve()`, `setup_environment()`)
- Variables: snake_case (e.g., `db_path`, `top_k`)
- Constants: UPPER_CASE (e.g., `COLORS`, `RESET`)
- Private methods: leading underscore (e.g., `_parse_json_response()`)

### Type Hints
- Use type hints for all function parameters and return values
- Import from `typing` module: `List`, `Dict`, `Optional`, `Any`
- Use union types with `|` syntax (Python 3.10+) where appropriate

### Documentation
- Use Google-style docstrings for all classes and functions
- Include `Args:` and `Returns:` sections for functions
- Document complex logic with inline comments
- Keep docstrings concise but informative

### Error Handling
- Use specific exception types rather than bare `except:`
- Log errors appropriately using the structured logging system
- Use `tenacity` for retry logic when calling external APIs
- Validate inputs using Pydantic models where possible

### Code Structure
- Follow the existing agent pattern from `agents.py`
- Use configuration classes from `config/settings.py` for settings
- Implement proper resource cleanup in `__del__` methods if needed
- Keep functions focused on single responsibilities

### Security
- Never log sensitive information (API keys, credentials)
- Use environment variables for configuration
- Follow the exclusion patterns in `IndexingConfig` for sensitive files

### Performance
- Use GPU acceleration when available (automatically detected)
- Implement caching where appropriate
- Batch operations for embedding generation
- Profile code before optimizing