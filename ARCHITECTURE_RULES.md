# 📋 Architecture & Coding Rules - Streaming Video RAG Project

## ✅ Code Style & Formatting

### Python Formatting
- **Line length**: 120 characters maximum
- **Quotes**: Use double quotes (`"`) for all strings
- **Formatter**: Black (configured in pyproject.toml)
- **Linter**: Ruff with enabled rules: E, F, I, B, C4, SIM, UP, RUF, TID, TRY, PERF
- **Import Order**:
  1. Future imports
  2. Standard library
  3. Third party packages
  4. First party project modules (api, ingestion, llm, processing, rag, storage, transcription, ui, vector_store)
  5. Local folder imports

### Command To Format Code
```bash
black .
ruff check . --fix
```

---

## 🔢 Constants & Magic Values
### Constant Definition Policy
- **Define named constants** for any literal value that appears **more than once** in a single file, or appears across multiple files in the project
- **No magic numbers/strings**: Avoid hardcoding raw values directly in logic
- Place file-local constants at the top of the file, after imports
- Place cross-project constants in `config.py` or dedicated `constants.py` module
- Use `UPPER_SNAKE_CASE` for all constant names
- Add comment explaining constant purpose if not obvious

Examples:
```python
# ❌ Bad (repeated magic value)
if len(chunk) > 512:
    process_chunk(chunk)
# ... later in file
if len(text) > 512:
    split_text(text)

# ✅ Good (defined constant)
MAX_CHUNK_LENGTH = 512
if len(chunk) > MAX_CHUNK_LENGTH:
    process_chunk(chunk)
# ... later in file
if len(text) > MAX_CHUNK_LENGTH:
    split_text(text)
```

---

## 📐 Architecture & Project Structure

### Module Organization
- Each domain has its own dedicated directory
- Use abstract base classes for interface definitions (`base.py`)
- Implement factory pattern for provider selection
- Maintain separation of concerns:
  ```
  api/           - FastAPI REST endpoints
  ingestion/     - Video source ingestion handlers
  transcription/ - Audio transcription services
  processing/    - Text chunking & preprocessing
  llm/           - LLM provider abstractions
  vector_store/  - Vector database implementations
  rag/           - RAG logic (retrieval, QA, search, summarization)
  storage/       - Relational database layer
  ui/            - Streamlit web interface
    ├─ tabs/     - Individual tab render functions
    └─ utils.py  - UI shared utilities
  ```

### Streamlit UI Standards
- **Tab Structure**: Each application tab is implemented in separate file under `ui/tabs/`
- **Naming Convention**: Tab files follow `[name]_tab.py` pattern with `render_*_tab()` export function
- **Entry Point**: `ui/app.py` handles page config, sidebar, tab routing only
- **Shared Logic**: Common utilities, API calls, CSS in `ui/utils.py`
- **No business logic in UI layer**: All operations call backend API endpoints

### File Naming
- Use snake_case for all filenames
- Abstract base classes: `base.py`
- Factory implementations: `factory.py`
- Module entry points: `__init__.py` exports public API only

---

## 🧩 Code Patterns

### Type Hints
- Add type hints for all function parameters and return values
- Use `dict[str, Any]` for unstructured data
- Pydantic models for all API request/response objects
- Type ignores are acceptable only for external library interfaces

### Error Handling
- Use Loguru for all logging (no `print()` statements)
- Implement retry logic with Tenacity for external API calls
- Catch specific exceptions not bare `except:`
- Always log exceptions with stack traces

### Documentation
- Every module starts with docstring header
- All public functions/methods have docstrings
- Comment sections with ASCII separators: `# ── Section Name ───────────────────`
- Keep README.md and architecture_plan.md updated

---

## 🧪 Development Workflow

### Git & Commits
- Create feature branches from main
- Write clear, descriptive commit messages
- Run formatting and linting before every commit
- Pre-commit hooks are configured for the project

### Dependencies
- Pin major versions in pyproject.toml
- Separate production and dev dependencies
- Add comments when adding new dependencies explaining purpose

### Environment Configuration
- All configuration lives in `config.py` using Pydantic Settings
- No hardcoded secrets, API keys or environment specific values in code
- `.env.example` contains all required environment variables

---

## 🚀 Implementation Standards

### Performance
- Use async I/O where appropriate for IO bound operations
- Avoid blocking calls in API routes
- Implement proper batching for vector operations
- Monitor and log execution time for long running tasks

### Security
- Never log sensitive data or user content
- Validate all user input with Pydantic
- CORS configuration is explicit
- File uploads have size limits and type validation

### Maintainability
- Keep functions focused and single purpose
- DRY principle: extract common logic to utilities
- Avoid deep inheritance hierarchies
- Prefer composition over inheritance

---

## ✅ Validation Checks Before PR
1. Code is formatted with `black .`
2. No Ruff errors: `ruff check .`
3. Mypy passes without critical errors
4. All existing tests pass
5. Documentation updated
6. No debug print statements or commented out code
7. All repeated literal values are defined as named constants