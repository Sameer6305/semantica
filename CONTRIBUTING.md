# Contributing to Semantica

Thank you for your interest in contributing to Semantica! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Commit Message Conventions](#commit-message-conventions)
- [Pull Request Process](#pull-request-process)
- [Documentation Standards](#documentation-standards)
- [Types of Contributions](#types-of-contributions)
- [Getting Help](#getting-help)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the maintainers.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/semantica.git
   cd semantica
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/Hawksight-AI/semantica.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher (3.9+ recommended)
- pip package manager
- Git

### Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install the project in editable mode with dev dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Verify Installation

```bash
python -c "import semantica; print(semantica.__version__)"
pytest --version
black --version
```

## Code Style Guidelines

We use several tools to maintain code quality and consistency:

### Formatting

- **Black**: Code formatting (line length: 88)
  ```bash
  black semantica/
  ```

- **isort**: Import sorting
  ```bash
  isort semantica/
  ```

### Linting

- **flake8**: Style guide enforcement
  ```bash
  flake8 semantica/
  ```

- **mypy**: Static type checking
  ```bash
  mypy semantica/
  ```

### Running All Checks

```bash
# Format code
black semantica/ tests/

# Sort imports
isort semantica/ tests/

# Lint
flake8 semantica/ tests/

# Type check
mypy semantica/
```

Or use pre-commit hooks (automatically runs on commit):
```bash
pre-commit run --all-files
```

## Testing Requirements

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=semantica --cov-report=html

# Run specific test file
pytest tests/test_specific.py

# Run with verbose output
pytest -v
```

### Test Coverage

- Minimum coverage: **80%**
- Critical modules: **90%+**
- Coverage reports are generated in `htmlcov/`

### Writing Tests

- Follow pytest conventions
- Use descriptive test names
- Include docstrings for complex tests
- Test both success and failure cases
- Use fixtures for common setup

Example:
```python
def test_entity_extraction():
    """Test basic entity extraction functionality."""
    from semantica.semantic_extract import NamedEntityRecognizer
    
    ner = NamedEntityRecognizer()
    entities = ner.extract("Apple Inc. was founded by Steve Jobs.")
    
    assert len(entities) > 0
    assert any(e.text == "Apple Inc." for e in entities)
```

## Commit Message Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes

### Examples

```
feat(kg): add temporal graph support

Add support for temporal knowledge graphs with version tracking
and time-based queries.

Closes #123
```

```
fix(parse): handle empty PDF files gracefully

Previously, empty PDF files would cause a crash. Now they return
an empty document with appropriate warnings.

Fixes #456
```

## Pull Request Process

### Before Submitting

1. **Update your fork**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

3. **Make your changes** and commit following our conventions

4. **Run all checks**:
   ```bash
   pytest
   black semantica/ tests/
   isort semantica/ tests/
   flake8 semantica/ tests/
   mypy semantica/
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] No merge conflicts
- [ ] PR description is clear and complete

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issues
Closes #123
Related to #456

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Updated existing tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
```

## Documentation Standards

### Code Documentation

- Use Google-style docstrings
- Include type hints
- Document all public functions and classes
- Include examples for complex functions

Example:
```python
def extract_entities(
    text: str,
    model: str = "transformer",
    confidence_threshold: float = 0.7
) -> List[Entity]:
    """Extract named entities from text.
    
    Args:
        text: Input text to process
        model: NER model to use (default: "transformer")
        confidence_threshold: Minimum confidence score (default: 0.7)
    
    Returns:
        List of extracted Entity objects
    
    Raises:
        ValueError: If text is empty or model is invalid
    
    Example:
        >>> ner = NamedEntityRecognizer()
        >>> entities = ner.extract("Apple Inc. was founded in 1976.")
        >>> len(entities)
        2
    """
    ...
```

### Documentation Files

- Update relevant documentation in `docs/`
- Add examples to cookbook if applicable
- Update API reference if adding new public APIs
- Keep README.md up to date

## Types of Contributions

### 💻 Code Contributions

- **Bug Fixes**: Resolving issues reported in the issue tracker.
- **New Features**: Implementing new capabilities (please discuss via an issue first!).
- **Refactoring**: Improving code structure and maintainability without changing behavior.
- **Algorithm Optimization**: Improving the efficiency of graph algorithms and vector search.

#### ⚡ Performance and Latency
We deeply value efficiency. Contributions that make Semantica faster and lighter are highly appreciated!

- **Latency Reduction**: Optimizing critical paths, reducing function call overhead, and improving response times for RAG pipelines.
- **Memory Optimization**: Reducing memory footprint during graph processing and vector operations.
- **Throughput**: Improving the number of operations processed per second (e.g., bulk ingestion, parallel queries).
- **Benchmarks**: Adding performance benchmarks to track regressions.
- **Async/Concurrency**: Improving asynchronous execution and concurrency handling.

### 📚 Documentation Contributions

- Fix typos and grammar
- Improve clarity
- Add examples
- Create tutorials
- Translate documentation

### Testing Contributions

- Add test coverage
- Improve test quality
- Add integration tests
- Performance benchmarks

### Other Contributions

- Answer questions in discussions
- Help with issues
- Review pull requests
- Share use cases
- Report bugs
- Suggest features

## Getting Help

### Communication Channels

- **GitHub Discussions**: General questions and discussions
- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time chat and community support

### Before Asking for Help

1. Check existing documentation
2. Search GitHub issues and discussions
3. Review code examples in cookbook
4. Check FAQ in documentation

### Asking Good Questions

- Provide context and environment details
- Include code examples
- Show what you've tried
- Include error messages and logs
- Be specific about what you need

## Recognition

Contributors are recognized in:
- [CONTRIBUTORS.md](CONTRIBUTORS.md)
- GitHub contributors page
- Release notes for significant contributions

Thank you for contributing to Semantica! 🎉

