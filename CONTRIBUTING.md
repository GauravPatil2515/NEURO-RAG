# Contributing to NeuroRAG

Thank you for your interest in contributing to NeuroRAG! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version)
- Error messages or screenshots

### Suggesting Features

Feature requests are welcome! Please include:
- Clear description of the feature
- Use cases and benefits
- Potential implementation approach

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Test thoroughly**
   ```bash
   python -m tests.test_complete
   ```
5. **Commit with clear messages**
   ```bash
   git commit -m "Add: brief description of changes"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**

## 📝 Code Style

### Python

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and concise

### Example

```python
def calculate_similarity(query: str, documents: list) -> list:
    """
    Calculate similarity scores between query and documents.
    
    Args:
        query: User search query
        documents: List of document chunks
        
    Returns:
        List of similarity scores
    """
    # Implementation here
    pass
```

## 🧪 Testing

All contributions should include tests:

```python
def test_new_feature():
    """Test description"""
    # Arrange
    input_data = "test"
    
    # Act
    result = new_feature(input_data)
    
    # Assert
    assert result == expected_output
```

## 📁 Project Structure

When adding new files, place them in the appropriate directory:

- `src/` - Core application code
- `tests/` - Test files
- `docs/` - Documentation
- `scripts/` - Utility scripts
- `templates/` - HTML templates
- `static/` - CSS, JavaScript, images

## 🔧 Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python -m tests.test_complete`
4. Start server: `python run_server.py`

## 📋 Commit Message Guidelines

Use clear, descriptive commit messages:

- `Add:` New features
- `Fix:` Bug fixes
- `Update:` Changes to existing features
- `Refactor:` Code restructuring
- `Docs:` Documentation changes
- `Test:` Test additions or changes

## 🌟 Areas for Contribution

- Performance optimization
- Additional test coverage
- Documentation improvements
- New features (see Issues)
- Bug fixes
- UI/UX enhancements

## ❓ Questions?

Feel free to open an issue for questions or discussions.

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to NeuroRAG! 🧠✨
