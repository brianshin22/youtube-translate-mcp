# Contributing to YouTube Translate MCP

Thank you for your interest in contributing to YouTube Translate MCP! This document provides guidelines and instructions for contributing.

## Setting Up Development Environment

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/your-username/youtube-translate-mcp.git
   cd youtube-translate-mcp
   ```

2. Create and activate a virtual environment:
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Alternatively, with standard tools
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   # Using uv (recommended)
   uv pip install -e .
   
   # Using pip
   pip install -e .
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test them thoroughly.

3. Run the cleanup script before committing to remove macOS resource fork files:
   ```bash
   ./cleanup.sh
   ```

4. Commit your changes with a clear message:
   ```bash
   git commit -m "Add feature: your feature description"
   ```

5. Push your branch to GitHub:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a pull request from your branch to the main repository.

## Testing Locally

To test your changes with the MCP server locally:

```bash
# Run the server directly
YOUTUBE_TRANSLATE_API_KEY=your_api_key uv run -m youtube_translate_mcp
```

### Testing with Claude Desktop

Claude Desktop provides a great way to test your MCP server. There are two methods:

1. **Local Development Testing**: Configure Claude Desktop to run your local development version using `uv`.

2. **Docker-based Testing**: Test using the Docker container for more reproducible results.

Refer to the "Testing with Claude Desktop" section in the README.md for detailed configuration instructions for both methods.

## Code Style

- Follow PEP 8 for Python code style.
- Use meaningful variable and function names.
- Include docstrings for all functions, classes, and modules.
- Add type hints to function signatures.

## Testing

- Add tests for new features or bug fixes.
- Make sure all tests pass before submitting a pull request.

## Documentation

- Update documentation when adding or changing features.
- Keep the README.md up to date.
- Add examples for new features.

## Pull Request Process

1. Update the README.md and documentation with details of changes if appropriate.
2. Update the version number in pyproject.toml according to semantic versioning.
3. Your pull request will be reviewed by maintainers, who may request changes.
4. Once approved, your pull request will be merged.

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT license. 