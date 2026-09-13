# Python Packaging Notes

## Virtual Environments

Common commands for managing a venv:

- `python -m venv .venv` — create a new virtual environment in `.venv`
- `.venv\Scripts\activate` — activate on Windows (PowerShell: `.venv\Scripts\Activate.ps1`)
- `source .venv/bin/activate` — activate on macOS/Linux
- `deactivate` — leave the current venv
- `pip freeze > requirements.txt` — snapshot installed packages

## Build Tools

A minimal `pyproject.toml` for a package that uses setuptools as the build backend:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-package"
version = "0.1.0"
dependencies = [
    "requests>=2.31",
]
```

Running `pip install -e .` from the project root installs it in editable mode, so code changes are picked up without reinstalling.

## Dependency Management

Comparison of common tools:

| Tool | Lockfile | Speed | Notes |
|------|----------|-------|-------|
| pip + requirements.txt | No (unless pip-tools) | Baseline | Simplest, no real dependency resolution |
| Poetry | Yes (poetry.lock) | Slower installs | Also handles packaging/publishing |
| uv | Yes (uv.lock) | Very fast | Newer, drop-in pip-compatible CLI |
