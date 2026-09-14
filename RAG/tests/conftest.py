"""
conftest.py — pytest auto-loads this before collecting any tests in this
directory. Its only job here: put RAG/ (this file's parent directory) on
sys.path so `from hybrid_search import ...` resolves, regardless of which
directory `pytest` was invoked from.

Why this is needed: pytest's default import mode adds the directory
CONTAINING the test file (tests/, since it has no __init__.py) to sys.path —
not the directory the test file needs to import FROM (RAG/, one level up).
Without this, `import hybrid_search` fails with ModuleNotFoundError even
though hybrid_search.py is right there in the project.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
