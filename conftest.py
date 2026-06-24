"""Pytest bootstrap: make the repo root importable so `import src.*` works
regardless of where pytest is invoked from."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
