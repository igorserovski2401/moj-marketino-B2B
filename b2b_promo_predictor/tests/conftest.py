"""Pytest configuration: ensures src/ is importable from the tests/ directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
