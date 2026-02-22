"""Shared pytest fixtures for STT benchmark tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project root importable without installing the package.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
