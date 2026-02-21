"""
Report formatting utilities
Re-exported from pipeline2.py
"""

import sys
from pathlib import Path

# Import from pipeline2
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipeline2 import format_report

__all__ = ['format_report']
