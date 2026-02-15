"""
Enhanced metrics re-exported from pipeline2.py
This module provides backward compatibility and easy imports
"""

import sys
from pathlib import Path

# Import from pipeline2
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipeline2 import EnhancedVoiceMetrics, AlignmentCounts

__all__ = ['EnhancedVoiceMetrics', 'AlignmentCounts']
