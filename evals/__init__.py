"""Evaluation framework for FinSight Financial Report Analyzer."""

import sys
from pathlib import Path

# Ensure 'src' and repo root are accessible on python path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

