"""Convenience wrapper to expose configuration utilities.

This thin module simply re-exports everything from
``adaptive_graph_of_thoughts.config`` so that older imports like
``import config`` continue to work in tests and examples.
"""

from pathlib import Path
import sys

# Ensure the ``src`` directory is on ``sys.path`` when running directly
project_src = Path(__file__).parent / "src"
if project_src.exists() and str(project_src) not in sys.path:
    sys.path.insert(0, str(project_src))

from adaptive_graph_of_thoughts.config import *  # noqa: F401,F403
