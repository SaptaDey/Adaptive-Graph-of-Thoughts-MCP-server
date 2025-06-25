"""Convenience wrapper to expose configuration utilities.

This thin module simply re-exports everything from
``adaptive_graph_of_thoughts.config`` so that older imports like
``import config`` continue to work in tests and examples.
"""

from pathlib import Path
import sys

import os

project_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.exists(project_src) and project_src not in sys.path:
    sys.path.insert(0, project_src)

from adaptive_graph_of_thoughts.config import *  # noqa: F401,F403
