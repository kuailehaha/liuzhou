"""Explicit loader for the repository-local portable C++ build artifact."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_DIR = PROJECT_ROOT / "build" / "portable_cpp"
MODULE_NAME = "_liuzhou_portable_cpp"


def load_portable_cpp(*, required: bool = True) -> Optional[ModuleType]:
    """Load the opt-in C++ core without changing the Python fallback path."""

    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as first_error:
        build_dir = str(DEFAULT_BUILD_DIR)
        if build_dir not in sys.path:
            sys.path.insert(0, build_dir)
        try:
            return importlib.import_module(MODULE_NAME)
        except ModuleNotFoundError as second_error:
            if not required:
                return None
            raise RuntimeError(
                "Portable C++ backend was explicitly requested but is not built. "
                "Run the project environment's Python with "
                "`scripts/build_portable_cpp.py --force`."
            ) from second_error
        except ImportError:
            raise
    except ImportError:
        raise
