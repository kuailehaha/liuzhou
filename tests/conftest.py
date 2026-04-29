"""Shared pytest configuration — sets up DLL search paths for v0_core."""
import os
import sys

# Try common locations for the torch library directory (needed for v0_core DLL loading)
_torch_lib_candidates = []
_torch_home = os.environ.get("TORCH_LIB_PATH", "")
if _torch_home:
    _torch_lib_candidates.append(_torch_home)
# Common conda env patterns
_conda_root = os.environ.get("CONDA_PREFIX", "")
if _conda_root:
    _torch_lib_candidates.append(os.path.join(_conda_root, "Lib", "site-packages", "torch", "lib"))
# Walk sys.path for existing torch installations
for _p in list(sys.path):
    _cand = os.path.join(_p, "torch", "lib")
    if os.path.isdir(_cand):
        _torch_lib_candidates.append(_cand)

for _lib_dir in _torch_lib_candidates:
    if os.path.isdir(_lib_dir):
        try:
            os.add_dll_directory(_lib_dir)
        except (OSError, AttributeError):
            pass

# Add build/v0/src to sys.path if present
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_build_dir = os.path.join(_project_root, "build", "v0", "src")
if os.path.isdir(_build_dir) and _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)
