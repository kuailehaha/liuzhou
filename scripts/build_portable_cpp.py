#!/usr/bin/env python3
"""Build the CPU-only portable MCTS extension into the ignored build tree."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
from pathlib import Path

import torch
from torch.utils import cpp_extension
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_LIB = PROJECT_ROOT / "build" / "portable_cpp"
DEFAULT_BUILD_TEMP = PROJECT_ROOT / "build" / "portable_cpp_temp"


def _compile_args() -> list[str]:
    if os.name == "nt":
        return ["/O2", "/std:c++17", "/EHsc", "/bigobj"]
    args = ["-O3", "-std=c++17", "-fvisibility=hidden"]
    if platform.system() == "Darwin":
        args.extend(["-stdlib=libc++"])
    return args


def build_extension(*, build_lib: Path, build_temp: Path, force: bool) -> Path:
    include_paths = cpp_extension.include_paths()
    extension = Extension(
        "_liuzhou_portable_cpp",
        sources=[
            str(PROJECT_ROOT / "v1" / "cpp" / "portable_mcts.cpp"),
            str(PROJECT_ROOT / "v0" / "src" / "game" / "game_state.cpp"),
            str(PROJECT_ROOT / "v0" / "src" / "rules" / "rule_engine.cpp"),
            str(PROJECT_ROOT / "v0" / "src" / "moves" / "move_generator.cpp"),
        ],
        include_dirs=[
            str(PROJECT_ROOT / "v0" / "include"),
            *include_paths,
        ],
        language="c++",
        extra_compile_args=_compile_args(),
    )
    distribution = Distribution(
        {
            "name": "liuzhou-portable-cpp",
            "ext_modules": [extension],
        }
    )
    command = build_ext(distribution)
    command.build_lib = str(build_lib)
    command.build_temp = str(build_temp)
    command.force = bool(force)
    command.ensure_finalized()
    command.run()
    output = Path(command.get_ext_fullpath("_liuzhou_portable_cpp")).resolve()
    if not output.is_file():
        raise RuntimeError(f"build reported success but extension is missing: {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-lib", type=Path, default=DEFAULT_BUILD_LIB)
    parser.add_argument("--build-temp", type=Path, default=DEFAULT_BUILD_TEMP)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove only the portable C++ build directories before compiling.",
    )
    args = parser.parse_args()
    build_lib = args.build_lib.resolve()
    build_temp = args.build_temp.resolve()
    if args.clean:
        for target in (build_lib, build_temp):
            if target in (PROJECT_ROOT, PROJECT_ROOT.parent) or PROJECT_ROOT not in target.parents:
                raise RuntimeError(f"refusing to clean broad path: {target}")
            shutil.rmtree(target, ignore_errors=True)
    build_lib.mkdir(parents=True, exist_ok=True)
    build_temp.mkdir(parents=True, exist_ok=True)
    output = build_extension(
        build_lib=build_lib,
        build_temp=build_temp,
        force=bool(args.force),
    )
    print(f"portable_cpp_extension={output}")
    print(f"torch={torch.__version__}")
    print(f"cuda_required=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
