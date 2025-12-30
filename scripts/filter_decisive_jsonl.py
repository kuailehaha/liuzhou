#!/usr/bin/env python3
"""Filter JSONL training samples to keep only decisive outcomes (value != 0)."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable, List, Tuple


def _expand_inputs(items: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for item in items:
        matches = glob.glob(item)
        if matches:
            for match in matches:
                path = Path(match)
                if path.is_dir():
                    files.extend(sorted(path.glob("*.jsonl")))
                elif path.is_file():
                    files.append(path)
            continue

        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"No match for input: {item}")

    seen = set()
    unique: List[Path] = []
    for path in files:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _output_name(src: Path, suffix: str) -> str:
    if not suffix:
        return src.name
    if src.suffix == ".jsonl":
        return f"{src.stem}{suffix}{src.suffix}"
    return f"{src.name}{suffix}"


def _filter_file(src: Path, dst: Path, min_abs_value: float) -> Tuple[int, int]:
    total = 0
    kept = 0
    with src.open("r", encoding="utf-8") as fh, dst.open("w", encoding="utf-8") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            if "value" in record:
                value = record["value"]
            elif "result" in record:
                value = record["result"]
            else:
                raise KeyError(f"Missing value/result in record from {src}")

            try:
                value_f = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid value in record from {src}: {value}") from exc

            if abs(value_f) <= min_abs_value:
                continue
            out.write(json.dumps(record, ensure_ascii=False))
            out.write("\n")
            kept += 1
    return total, kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter JSONL training samples to keep only decisive outcomes."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input JSONL files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write filtered JSONL files.",
    )
    parser.add_argument(
        "--suffix",
        default="_decisive",
        help="Suffix appended before .jsonl in output filenames (set '' to keep name).",
    )
    parser.add_argument(
        "--min-abs-value",
        type=float,
        default=1e-6,
        help="Keep samples with |value| greater than this threshold.",
    )
    args = parser.parse_args()

    input_files = _expand_inputs(args.inputs)
    if not input_files:
        raise SystemExit("No input JSONL files found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_all = 0
    kept_all = 0

    for src in input_files:
        dst_name = _output_name(src, args.suffix)
        dst = output_dir / dst_name
        if dst.resolve() == src.resolve():
            raise ValueError(
                f"Refusing to overwrite input file {src}; choose another output_dir or suffix."
            )
        total, kept = _filter_file(src, dst, args.min_abs_value)
        dropped = total - kept
        total_all += total
        kept_all += kept
        ratio = (kept / total * 100.0) if total else 0.0
        print(f"{src} -> {dst} | kept {kept}/{total} ({ratio:.2f}%), dropped {dropped}")

    dropped_all = total_all - kept_all
    ratio_all = (kept_all / total_all * 100.0) if total_all else 0.0
    print(f"TOTAL: kept {kept_all}/{total_all} ({ratio_all:.2f}%), dropped {dropped_all}")


if __name__ == "__main__":
    main()
