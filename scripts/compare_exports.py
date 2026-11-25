#!/usr/bin/env python3
"""Compare the regenerated thesis_exports/tables/ against the committed ones.

Byte-for-byte equality is the wrong test for floating-point CSVs: a different
core count moves the last decimal of a repeated-CV mean without changing any
number the thesis quotes. This compares every table numerically with a
tolerance, treats a missing or extra file as drift, and ignores tables that
measure the machine rather than the model.

Usage:
    python scripts/compare_exports.py [--ref <git-ref>] [--tol 1e-5]

Exit codes: 0 = no drift, 4 = drift (details printed), 1 = usage/error.
"""

import argparse
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path("thesis_exports/tables")
# Machine-dependent by construction; regenerated every run, never compared.
IGNORE = {"inference_latency.csv"}


def committed(path: Path, ref: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{path.as_posix()}"], text=True
        )
    except subprocess.CalledProcessError:
        return None


def compare(name: str, old_text: str, new_text: str, tol: float) -> list[str]:
    problems = []
    try:
        old = pd.read_csv(io.StringIO(old_text))
        new = pd.read_csv(io.StringIO(new_text))
    except Exception as exc:
        return [f"{name}: cannot parse reference or regenerated CSV ({exc})"]
    if list(old.columns) != list(new.columns):
        return [f"{name}: columns differ {list(old.columns)} -> {list(new.columns)}"]
    if old.shape != new.shape:
        return [f"{name}: shape differs {old.shape} -> {new.shape}"]
    for col in old.columns:
        a, b = old[col], new[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            a_, b_ = a.astype(float).to_numpy(), b.astype(float).to_numpy()
            bad = ~np.isclose(a_, b_, rtol=0, atol=tol, equal_nan=True)
            if bad.any():
                i = int(np.argmax(bad))
                problems.append(
                    f"{name}: column {col!r} differs at row {i}: {a_[i]!r} -> {b_[i]!r} "
                    f"({int(bad.sum())} cell(s) beyond tol={tol})"
                )
        else:
            if not a.astype(str).equals(b.astype(str)):
                i = int(np.argmax((a.astype(str) != b.astype(str)).to_numpy()))
                problems.append(
                    f"{name}: text column {col!r} differs at row {i}: {a.iloc[i]!r} -> {b.iloc[i]!r}"
                )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group()
    source.add_argument(
        "--ref", default="HEAD", help="git ref holding the reference exports"
    )
    source.add_argument(
        "--reference-dir", type=Path,
        help="saved tables directory from an independent run, instead of a git ref",
    )
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    on_disk = {p.name for p in TABLES.glob("*.csv")} - IGNORE
    if not np.isfinite(args.tol) or args.tol < 0:
        ap.error("--tol must be finite and nonnegative")
    if args.reference_dir is not None:
        if not args.reference_dir.is_dir():
            ap.error("--reference-dir must be an existing directory")
        if args.reference_dir.resolve() == TABLES.resolve():
            ap.error("--reference-dir must differ from the regenerated tables directory")
        in_ref = {p.name for p in args.reference_dir.glob("*.csv")} - IGNORE
        if not in_ref:
            ap.error("--reference-dir contains no comparable CSV files")
        label = str(args.reference_dir)
    else:
        listed = subprocess.check_output(
            ["git", "ls-tree", "--name-only", args.ref, TABLES.as_posix() + "/"], text=True
        )
        in_ref = {Path(p).name for p in listed.split() if p.endswith(".csv")} - IGNORE
        label = args.ref
        if not in_ref:
            ap.error("git reference contains no comparable CSV files")

    problems = []
    for name in sorted(in_ref - on_disk):
        problems.append(f"{name}: present in {label}, missing on disk")
    for name in sorted(on_disk - in_ref):
        problems.append(
            f"{name}: new on disk, absent from {label} (add it to the manifest and the thesis, or remove it)"
        )
    for name in sorted(on_disk & in_ref):
        old = ((args.reference_dir / name).read_text() if args.reference_dir is not None
               else committed(TABLES / name, args.ref))
        if old is None:
            problems.append(f"{name}: could not read reference from {label}")
            continue
        new = (TABLES / name).read_text()
        problems.extend(compare(name, old, new, args.tol))

    n = len(on_disk & in_ref)
    if problems:
        print(
            f"DRIFT: {len(problems)} problem(s) across {n} compared tables (tol={args.tol}):"
        )
        for p in problems:
            print("  - " + p)
        return 4
    print(
        f"OK: {n} tables match {label} within tol={args.tol}; ignored: {sorted(IGNORE)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
