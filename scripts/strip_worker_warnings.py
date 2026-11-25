#!/usr/bin/env python3
"""Remove repeated scikit-learn worker warning streams from an executed notebook.

A raw `nbconvert --execute` run leaves thousands of identical stderr lines in
the tuning and cross-validation cells (joblib workers re-emit the same
`sklearn.utils.parallel.delayed` / deprecation warnings once per task). They
carry no information and once made the file 37 % larger. This script deletes
exactly those stream outputs and nothing else, so that the committed notebook
and a fresh execution differ only in what this script removes.

Kept on purpose: the SVM's `UndefinedMetricWarning` after the holdout table,
because the note that follows it explains it (see the notebook's front matter).

Usage:
    python scripts/strip_worker_warnings.py churn_modeling.ipynb [--dry-run]
"""

import re
import sys
from pathlib import Path

import nbformat

STRIP_PATTERNS = [
    r"sklearn\.utils\.parallel\.delayed",
    r"Inconsistent values: penalty=",
    r"'penalty' was deprecated",
    r"'l1_ratio=None' was deprecated",
    r"'n_jobs' has no effect since 1\.8",
    r"FutureWarning",
    r"ConvergenceWarning",
    r"UserWarning: X does not have valid feature names",
    r"TqdmWarning: IProgress not found",
]
KEEP_PATTERNS = [r"UndefinedMetricWarning"]

_strip = re.compile("|".join(STRIP_PATTERNS))
_keep = re.compile("|".join(KEEP_PATTERNS))


def clean_stream(text: str) -> str:
    """Drop warning lines (and their `warnings.warn(...)` continuation lines)."""
    # the kept SVM warning carries the absolute venv path of whoever ran it; neutralise it
    text = re.sub(r"/[^\s:]*?/site-packages/", ".venv/lib/python3.14/site-packages/", text)
    out, skip_next = [], False
    for line in text.splitlines(keepends=True):
        if _keep.search(line):
            out.append(line)
            skip_next = False
            continue
        if _strip.search(line):
            skip_next = True
            continue
        if skip_next and (line.startswith("  ") or line.strip() == ""):
            # continuation of a warnings.warn(...) traceback line
            continue
        skip_next = False
        out.append(line)
    return "".join(out)


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    path = Path(argv[0])
    dry = "--dry-run" in argv
    nb = nbformat.read(path, as_version=4)
    removed_chars, touched = 0, 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        new_outputs = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                cleaned = clean_stream(out["text"])
                removed_chars += len(out["text"]) - len(cleaned)
                if cleaned != out["text"]:
                    touched += 1
                if cleaned.strip():
                    out["text"] = cleaned
                    new_outputs.append(out)
                continue
            new_outputs.append(out)
        cell["outputs"] = new_outputs
    print(
        f"{path}: {touched} stderr stream(s) cleaned, {removed_chars:,} characters removed"
    )
    if not dry:
        nbformat.write(nb, path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
