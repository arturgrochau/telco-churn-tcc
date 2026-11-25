#!/usr/bin/env bash
# verify_reproducibility.sh
#
# One-command reproducibility smoke test for the thesis repo.
#
# The pins this script installs (.python-version = 3.14, requirements.txt) are
# the environment that actually produced the committed thesis_exports/ — not an
# aspirational target. Python 3.14 is the tested interpreter series. The
# notebook requires pandas >= 3.0: it passes "str" to DataFrame.select_dtypes, which
# raises TypeError on pandas 2.x. See the header of requirements.txt.
#
# Steps:
#   1. Resolve the Python version pinned in .python-version with uv.
#   2. Create an isolated venv at ./.verify_venv (or reuse one if present).
#   3. Install pinned dependencies from requirements.txt.
#   4. Execute churn_modeling.ipynb headlessly (no kernel UI).
#   5. Strip the repeated worker-warning streams (scripts/strip_worker_warnings.py)
#      so the executed copy is comparable with the committed notebook.
#   6. Compare thesis_exports/tables/ against the committed tables numerically
#      (scripts/compare_exports.py, tolerance 1e-5, timing tables excluded) and
#      FAIL with exit code 4 on drift.
#
# Allow roughly 30 minutes on the reference M5 Pro; elapsed time is logged
# for each run. SVM probability calibration and repeated refits dominate it.
#
# macOS PREREQUISITE: `brew install libomp`. XGBoost links against the OpenMP
# runtime and pip cannot supply it; step 0 below checks for it and stops early
# with a clear message rather than failing 13 seconds into the notebook.
#
# Run from the repo root:
#   bash scripts/verify_reproducibility.sh
#
# Exit codes:
#   0 — notebook executed and exported tables match the reference
#   1 — environment setup failed
#   2 — notebook execution failed
#   3 — repo not at expected layout
#   4 — thesis_exports/tables/ drifted from the committed state

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

NOTEBOOK="churn_modeling.ipynb"
REQUIREMENTS="requirements.txt"
PYTHON_VERSION_FILE=".python-version"
VENV_DIR=".verify_venv"
REFERENCE="${REFERENCE:-HEAD}"
REFERENCE_DIR="${REFERENCE_DIR:-}"

# --- Step 0: OpenMP runtime check (macOS) -----------------------------------
# XGBoost's wheel dlopen()s libomp.dylib. If it is absent the notebook raises
# XGBoostError on its first modeling cell. Fail fast with an actionable message.
if [[ "$(uname -s)" == "Darwin" ]]; then
  # Either Homebrew prefix will do (Apple silicon or Intel). `ls a b` exits non-zero
  # when any one path is missing, so test the two paths separately.
  if [[ ! -f /opt/homebrew/opt/libomp/lib/libomp.dylib \
        && ! -f /usr/local/opt/libomp/lib/libomp.dylib ]]; then
    echo "ERROR: OpenMP runtime (libomp) not found." >&2
    echo "       XGBoost cannot load without it and the notebook will fail" >&2
    echo "       on its first modeling cell. Fix with:" >&2
    echo "         brew install libomp" >&2
    exit 1
  fi
fi

log() { printf '[verify] %s\n' "$*"; }

if [[ ! -f "$NOTEBOOK" || ! -f "$REQUIREMENTS" ]]; then
  log "ERROR: must run from repo root (missing $NOTEBOOK or $REQUIREMENTS)."
  exit 3
fi

if ! command -v uv >/dev/null 2>&1; then
  log "ERROR: uv is required to build the pinned environment."
  exit 1
fi
if [[ ! -f "$PYTHON_VERSION_FILE" ]]; then
  log "ERROR: missing $PYTHON_VERSION_FILE."
  exit 3
fi
pinned_py="$(tr -d '[:space:]' < "$PYTHON_VERSION_FILE")"
log "creating/reusing $VENV_DIR with Python $pinned_py"
uv venv --allow-existing --python "$pinned_py" "$VENV_DIR"
PYTHON_BIN="$REPO_ROOT/$VENV_DIR/bin/python"
current_py="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$current_py" != "$pinned_py" ]]; then
  log "ERROR: environment Python $current_py differs from pin $pinned_py."
  exit 1
fi
log "installing pinned dependencies with uv"
uv pip install --python "$PYTHON_BIN" --requirement "$REQUIREMENTS"

log "checking the dataset snapshot"
if ! "$PYTHON_BIN" - <<'PY'
import hashlib
from pathlib import Path
name = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
paths = [p for p in (Path(name), Path("data") / name) if p.is_file()]
expected = "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"
if not paths:
    raise SystemExit("Missing Telco dataset")
for path in paths:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"Dataset hash mismatch: {path}: {actual}")
print(f"Dataset SHA256 verified ({len(paths)} matching copy/copies)")
PY
then
  exit 1
fi

# The kernel is installed only inside this environment, not in the user's
# global Jupyter registry. Prefer it over any same-name user kernelspec.
"$PYTHON_BIN" -m ipykernel install --prefix "$REPO_ROOT/$VENV_DIR" \
  --name tcc-verify --display-name "TCC verification"
export JUPYTER_PATH="$REPO_ROOT/$VENV_DIR/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"

# We deliberately invoke nbconvert via `python -m nbconvert`. The
# `jupyter` command dispatcher is not always present in a minimal
# venv that only installs `nbconvert` (the dispatcher ships with
# jupyter-core's `jupyter` console_script, which `nbconvert` does
# not pull in transitively on all platforms).
#
# We write the executed copy to <name>.executed.ipynb (gitignored)
# so the canonical $NOTEBOOK is never overwritten by the verifier.
# The artifact-drift check below still runs against the same
# thesis_exports/ that the executed notebook writes to.
EXECUTED_NB="${NOTEBOOK%.ipynb}.executed.ipynb"
cp "$NOTEBOOK" "$EXECUTED_NB"

# ExecutePreprocessor.timeout is PER CELL, not for the whole notebook.
# Historical measurements from the September run (cell metadata, M5 Pro):
#   calibration (5x isotonic CalibratedClassifierCV refits)   453 s
#   SVM RandomizedSearchCV                                      303 s
#   RepeatedStratifiedKFold, 15 folds x 5 models                167 s
# 5400 s (90 min) leaves headroom for slower machines and limits how long a
# stalled cell can block verification. Current elapsed times are in release history.
CELL_TIMEOUT="${CELL_TIMEOUT:-5400}"

log "executing $EXECUTED_NB headlessly (canonical $NOTEBOOK left untouched)"
log "per-cell timeout: ${CELL_TIMEOUT}s"
start_ts=$(date +%s)
if ! "$PYTHON_BIN" -m nbconvert \
      --to notebook \
      --execute "$EXECUTED_NB" \
      --output "$EXECUTED_NB" \
      --ExecutePreprocessor.timeout="$CELL_TIMEOUT" \
      --ExecutePreprocessor.kernel_name=tcc-verify; then
  log "ERROR: notebook execution failed (see $EXECUTED_NB for the partial run)"
  exit 2
fi
end_ts=$(date +%s)
log "notebook executed in $((end_ts - start_ts))s"

log "stripping repeated worker-warning streams from $EXECUTED_NB"
"$PYTHON_BIN" scripts/strip_worker_warnings.py "$EXECUTED_NB"

log "comparing thesis_exports/tables/ against the reference (numeric, tol 1e-5)"
compare_args=(--ref "$REFERENCE")
if [[ -n "$REFERENCE_DIR" ]]; then
  compare_args=(--reference-dir "$REFERENCE_DIR")
fi
if "$PYTHON_BIN" scripts/compare_exports.py "${compare_args[@]}" --tol 1e-5; then
  log "OK: exported tables reproduce the selected reference"
else
  log "ERROR: thesis_exports/tables/ drifted from the selected reference (see above)."
  log "       Figures are regenerated but not compared (PDF metadata carries a timestamp);"
  log "       review them with: git status thesis_exports/figures/"
  exit 4
fi

log "done."
