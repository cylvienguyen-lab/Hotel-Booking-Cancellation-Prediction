"""Reproduce the frozen finalist decisions from Development OOF results.

Only code cells through the validation decision in evaluation_notebook.ipynb
are executed. Test evaluation starts later in the notebook and is excluded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from IPython.display import display


EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_ROOT.parents[1]
DEVELOPMENT_ROOT = PROJECT_ROOT / "artifacts/development"
NOTEBOOK = PROJECT_ROOT / "notebooks/evaluation_notebook.ipynb"
DEVELOPMENT_AUDIT = DEVELOPMENT_ROOT / "development_job_audit.json"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/evaluation"
SELECTION_AUDIT = OUTPUT_ROOT / "selection_job_audit.json"
LAST_VALIDATION_CELL = 32
EXPECTED_FINALISTS = {
    "H1": ("H1-E2 (RF30+XGB30 ensemble)", "Uncalibrated", "H1_E2_RF30_XGB30_e1"),
    "H2": ("H2-E1 (C2+TabM ensemble)", "Uncalibrated", "H2_E1_C2_TabME8_e1"),
}
VALIDATION_OUTPUTS = (
    "layer_a_pooled_native_folds.parquet",
    "layer_a_per_fold.parquet",
    "layer_a_pooled_aligned.parquet",
    "layer_a_per_fold_aligned.parquet",
    "layer_a_operating_points.parquet",
    "finalist_choice.parquet",
)


def _preflight() -> None:
    if not DEVELOPMENT_AUDIT.is_file():
        raise FileNotFoundError(f"Run development-job first: {DEVELOPMENT_AUDIT}")
    audit = json.loads(DEVELOPMENT_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("split_id") != "split_v2":
        raise ValueError(f"Development audit did not pass: {DEVELOPMENT_AUDIT}")
    split_root = DEVELOPMENT_ROOT / "splits/split_v2"
    if not (split_root / "membership.parquet").is_file():
        raise FileNotFoundError(f"Missing Development membership: {split_root}")
    for _, _, artifact_id in EXPECTED_FINALISTS.values():
        folder = DEVELOPMENT_ROOT / "experiments" / artifact_id
        for name in ("config.json", "metrics.parquet", "predictions.parquet"):
            if not (folder / name).is_file():
                raise FileNotFoundError(f"Missing OOF experiment artifact: {folder / name}")


def _run_validation_cells() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [
        (index, "".join(cell["source"]))
        for index, cell in enumerate(notebook["cells"])
        if index <= LAST_VALIDATION_CELL and cell["cell_type"] == "code"
    ]
    if not cells or cells[-1][0] != LAST_VALIDATION_CELL:
        raise ValueError("Notebook validation boundary changed; review cell selection")
    forbidden = ("test_predictions.parquet", "layer_a_test.parquet")
    for index, source in cells:
        if any(name in source for name in forbidden):
            raise ValueError(f"Test artifact reference found before validation boundary: cell {index}")

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(EVAL_ROOT)
    namespace = {"__name__": "__main__", "__file__": str(NOTEBOOK), "display": display}
    for number, (index, source) in enumerate(cells, start=1):
        print(f"Validation cell {number}/{len(cells)} (notebook index {index})", flush=True)
        exec(compile(source, f"{NOTEBOOK.name}:cell_{index}", "exec"), namespace)
    return len(cells)


def _audit(code_cells: int) -> None:
    for name in VALIDATION_OUTPUTS:
        path = OUTPUT_ROOT / name
        if not path.is_file() or pd.read_parquet(path).empty:
            raise ValueError(f"Missing or empty validation output: {path}")

    decisions = pd.read_parquet(OUTPUT_ROOT / "finalist_choice.parquet")
    expected_columns = {"hotel", "finalist_label", "calibration", "artifact_id"}
    if set(decisions.columns) != expected_columns:
        raise ValueError(f"Unexpected finalist choice columns: {list(decisions.columns)}")
    if len(decisions) != 2 or set(decisions["hotel"]) != set(EXPECTED_FINALISTS):
        raise ValueError("Finalist choice must contain one row each for H1 and H2")
    for row in decisions.itertuples(index=False):
        actual = (row.finalist_label, row.calibration, row.artifact_id)
        if actual != EXPECTED_FINALISTS[row.hotel]:
            raise ValueError(f"{row.hotel}: finalist decision differs from the frozen choice: {actual}")

    audit = {
        "status": "passed",
        "source": "Development OOF validation artifacts",
        "code_cells_executed": code_cells,
        "finalists": {
            row.hotel: {
                "label": row.finalist_label,
                "calibration": row.calibration,
                "artifact_id": row.artifact_id,
            }
            for row in decisions.itertuples(index=False)
        },
        "validation_outputs": list(VALIDATION_OUTPUTS),
    }
    SELECTION_AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("Validation selection passed:", json.dumps(audit), flush=True)


def main() -> None:
    SELECTION_AUDIT.unlink(missing_ok=True)
    _preflight()
    for name in VALIDATION_OUTPUTS:
        (OUTPUT_ROOT / name).unlink(missing_ok=True)
    code_cells = _run_validation_cells()
    _audit(code_cells)


if __name__ == "__main__":
    main()
