"""Execute the Development notebook as a non-interactive container job.

The notebook is the current source of the experiment definitions. This runner
keeps its cell order and writes a small audit only after every code cell and
the required artifact checks finish successfully.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks/model_development.ipynb"
DATA_AUDIT = PROJECT_ROOT / "data/datamart/gold/data_build_audit.json"
DEVELOPMENT_ROOT = PROJECT_ROOT / "artifacts/development"
EXPERIMENT_ROOT = DEVELOPMENT_ROOT / "experiments"
REQUIRED_FINALIST_ARTIFACTS = {
    "H1_E2_RF30_XGB30_e1",
    "H2_E1_C2_TabME8_e1",
}
ARTIFACT_FILES = {"config.json", "metrics.parquet", "predictions.parquet"}


def _preflight() -> dict:
    if not DATA_AUDIT.is_file():
        raise FileNotFoundError(f"Run data-job first: {DATA_AUDIT}")
    data_audit = json.loads(DATA_AUDIT.read_text(encoding="utf-8"))
    if data_audit.get("status") != "passed":
        raise ValueError(f"Data audit did not pass: {DATA_AUDIT}")
    for hotel in ("H1", "H2"):
        for group, prefix in (
            ("features", "gold_feature_store"),
            ("labels", "gold_label_store"),
            ("model_dataset", "gold_model_dataset"),
        ):
            path = PROJECT_ROOT / "data/datamart/gold" / group / f"{prefix}_{hotel}.parquet"
            if not path.is_dir():
                raise FileNotFoundError(f"Missing Gold input: {path}")
    return json.loads((PROJECT_ROOT / "config/development_config.json").read_text(encoding="utf-8"))


def _run_notebook() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "__file__": str(NOTEBOOK)}
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(MODEL_ROOT)
    code_cells = [(i, "".join(cell["source"])) for i, cell in enumerate(notebook["cells"])
                  if cell["cell_type"] == "code"]
    for number, (index, source) in enumerate(code_cells, start=1):
        print(f"Development cell {number}/{len(code_cells)} (notebook index {index})", flush=True)
        exec(compile(source, f"{NOTEBOOK.name}:cell_{index}", "exec"), namespace)
    return len(code_cells)


def _audit(config: dict, code_cells: int) -> None:
    split_id = config["split_id"]
    split_dir = DEVELOPMENT_ROOT / "splits" / split_id
    expected_split = {"config.json", "membership.parquet", "audit.json"}
    if not split_dir.is_dir() or {p.name for p in split_dir.iterdir()} != expected_split:
        raise ValueError(f"Incomplete split artifact: {split_dir}")

    experiment_ids = sorted(
        p.name for p in EXPERIMENT_ROOT.iterdir()
        if p.is_dir() and ARTIFACT_FILES.issubset({f.name for f in p.iterdir()})
    )
    missing = sorted(REQUIRED_FINALIST_ARTIFACTS.difference(experiment_ids))
    if missing:
        raise ValueError(f"Missing Development finalist artifacts: {missing}")

    audit = {
        "status": "passed",
        "split_id": split_id,
        "code_cells_executed": code_cells,
        "complete_experiments": len(experiment_ids),
        "required_finalist_artifacts": sorted(REQUIRED_FINALIST_ARTIFACTS),
    }
    audit_path = DEVELOPMENT_ROOT / "development_job_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("Development build passed:", json.dumps(audit), flush=True)


def main() -> None:
    (DEVELOPMENT_ROOT / "development_job_audit.json").unlink(missing_ok=True)
    config = _preflight()
    code_cells = _run_notebook()
    _audit(config, code_cells)


if __name__ == "__main__":
    main()
