"""Load saved OOF predictions and discover artifacts for Section 4 evaluation.

Reads immutable artifacts produced by the Section 3 model development
pipeline (``../Model pipeline/experiments/<artifact_id>/``). This module
never fits a model or writes to the experiment store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------------
# Shortlist (mirrors Section 7 of model_development.ipynb and
# utils/model_calibration.py's SHORTLIST dict).
#
# These are the uncalibrated base artifacts. Calibrated variants (Sigmoid,
# Isotonic) are discovered by searching the experiment store for configs
# that reference these as their source.
# -----------------------------------------------------------------------------
UNCALIBRATED_SHORTLIST: dict[str, dict[str, str]] = {
    "H1": {
        "H1-E2 (RF30+XGB30 ensemble)": "H1_E2_RF30_XGB30_e1",
        "RW365 (RF30 + 365d recency)": "H1_RF_RW365_s1",
    },
    "H2": {
        "H2-E4 (C2+C4+TabM ensemble)":  "H2_E4_C2_C4_TabME8_e1",
        "H2-E3 (C2+C4 ensemble)":        "H2_E3_C2_C4_e1",
        "H2-E1 (C2+TabM ensemble)":     "H2_E1_C2_TabME8_e1",
        "C2 (RW180 + FE4)":             "H2_C2_RW180_FE4_c1",
        "C4 (RF-T5 + RW90)":            "H2_C4_T5_RW90_c1",
        "TabM-E8 (standalone)":          "H2_TabM_s1",
    },
}


# -----------------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedArtifact:
    """One loaded experiment: config + metrics table + per-booking predictions."""
    artifact_id: str
    label: str
    hotel: str
    calibration: str  # "Uncalibrated" | "Sigmoid" | "Isotonic"
    config: dict = field(repr=False)
    predictions: pd.DataFrame = field(repr=False)
    metrics: pd.DataFrame = field(repr=False)

    @property
    def n_predictions(self) -> int:
        return int(len(self.predictions))


# -----------------------------------------------------------------------------
# Low-level readers
# -----------------------------------------------------------------------------
def load_artifact_files(
    experiment_root: Path, artifact_id: str
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load one experiment folder's config, metrics and predictions parquet."""
    folder = Path(experiment_root) / artifact_id
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Artifact folder missing: {folder}\n"
            f"  Section 3 must have been run with its model_development.ipynb\n"
            f"  and its ``experiments/`` directory copied in alongside the\n"
            f"  model pipeline before this evaluation can run."
        )
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    metrics = pd.read_parquet(folder / "metrics.parquet")
    predictions = pd.read_parquet(folder / "predictions.parquet")
    return config, metrics, predictions


def list_experiment_ids(experiment_root: Path) -> list[str]:
    """List every artifact folder present under the experiment root."""
    root = Path(experiment_root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


# -----------------------------------------------------------------------------
# Shortlist loading
# -----------------------------------------------------------------------------
def load_shortlist_uncalibrated(experiment_root: Path) -> list[LoadedArtifact]:
    """Load the uncalibrated OOF predictions for every shortlisted finalist."""
    out: list[LoadedArtifact] = []
    for hotel, items in UNCALIBRATED_SHORTLIST.items():
        for label, artifact_id in items.items():
            config, metrics, predictions = load_artifact_files(
                experiment_root, artifact_id
            )
            out.append(LoadedArtifact(
                artifact_id=artifact_id,
                label=label,
                hotel=hotel,
                calibration="Uncalibrated",
                config=config,
                predictions=predictions,
                metrics=metrics,
            ))
    return out


def _calibration_source_from_config(config: dict) -> tuple[str | None, str | None]:
    """Best-effort extraction of (source_artifact_id, method) from a config.

    Section 3's calibration configs vary slightly in exact shape across runs;
    we check all reasonable field names so this works even if a future save
    schema moves things around.
    """
    request = config.get("request", config)
    method = (
        request.get("method")
        or request.get("calibration")
        or request.get("calibration_method")
    )
    source = (
        request.get("source_artifact_id")
        or request.get("source_id")
        or request.get("source")
    )
    if source is None:
        for key in ("component", "components", "source_components"):
            components = request.get(key)
            if isinstance(components, list) and len(components) == 1:
                item = components[0]
                if isinstance(item, dict):
                    source = item.get("artifact_id") or item.get("source_id")
                    if source:
                        break
                elif isinstance(item, str):
                    source = item
                    break
            elif isinstance(components, dict) and len(components) == 1:
                source = next(iter(components.values()))
                break
    return source, method


def discover_calibrated_variants(
    experiment_root: Path, source_artifact_id: str,
) -> dict[str, str]:
    """Find {method: artifact_id} for Sigmoid and Isotonic variants of a source."""
    root = Path(experiment_root)
    found: dict[str, str] = {}
    if not root.is_dir():
        return found
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        config_path = folder / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        src, method = _calibration_source_from_config(config)
        if src == source_artifact_id and method in ("Sigmoid", "Isotonic"):
            found.setdefault(method, folder.name)
    return found


def load_shortlist_all_calibrations(
    experiment_root: Path, verbose: bool = True,
) -> list[LoadedArtifact]:
    """Load the shortlist including Uncalibrated + any discovered calibrated variants.

    Returns one LoadedArtifact per (candidate, calibration) pair found. If a
    calibrated variant is missing we quietly skip it; print a short line per
    skipped candidate when ``verbose`` is True so the notebook makes clear
    what is on disk and what is not.
    """
    out = list(load_shortlist_uncalibrated(experiment_root))
    for base in list(out):
        variants = discover_calibrated_variants(experiment_root, base.artifact_id)
        for method in ("Sigmoid", "Isotonic"):
            variant_id = variants.get(method)
            if variant_id is None:
                if verbose:
                    print(f"  [skip] {base.hotel} {base.label!r}: no {method} artifact found")
                continue
            config, metrics, predictions = load_artifact_files(
                experiment_root, variant_id
            )
            out.append(LoadedArtifact(
                artifact_id=variant_id,
                label=base.label,
                hotel=base.hotel,
                calibration=method,
                config=config,
                predictions=predictions,
                metrics=metrics,
            ))
    return out


# -----------------------------------------------------------------------------
# Convenience accessors
# -----------------------------------------------------------------------------
def artifacts_to_long_predictions(artifacts: list[LoadedArtifact]) -> pd.DataFrame:
    """Stack every artifact's OOF predictions into one long DataFrame.

    Adds columns ``_label``, ``_hotel`` and ``_calibration`` so downstream
    code can filter without re-joining to the artifact list.
    """
    frames = []
    for art in artifacts:
        chunk = art.predictions.copy()
        chunk["_label"] = art.label
        chunk["_hotel"] = art.hotel
        chunk["_calibration"] = art.calibration
        chunk["_artifact_id"] = art.artifact_id
        frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def align_calibration_variants(
    artifacts: list[LoadedArtifact],
    verbose: bool = True,
) -> list[LoadedArtifact]:
    """Within each (hotel, candidate_label) group, filter predictions to the
    fold set common to every calibration variant of that candidate.

    Why this matters
    ----------------
    Section 3's forward-only calibration drops folds that have no eligible
    earlier history under the readiness rule (typically Fold 1), so a saved
    Sigmoid or Isotonic artifact may cover only F2-F4 while its Uncalibrated
    source covers F1-F4. Pooled metrics computed on those different fold sets
    conflate the calibration effect with the fold-mix effect — a model may
    look "better calibrated" simply because F1 (a known difficult fold) was
    dropped. This helper filters every variant to the SAME folds so pooled
    comparisons are like-for-like.

    Parameters
    ----------
    artifacts : the output of ``load_shortlist_all_calibrations``.
    verbose   : when True, print one line per group showing which folds
                are kept and which are dropped.

    Returns
    -------
    list[LoadedArtifact]
        A new list where every variant within a group has predictions
        restricted to the shared fold set. Groups with no common folds are
        omitted entirely (and announced if ``verbose``).
    """
    by_group: dict[tuple[str, str], list[LoadedArtifact]] = {}
    for art in artifacts:
        key = (art.hotel, art.label)
        by_group.setdefault(key, []).append(art)

    out: list[LoadedArtifact] = []
    for (hotel, label), group in by_group.items():
        if not group:
            continue
        fold_sets = [set(int(f) for f in a.predictions["fold"].unique()) for a in group]
        common_folds = set.intersection(*fold_sets) if fold_sets else set()
        dropped_folds = set.union(*fold_sets) - common_folds if fold_sets else set()

        if verbose:
            per_variant = ", ".join(
                f"{a.calibration}={sorted(fs)}" for a, fs in zip(group, fold_sets)
            )
            if dropped_folds:
                print(
                    f"  [{hotel}] {label!r}: keep folds {sorted(common_folds)}, "
                    f"drop {sorted(dropped_folds)}  ({per_variant})"
                )
            else:
                print(f"  [{hotel}] {label!r}: all variants cover folds {sorted(common_folds)}")

        if not common_folds:
            if verbose:
                print(f"  [{hotel}] {label!r}: no common folds -> group omitted")
            continue

        for art in group:
            mask = art.predictions["fold"].isin(common_folds)
            filtered = art.predictions.loc[mask].reset_index(drop=True)
            out.append(LoadedArtifact(
                artifact_id=art.artifact_id,
                label=art.label,
                hotel=art.hotel,
                calibration=art.calibration,
                config=art.config,
                predictions=filtered,
                metrics=art.metrics,
            ))
    return out
