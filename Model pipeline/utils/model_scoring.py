"""Load persisted models and score new bookings.

Handoff interface for the deployment / MLOps team. Pairs with
``model_refit.persist_hotel_models`` which writes the artifacts this
module reads.

Typical usage::

    from utils.model_scoring import load_hotel_model, score_bookings

    # Daily batch at CutoffDate = today:
    bookings = <frame of today's T-14 cohort with the Gold feature columns>
    model = load_hotel_model("H1", models_root=Path("models"))
    p_cancel = score_bookings(model, bookings)
    # p_cancel is one float per booking; apply operations ranking from here.

The scoring function returns an ensemble equal-weight average of
component ``predict_proba[:, 1]``. For H1 both components are sklearn
pipelines; for H2 the C2 component is a sklearn pipeline but TabM-E8
requires a refit (Section 3's TabM fit does not expose a saved torch
module). The ``load_hotel_model`` helper raises a clear error if TabM
is required but not available, so the deploy team knows exactly what to
build.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class LoadedHotelModel:
    """A hotel's finalist ensemble, ready to score new bookings."""
    hotel: str
    model_version: str
    ensemble_id: str
    metadata: dict = field(repr=False)
    # Component name -> fitted sklearn Pipeline
    pipelines: dict = field(default_factory=dict, repr=False)
    # Component name -> FE4 mapping dict (segment -> rate, fallback_rate, ...)
    fe4_mappings: dict = field(default_factory=dict, repr=False)
    # Component name -> loaded TabM module + preprocess bundle + architecture
    tabm_models: dict = field(default_factory=dict, repr=False)


def load_hotel_model(
    hotel: str,
    models_root: Path,
    model_version: str | None = None,
) -> LoadedHotelModel:
    """Load the newest fitted finalist for ``hotel`` from ``models_root``.

    Pass ``model_version`` explicitly to pin a version. Otherwise the
    helper picks the most recent folder whose metadata declares the
    matching ``hotel``.
    """
    root = Path(models_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Models root does not exist: {root}")

    if model_version is None:
        candidates = []
        for folder in root.iterdir():
            metadata_path = folder / "metadata.json"
            if not metadata_path.exists():
                continue
            try:
                meta = json.loads(metadata_path.read_text())
                if meta.get("hotel") == hotel:
                    candidates.append((meta["created_at_utc"], folder.name))
            except (json.JSONDecodeError, KeyError):
                continue
        if not candidates:
            raise FileNotFoundError(f"No persisted model found for hotel {hotel!r}")
        candidates.sort()
        model_version = candidates[-1][1]

    folder = root / model_version
    metadata = json.loads((folder / "metadata.json").read_text())

    model = LoadedHotelModel(
        hotel=hotel,
        model_version=model_version,
        ensemble_id=metadata["ensemble_id"],
        metadata=metadata,
    )

    for component in metadata["components"]:
        cname = component["component"]
        artifact_type = component["artifact_type"]
        if artifact_type == "sklearn_pipeline_pickle":
            with open(folder / component["artifact_path"], "rb") as fh:
                model.pipelines[cname] = pickle.load(fh)
            if "fe4_mapping_path" in component:
                fe4_path = folder / component["fe4_mapping_path"]
                model.fe4_mappings[cname] = json.loads(fe4_path.read_text())
        elif artifact_type == "tabm_state_dict":
            # Lazy import — only pay the torch/tabm cost when needed
            import torch
            from tabm import TabM
            preprocess = json.loads((folder / component["preprocess_path"]).read_text())
            spec = json.loads((folder / component["spec_path"]).read_text())
            arch = spec["architecture"]
            tabm_module = TabM.make(
                n_num_features=len(preprocess["numeric_columns"]),
                cat_cardinalities=preprocess["cardinalities"],
                d_out=preprocess.get("d_out", 1),
                **arch,
            )
            state_dict = torch.load(
                folder / component["state_dict_path"], map_location="cpu",
            )
            tabm_module.load_state_dict(state_dict)
            tabm_module.eval()
            model.tabm_models[cname] = {
                "module": tabm_module,
                "preprocess": preprocess,
                "architecture": arch,
            }
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")

    return model


def _apply_fe4(bookings: pd.DataFrame, mapping: dict) -> pd.Series:
    """Map each booking's MarketSegment to the saved historical cancel rate."""
    seg = bookings["MarketSegment"].fillna("__MISSING__").astype(str)
    rate = seg.map(mapping["segments"]).fillna(mapping["fallback_rate"])
    return pd.Series(rate.to_numpy(), index=bookings.index, name=mapping["feature_name"])


def _score_tabm(bookings: pd.DataFrame, tabm_bundle: dict) -> np.ndarray:
    """Score bookings with a loaded TabM module using the saved preprocessing."""
    import torch
    pre = tabm_bundle["preprocess"]
    module = tabm_bundle["module"]

    # Numeric: median impute (train medians) + standard scale (train params)
    numeric = pre["numeric_columns"]
    medians = pd.Series(pre["medians"])
    num_vals = bookings[numeric].apply(pd.to_numeric, errors="coerce")
    num_vals = num_vals.fillna(medians[numeric])
    mean = np.array(pre["scaler_mean"], dtype="float32")
    scale = np.array(pre["scaler_scale"], dtype="float32")
    x_num = ((num_vals.to_numpy(dtype="float32") - mean) / scale).astype("float32")

    # Categorical: train mapping; missing/unseen → index 0
    categorical = pre["categorical_columns"]
    mappings = pre["categorical_mappings"]
    x_cat_cols = []
    for col, mapping in zip(categorical, mappings):
        series = bookings[col].fillna("__MISSING__").astype(str)
        encoded = series.map(mapping).fillna(0).astype("int64")
        x_cat_cols.append(encoded.to_numpy())
    x_cat = np.column_stack(x_cat_cols)

    batch_size = int(pre.get("batch_size", 1024))
    pieces = []
    module.eval()
    with torch.no_grad():
        for start in range(0, len(x_num), batch_size):
            stop = start + batch_size
            num = torch.from_numpy(x_num[start:stop])
            cat = torch.from_numpy(x_cat[start:stop])
            logits = module(num, cat).squeeze(-1)
            # Average sigmoid across TabM's k ensemble members
            pieces.append(torch.sigmoid(logits).mean(dim=1).cpu().numpy())
    return np.concatenate(pieces)


def score_bookings(
    model: LoadedHotelModel,
    bookings: pd.DataFrame,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return per-booking p_cancel for one or more bookings.

    Parameters
    ----------
    model : LoadedHotelModel
        Fitted finalist, from ``load_hotel_model``.
    bookings : DataFrame
        One row per booking, with the feature columns the model needs.
    as_of_date : str | Timestamp | None
        The business-cutoff date these predictions represent (what day
        we are standing on when we score). In production this is the
        batch date (``CutoffDate = today``); in a backtest it is the
        booking's CutoffDate. ISO ``YYYY-MM-DD`` string or Timestamp.
        If None, defaults to the current UTC date.

    Returns
    -------
    DataFrame with columns:
        Hotel, Booking_ID,
        p_cancel,                    (ensemble average)
        p_<component>,               (per-component p for audit)
        model_version,               (joinable to the model registry)
        prediction_as_of_date,       (business-cutoff date; what the row claims to represent)
        generated_at_utc,            (actual UTC wall-clock when the row was produced)
    """
    if bookings.empty:
        return pd.DataFrame(columns=[
            "Hotel", "Booking_ID", "p_cancel", "model_version",
            "prediction_as_of_date", "generated_at_utc",
        ])

    out = bookings[["Hotel", "Booking_ID"]].copy()
    out["model_version"] = model.model_version
    if as_of_date is None:
        as_of_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        as_of_str = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
    out["prediction_as_of_date"] = as_of_str
    out["generated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    component_scores = {}

    for component in model.metadata["components"]:
        cname = component["component"]
        if cname in model.pipelines:
            pipeline = model.pipelines[cname]
            feature_cols = component["feature_columns"]
            scored = bookings.copy()
            if cname in model.fe4_mappings:
                fe4_name = model.fe4_mappings[cname]["feature_name"]
                scored[fe4_name] = _apply_fe4(scored, model.fe4_mappings[cname])
            missing = [c for c in feature_cols if c not in scored.columns]
            if missing:
                raise ValueError(
                    f"Scoring {cname}: missing feature columns {missing}"
                )
            p = pipeline.predict_proba(scored[feature_cols])[:, 1]
            component_scores[cname] = p
            out[f"p_{cname}"] = p
        elif cname in model.tabm_models:
            p = _score_tabm(bookings, model.tabm_models[cname])
            component_scores[cname] = p
            out[f"p_{cname}"] = p
        else:
            raise RuntimeError(
                f"Component {cname!r} is listed in metadata but was not loaded. "
                f"The registry may be incomplete — check {model.model_version}/metadata.json."
            )

    if not component_scores:
        raise RuntimeError("No components produced scores")

    scores_matrix = np.column_stack(list(component_scores.values()))
    out["p_cancel"] = scores_matrix.mean(axis=1)
    return out
