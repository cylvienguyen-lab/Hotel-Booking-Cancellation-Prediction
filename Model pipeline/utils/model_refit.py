"""Refit the Section 4 finalists on all development data and score Test.

Section 3 fits per fold and never saves a production model. This module
does the "fit once on all eligible development rows, then score Test"
step the deployment story needs. It is intentionally a thin wrapper over
Cylvie's screening/experiment utilities so the same preprocessor,
recency weighting and TabM training path are reused — nothing is
reimplemented.

Finalists (set in `../Evaluation pipeline/outputs/finalist_choice.parquet`):
    H1 -> H1-E2 = RF30 + XGB30 equal-weight ensemble, Uncalibrated
    H2 -> H2-E1 = C2 + TabM-E8 equal-weight ensemble, Uncalibrated
         where C2 = RF30 + 180-day recency weighting + FE4
               TabM-E8 = TabM trained for 8 epochs

Public entry point: ``refit_and_score_finalist``.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from utils.model_preprocessing import build_preprocessor
from utils.model_screening import (
    _engineer,
    _fit_predict_sklearn,
    _fit_predict_tabm,
)


# -----------------------------------------------------------------------------
# Finalist specs — copied verbatim from the saved artifact configs.
#   H1_RF_v2, H1_XGB_v2  -> components of H1_E2_RF30_XGB30_e1
#   H2_C2_RW180_FE4_c1   -> C2 component of H2_E1
#   H2_TabM_s1 (epochs 8) -> TabM-E8 component of H2_E1
# -----------------------------------------------------------------------------
RF30_SPEC = {
    "family": "RF",
    "estimator_params": {
        "n_estimators": 300, "max_depth": None, "min_samples_leaf": 20,
        "max_features": "sqrt", "criterion": "gini", "bootstrap": True,
        "class_weight": None, "random_state": 42, "n_jobs": 2,
    },
    "seed": 42,
}

XGB30_SPEC = {
    "family": "XGB",
    "estimator_params": {
        "n_estimators": 300, "learning_rate": 0.05, "max_depth": 3,
        "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_lambda": 1, "scale_pos_weight": 1, "objective": "binary:logistic",
        "eval_metric": "logloss", "tree_method": "hist", "device": "cpu",
        "random_state": 42, "n_jobs": 2, "early_stopping_rounds": None,
    },
    "seed": 42,
}

C2_SPEC = {
    "family": "RF",
    "estimator_params": RF30_SPEC["estimator_params"],
    "half_life_days": 180,
    "feature_experiment": "FE4",
    "seed": 42,
}

TABM_E8_SPEC = {
    "family": "TabM",
    "architecture": {"d_block": 64, "n_blocks": 2, "k": 8, "dropout": 0.1},
    "training_params": {
        "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 1e-4,
        "batch_size": 1024, "epochs": 8,
    },
    "seed": 42,
}


FINALISTS = {
    "H1": {
        "ensemble_id": "H1-E2",
        "components": {"RF30": RF30_SPEC, "XGB30": XGB30_SPEC},
    },
    "H2": {
        "ensemble_id": "H2-E1",
        "components": {"C2": C2_SPEC, "TabM-E8": TABM_E8_SPEC},
    },
}


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RefitCohorts:
    """Train and Test frames for one hotel, with everything merged."""
    hotel: str
    train: pd.DataFrame
    test: pd.DataFrame
    fit_date: pd.Timestamp
    selected_features: dict
    feature_contract: dict


def load_cohorts(
    hotel: str,
    gold_dataset_root: Path,
    membership_path: Path,
    split_config_path: Path,
    feature_contract_path: Path,
) -> RefitCohorts:
    """Load Gold predictors + membership and build (train, test) frames.

    The training set is Dev bookings with ``FinalRefit == 1`` — the saved
    membership flag that implements the three-way eligibility rule
    (CutoffDate / LabelAvailableDate / ExpectedCheckoutDate strictly before
    ``test_start``). Test is simply ``Partition == "Test"``.
    """
    if hotel not in ("H1", "H2"):
        raise ValueError(f"hotel must be H1 or H2, got {hotel!r}")

    membership = pd.read_parquet(membership_path)
    split_cfg = json.loads(Path(split_config_path).read_text())
    calendar = split_cfg.get("calendar", split_cfg.get("split_calendar", split_cfg))
    fit_date = pd.Timestamp(calendar["test_start"])

    gold = pd.read_parquet(
        Path(gold_dataset_root) / f"gold_model_dataset_{hotel}.parquet"
    )
    feature_contract = json.loads(Path(feature_contract_path).read_text())

    # Build the selected_features dict the preprocessor and screening utils
    # expect. The feature contract already lists numeric and categorical
    # predictors; we split numeric into continuous vs binary using the
    # original Section 3 config.
    BINARY = ("IsRepeatedGuest", "IsPortugalHoliday")
    numeric_all = feature_contract["numeric_predictors"]
    numeric = [c for c in numeric_all if c not in BINARY]
    binary = [c for c in numeric_all if c in BINARY]
    categorical = feature_contract["categorical_predictors"]
    selected_features = {
        "numeric": numeric, "binary": binary, "categorical": categorical,
    }

    keys = ["Hotel", "Booking_ID"]
    required = keys + numeric + binary + categorical + ["IsCanceled"]
    gold_slice = gold.loc[gold["Hotel"].eq(hotel), required]
    member_slice = membership.loc[membership["Hotel"].eq(hotel)].copy()

    merged = member_slice.merge(
        gold_slice, on=keys, how="left", validate="one_to_one",
    )
    if merged[required].isna().any().any() and merged["Partition"].eq("Dev").any():
        # Test rows may legitimately carry NaN labels in some splits; Dev must not.
        bad = merged[merged["Partition"].eq("Dev") & merged["IsCanceled"].isna()]
        if not bad.empty:
            raise ValueError(f"{len(bad)} Dev bookings missing a target")

    train = merged.loc[
        merged["Partition"].eq("Dev") & merged["FinalRefit"].eq(1)
    ].reset_index(drop=True)
    test = merged.loc[merged["Partition"].eq("Test")].reset_index(drop=True)

    if train.empty or test.empty:
        raise ValueError(f"{hotel}: empty train ({len(train)}) or test ({len(test)}) cohort")
    if train["IsCanceled"].nunique() < 2:
        raise ValueError(f"{hotel}: training set has only one class")

    return RefitCohorts(
        hotel=hotel, train=train, test=test, fit_date=fit_date,
        selected_features=selected_features, feature_contract=feature_contract,
    )


# -----------------------------------------------------------------------------
# Per-model refit
# -----------------------------------------------------------------------------
def _recency_weights(train: pd.DataFrame, fit_date: pd.Timestamp, half_life_days: int) -> np.ndarray:
    """Match the recency-weighting formula used in model_screening.run_screening_experiment."""
    age_days = (fit_date - train["CutoffDate"]).dt.days.to_numpy()
    weights = np.power(0.5, age_days / half_life_days)
    weights = weights / weights.mean()
    return weights


def _extract_fe4_mapping(
    train: pd.DataFrame, smoothing: float = 20.0,
) -> dict:
    """Rebuild the final segment -> smoothed historical cancel-rate mapping.

    This mirrors ``model_screening._chronological_segment_rate``'s final
    (validation-side) mapping, which is what a deployed scorer applies to
    new bookings. Saved as JSON so new data at inference time maps each
    MarketSegment to its historical cancel rate; unseen segments get the
    training global rate.
    """
    seg = train["MarketSegment"].fillna("__MISSING__").astype(str)
    global_rate = float(train["IsCanceled"].mean())
    agg = (
        train.assign(_segment=seg)
        .groupby("_segment")["IsCanceled"]
        .agg(["sum", "count"])
    )
    mapping = (
        (agg["sum"] + smoothing * global_rate)
        / (agg["count"] + smoothing)
    ).to_dict()
    return {
        "segments": {str(k): float(v) for k, v in mapping.items()},
        "fallback_rate": global_rate,
        "smoothing": smoothing,
        "feature_name": "HistoricalSegmentCancelRate",
    }


def _refit_sklearn_component(
    name: str, spec: dict, cohorts: RefitCohorts,
) -> tuple[pd.Series, dict, Pipeline, dict | None]:
    """Fit one sklearn-based component and return Test p_cancel + the fitted pipeline.

    Unlike the Section 3 screening path (which discards the fitted
    pipeline), this returns the ``Pipeline`` so it can be pickled for
    the deployment handoff. Also returns the FE4 segment mapping (JSON-
    serialisable dict) when the component uses that feature.
    """
    train = cohorts.train.copy()
    test = cohorts.test.copy()

    # Fold-equivalent feature engineering (fit on train, apply to test)
    train, test, added, feature_detail = _engineer(
        train, test, spec.get("feature_experiment")
    )
    roles = {
        "numeric": cohorts.selected_features["numeric"] + added,
        "binary": cohorts.selected_features["binary"],
        "categorical": cohorts.selected_features["categorical"],
    }

    weights = None
    weight_detail = None
    if spec.get("half_life_days") is not None:
        weights = _recency_weights(train, cohorts.fit_date, spec["half_life_days"])
        weight_detail = {
            "n": int(len(weights)),
            "min": float(weights.min()), "max": float(weights.max()),
            "mean": float(weights.mean()),
            "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
        }

    # Instantiate the right estimator for the family
    family = spec["family"]
    if family == "RF":
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier(**spec["estimator_params"])
    elif family == "XGB":
        from xgboost import XGBClassifier
        estimator = XGBClassifier(**spec["estimator_params"])
    else:
        raise ValueError(f"Unsupported sklearn family in refit: {family}")

    preprocessor = build_preprocessor(
        roles if added else cohorts.selected_features,
        cohorts.feature_contract,
        model_name=family,
        engineered_numeric_features=tuple(added),
    )
    columns = sum(
        (roles[g] for g in ("numeric", "binary", "categorical")), [],
    )
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)), ("model", estimator),
    ])
    fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
    t0 = time.perf_counter()
    pipeline.fit(train[columns], train["IsCanceled"].astype(int), **fit_kwargs)
    p_test = pipeline.predict_proba(test[columns])[:, 1]
    fit_detail = {
        "preprocessing": "Section 3 preprocessor; " + (
            "LR scaling" if family == "LR" else "unscaled numeric"
        ) + "; one-hot categorical",
        "weighted_estimator_fit": weights is not None,
        "training_seconds": time.perf_counter() - t0,
    }

    # Extract FE4 mapping for the deployment artifact if used
    fe4_mapping = None
    if spec.get("feature_experiment") == "FE4":
        fe4_mapping = _extract_fe4_mapping(cohorts.train, smoothing=20.0)

    detail = {
        "name": name, "family": family,
        "added_features": added, "feature_engineering": feature_detail,
        "weight_detail": weight_detail, "fit_detail": fit_detail,
        "feature_columns": columns,
        "train_n": int(len(cohorts.train)), "test_n": int(len(cohorts.test)),
    }
    return (
        pd.Series(p_test, index=cohorts.test.index, name=name),
        detail,
        pipeline,
        fe4_mapping,
    )


def _fit_tabm_full(
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected_features: dict,
    spec: dict,
    pseudo_fold: int = 5,
) -> tuple[np.ndarray, object, dict, dict]:
    """Full TabM training that returns p_test AND the trained torch module.

    This mirrors ``model_screening._fit_predict_tabm`` line-for-line (same
    preprocessing, same seed derivation = spec['seed'] + pseudo_fold, same
    AdamW setup, same per-epoch batching, same inference averaging). The
    only difference is that this one returns the trained ``torch.nn.Module``
    plus a JSON-serialisable preprocessing bundle, so both can be saved for
    deployment scoring.

    Returns
    -------
    (p_test, model, preprocess_bundle, detail)
        p_test            — np.ndarray[len(test)], the ensemble averaged sigmoid.
        model             — the trained TabM torch module (on CPU after training).
        preprocess_bundle — dict suitable for json.dump: numeric columns,
                            categorical columns, medians, scaler mean/scale,
                            categorical mappings, cardinalities, batch size.
        detail            — timing + loss history + preprocessing note.
    """
    import torch
    import torch.nn.functional as F
    from sklearn.preprocessing import StandardScaler
    from tabm import TabM

    params = spec["training_params"]
    seed = int(spec["seed"]) + pseudo_fold
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    numeric = selected_features["numeric"] + selected_features["binary"]
    categorical = selected_features["categorical"]

    # Numeric: train-median impute, train-fit standard scale
    medians = train[numeric].apply(pd.to_numeric, errors="coerce").median().fillna(0)
    scaler = StandardScaler()
    x_train_num = scaler.fit_transform(
        train[numeric].apply(pd.to_numeric, errors="coerce").fillna(medians)
    ).astype("float32")
    x_test_num = scaler.transform(
        test[numeric].apply(pd.to_numeric, errors="coerce").fillna(medians)
    ).astype("float32")

    # Categorical: train-only integer mapping, missing/unseen -> index 0
    mappings = []
    for column in categorical:
        observed = train[column].fillna("__MISSING__").astype(str)
        categories = sorted(set(observed) - {"__MISSING__"})
        mappings.append(
            {name: index for index, name in enumerate(["__MISSING__"] + categories)}
        )

    def _encode(frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            frame[column].fillna("__MISSING__").astype(str)
            .map(mapping).fillna(0).astype("int64").to_numpy()
            for column, mapping in zip(categorical, mappings)
        ])

    x_train_cat = _encode(train)
    x_test_cat = _encode(test)
    cardinalities = [len(mapping) for mapping in mappings]
    if any(x_train_cat[:, i].max() >= size for i, size in enumerate(cardinalities)):
        raise ValueError("TabM categorical index exceeds Train cardinality")
    y_train = train.IsCanceled.astype("float32").to_numpy()

    architecture = spec["architecture"]
    model = TabM.make(
        n_num_features=x_train_num.shape[1],
        cat_cardinalities=cardinalities,
        d_out=1,
        **architecture,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    batch_size = int(params["batch_size"])
    rng = np.random.default_rng(seed)
    loss_history = []
    started = time.perf_counter()
    for _ in range(int(params["epochs"])):
        model.train()
        epoch_loss = 0.0
        for indices in np.array_split(
            rng.permutation(len(train)),
            max(1, int(np.ceil(len(train) / batch_size))),
        ):
            num = torch.from_numpy(x_train_num[indices]).to(device)
            cat = torch.from_numpy(x_train_cat[indices]).to(device)
            label = torch.from_numpy(y_train[indices]).to(device)
            optimizer.zero_grad()
            logits = model(num, cat).squeeze(-1)
            if logits.ndim != 2 or logits.shape[1] != model.k:
                raise ValueError("TabM did not return member-level logits")
            loss = F.binary_cross_entropy_with_logits(
                logits, label[:, None].expand_as(logits)
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(indices)
        loss_history.append(epoch_loss / len(train))

    def _infer(num_arr: np.ndarray, cat_arr: np.ndarray) -> np.ndarray:
        model.eval()
        pieces = []
        with torch.no_grad():
            for start in range(0, len(num_arr), batch_size):
                stop = start + batch_size
                num = torch.from_numpy(num_arr[start:stop]).to(device)
                cat = torch.from_numpy(cat_arr[start:stop]).to(device)
                logits = model(num, cat).squeeze(-1)
                pieces.append(torch.sigmoid(logits).mean(dim=1).cpu().numpy())
        return np.concatenate(pieces)

    p_test = _infer(x_test_num, x_test_cat)

    # Move model to CPU so saving + reloading is portable
    model = model.cpu()

    preprocess_bundle = {
        "numeric_columns": list(numeric),
        "categorical_columns": list(categorical),
        "medians": {k: float(v) for k, v in medians.items()},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "categorical_mappings": [
            {str(k): int(v) for k, v in m.items()} for m in mappings
        ],
        "cardinalities": [int(c) for c in cardinalities],
        "batch_size": batch_size,
        "d_out": 1,
    }
    detail = {
        "preprocessing": (
            "Train median imputation + StandardScaler for numeric/binary; "
            "Train-only categorical integer mapping; missing/unseen index 0"
        ),
        "categorical_cardinalities": [int(c) for c in cardinalities],
        "device": str(device),
        "optimizer": "AdamW",
        "architecture": architecture,
        "training_params": params,
        "stopping": "fixed epoch budget; no outer-Validation monitoring",
        "train_loss_by_epoch": loss_history,
        "training_seconds": time.perf_counter() - started,
    }
    return p_test, model, preprocess_bundle, detail


def _refit_tabm_component(
    name: str, spec: dict, cohorts: RefitCohorts, pseudo_fold: int = 5,
) -> tuple[pd.Series, dict, dict | None]:
    """Fit TabM via ``_fit_tabm_full`` and return (series, detail, tabm_payload).

    ``tabm_payload`` carries the trained torch module and the preprocessing
    bundle so ``persist_hotel_models`` can serialise both. If torch or tabm
    are unavailable the function raises; TabM is a hard dependency for H2
    scoring in this project.
    """
    p_test, model, preprocess_bundle, fit_detail = _fit_tabm_full(
        train=cohorts.train, test=cohorts.test,
        selected_features=cohorts.selected_features,
        spec=spec, pseudo_fold=pseudo_fold,
    )
    detail = {
        "name": name, "family": "TabM", "fit_detail": fit_detail,
        "train_n": int(len(cohorts.train)), "test_n": int(len(cohorts.test)),
    }
    tabm_payload = {
        "spec": spec,
        "model": model,
        "preprocess": preprocess_bundle,
        "pseudo_fold": pseudo_fold,
    }
    return (
        pd.Series(p_test, index=cohorts.test.index, name=name),
        detail,
        tabm_payload,
    )


# -----------------------------------------------------------------------------
# Full ensemble refit
# -----------------------------------------------------------------------------
def refit_and_score_finalist(
    hotel: str,
    cohorts: RefitCohorts,
    include_tabm: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Fit each component of the hotel's finalist and return the ensemble output.

    Returns
    -------
    predictions : DataFrame
        One row per Test booking with the columns the Section 4 Layer A/B/C
        code expects: Hotel, Booking_ID, ArrivalDate, ADR, TotalNights,
        p_cancel, y_true, plus per-component p_cancel columns for audit.
    detail : dict
        Timing, feature-engineering summaries and per-component configs.
    """
    finalist = FINALISTS[hotel]
    components = finalist["components"]
    per_component: dict[str, pd.Series] = {}
    details: dict[str, dict] = {}
    skipped: list[str] = []

    # Collected artifacts for persistence (populated per component)
    sklearn_pipelines: dict[str, Pipeline] = {}
    fe4_mappings: dict[str, dict] = {}
    tabm_artifacts: dict[str, dict] = {}

    for name, spec in components.items():
        if spec["family"] == "TabM" and not include_tabm:
            print(f"  [{hotel}] skipping {name} (TabM) — include_tabm=False")
            skipped.append(name)
            continue
        print(f"  [{hotel}] fitting {name} ({spec['family']})...", flush=True)
        if spec["family"] == "TabM":
            series, detail, tabm_art = _refit_tabm_component(name, spec, cohorts)
            if tabm_art is not None:
                tabm_artifacts[name] = tabm_art
        else:
            series, detail, pipeline, fe4 = _refit_sklearn_component(name, spec, cohorts)
            sklearn_pipelines[name] = pipeline
            if fe4 is not None:
                fe4_mappings[name] = fe4
        per_component[name] = series
        details[name] = detail
        print(f"    ok  n_test={len(series):,}  "
              f"took {detail['fit_detail'].get('training_seconds', float('nan')):.1f}s")

    if not per_component:
        raise RuntimeError(f"No components fitted for {hotel}")

    component_df = pd.DataFrame(per_component)
    # Equal-average ensemble over the components actually fitted.
    p_cancel = component_df.mean(axis=1).rename("p_cancel")

    # Metadata needed downstream for Layer B (per-night) and Layer C (revenue)
    test = cohorts.test
    out = pd.DataFrame({
        "Hotel": test["Hotel"],
        "Booking_ID": test["Booking_ID"],
        "ArrivalDate": test["ArrivalDate"],
        "ADR": test.get("ADR"),
        "TotalNights": test["TotalNights"],
        "p_cancel": p_cancel.to_numpy(),
        "y_true": test["IsCanceled"].astype(int).to_numpy(),
        "CutoffDate": test["CutoffDate"],
    })
    # Preserve per-component scores for audit and later weighting experiments.
    for name, series in per_component.items():
        out[f"p_{name}"] = series.to_numpy()

    summary = {
        "hotel": hotel,
        "ensemble_id": finalist["ensemble_id"],
        "components_fitted": list(per_component),
        "components_skipped": skipped,
        "component_detail": details,
        "n_train": int(len(cohorts.train)),
        "n_test": int(len(cohorts.test)),
        "_sklearn_pipelines": sklearn_pipelines,  # internal, used by persist step
        "_fe4_mappings": fe4_mappings,
        "_tabm_artifacts": tabm_artifacts,
    }
    return out, summary


# -----------------------------------------------------------------------------
# Model persistence (for MLOps handoff)
# -----------------------------------------------------------------------------
def _library_versions() -> dict:
    """Capture library versions for reproducibility in metadata.json."""
    def _v(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return "unavailable"
    return {name: _v(name) for name in
            ("numpy", "pandas", "scikit-learn", "xgboost", "torch", "tabm", "pyarrow")}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def persist_hotel_models(
    hotel: str,
    cohorts: RefitCohorts,
    summary: dict,
    models_root: Path,
    model_version: str | None = None,
) -> dict:
    """Save the fitted finalists for ``hotel`` and emit a metadata.json manifest.

    Layout::

        models_root/<model_version>/
            metadata.json
            <component>_pipeline.pkl        <- sklearn ensemble members
            <component>_fe4_mapping.json    <- FE4 segment -> rate (if used)
            <component>_tabm_artifact.json  <- TabM spec + preprocessing recipe

    ``model_version`` defaults to ``<ensemble_id>_v1_<YYYYMMDD>``. The
    metadata manifest records exactly which components exist and how to
    load them, plus library versions and the split fingerprint.
    """
    models_root = Path(models_root)
    ensemble_id = summary["ensemble_id"]
    if model_version is None:
        model_version = f"{ensemble_id}_v1_{cohorts.fit_date.strftime('%Y%m%d')}"

    out_dir = models_root / model_version
    out_dir.mkdir(parents=True, exist_ok=True)

    sklearn_pipelines = summary.pop("_sklearn_pipelines", {})
    fe4_mappings = summary.pop("_fe4_mappings", {})
    tabm_artifacts = summary.pop("_tabm_artifacts", {})

    manifest_components = []
    for cname, pipeline in sklearn_pipelines.items():
        pkl = out_dir / f"{cname}_pipeline.pkl"
        with open(pkl, "wb") as fh:
            pickle.dump(pipeline, fh, protocol=pickle.HIGHEST_PROTOCOL)
        entry = {
            "component": cname,
            "family": summary["component_detail"][cname]["family"],
            "artifact_type": "sklearn_pipeline_pickle",
            "artifact_path": pkl.name,
            "sha256": _file_sha256(pkl),
            "feature_columns": summary["component_detail"][cname]["feature_columns"],
            "weighted_training": (
                summary["component_detail"][cname]["weight_detail"] is not None
            ),
        }
        if cname in fe4_mappings:
            fe4_path = out_dir / f"{cname}_fe4_mapping.json"
            fe4_path.write_text(json.dumps(fe4_mappings[cname], indent=2))
            entry["fe4_mapping_path"] = fe4_path.name
            entry["fe4_mapping_sha256"] = _file_sha256(fe4_path)
        manifest_components.append(entry)

    for cname, tabm_payload in tabm_artifacts.items():
        # Save three TabM files: state dict, preprocessing bundle, architecture/spec
        import torch
        state_dict_path = out_dir / f"{cname}_state_dict.pt"
        preprocess_path = out_dir / f"{cname}_preprocess.json"
        spec_path = out_dir / f"{cname}_spec.json"

        torch.save(tabm_payload["model"].state_dict(), state_dict_path)
        preprocess_path.write_text(json.dumps(tabm_payload["preprocess"], indent=2))
        spec_path.write_text(json.dumps({
            "architecture": tabm_payload["spec"]["architecture"],
            "training_params": tabm_payload["spec"]["training_params"],
            "seed": tabm_payload["spec"]["seed"],
            "pseudo_fold": tabm_payload["pseudo_fold"],
        }, indent=2))

        manifest_components.append({
            "component": cname,
            "family": "TabM",
            "artifact_type": "tabm_state_dict",
            "state_dict_path": state_dict_path.name,
            "state_dict_sha256": _file_sha256(state_dict_path),
            "preprocess_path": preprocess_path.name,
            "preprocess_sha256": _file_sha256(preprocess_path),
            "spec_path": spec_path.name,
            "spec_sha256": _file_sha256(spec_path),
            "architecture": tabm_payload["spec"]["architecture"],
            "note": (
                "Load the state dict into TabM.make(**architecture, "
                "n_num_features=len(preprocess['numeric_columns']), "
                "cat_cardinalities=preprocess['cardinalities'], d_out=1). "
                "See utils/model_scoring.py for the reference loader."
            ),
        })

    metadata_doc = {
        "model_version": model_version,
        "ensemble_id": ensemble_id,
        "hotel": hotel,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fit_date": cohorts.fit_date.strftime("%Y-%m-%d"),
        "cohort_size": summary["n_train"],
        "components": manifest_components,
        "components_skipped": summary["components_skipped"],
        "ensemble_rule": "equal_average of component p_cancel",
        "selected_features": cohorts.selected_features,
        "library_versions": _library_versions(),
        "split_v2_fingerprint": _split_fingerprint(cohorts),
        "scoring_instructions": {
            "preprocessing": (
                "For sklearn pipelines, pass the booking frame with the "
                "columns listed in 'feature_columns' and the pipeline "
                "will apply the fitted preprocessor + model. For C2, "
                "first add the HistoricalSegmentCancelRate feature using "
                "the fe4_mapping JSON (unseen segments -> fallback_rate)."
            ),
            "ensemble": "Return the arithmetic mean of all component p_cancel outputs.",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata_doc, indent=2))
    print(f"  [{hotel}] saved {len(manifest_components)} components -> {out_dir}")
    return {"model_version": model_version, "models_dir": str(out_dir),
            "metadata": metadata_doc}


def _split_fingerprint(cohorts: RefitCohorts) -> str:
    """Fingerprint the Dev training cohort for traceability."""
    keys = (
        cohorts.train[["Hotel", "Booking_ID"]]
        .sort_values(["Hotel", "Booking_ID"])
        .to_csv(index=False, header=False)
        .encode("utf-8")
    )
    return hashlib.sha256(keys).hexdigest()[:16]
