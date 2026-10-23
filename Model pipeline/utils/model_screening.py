"""Independent, versioned screening runs on the saved split_v2 cohorts.

Each run changes one intervention only. This module reads Development rows,
fits all transformations inside an outer training fold, and saves the same
three-file metric/prediction contract as the initial experiments.
"""

import json
import os
import time
import warnings
from collections import defaultdict
from importlib import metadata
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.model_preprocessing import build_preprocessor


KEYS = ["Hotel", "Booking_ID"]
SPLIT_ID = "split_v2"
ARTIFACT_FILES = {"config.json", "metrics.parquet", "predictions.parquet"}


def _version(package: str) -> str:
    """Return an installed distribution version for artifact provenance."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _cohorts(
    model_data: pd.DataFrame,
    membership: pd.DataFrame,
    hotel: str,
    base_features: list[str],
    fold_number: int,
    fold_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Retrieve exact Rule B Train and saved ValFold cohorts for one hotel.

    Inputs contain Development rows only. Validation is the full saved fold;
    the Complete Follow-up Rule applies only to training eligibility.
    """
    if model_data.duplicated(KEYS).any() or membership.duplicated(KEYS).any():
        raise ValueError("Duplicate booking keys in Development inputs")
    if not membership.Partition.eq("Dev").all():
        raise ValueError("Screening membership must contain Development only")
    fields = KEYS + base_features + ["IsCanceled"]
    source = model_data.loc[model_data.Hotel.eq(hotel), fields]
    member = membership.loc[membership.Hotel.eq(hotel)]
    data = member.merge(source, on=KEYS, how="left", validate="one_to_one")
    if len(data) != len(member) or data.IsCanceled.isna().any():
        raise ValueError("Development feature/label join is incomplete")
    if not data.ExpectedCheckoutDate.eq(
        data.ArrivalDate + pd.to_timedelta(data.TotalNights, unit="D")
    ).all():
        raise ValueError("Saved expected checkout differs from Gold stay length")
    fit_date = pd.Timestamp(fold_config["fit_date"])
    train = data.loc[
        data.CutoffDate.lt(fit_date)
        & data.LabelAvailableDate.lt(fit_date)
        & data.ExpectedCheckoutDate.lt(fit_date)
    ].copy().reset_index(drop=True)
    val = data.loc[data.ValFold.eq(fold_number).fillna(False)].copy().reset_index(drop=True)
    if train.empty or val.empty or train.IsCanceled.nunique() != 2:
        raise ValueError(f"{hotel} fold {fold_number}: invalid cohort")
    if not val.CutoffDate.ge(fit_date).all() or not val.CutoffDate.lt(
        pd.Timestamp(fold_config["val_end"])
    ).all():
        raise ValueError("Validation cohort outside saved calendar")
    if set(train.Booking_ID).intersection(val.Booking_ID):
        raise ValueError("Train and Validation booking overlap")
    return train, val, fit_date


def _chronological_segment_rate(
    train: pd.DataFrame,
    val: pd.DataFrame,
    smoothing: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode each training row from strictly earlier completed outcomes.

    An outcome enters history only after both its label-availability date and
    expected checkout. Rows sharing a CutoffDate see the same prior history;
    their own labels and later labels are excluded. The outer Validation map
    uses all eligible Train labels and falls back to its global Train rate.
    """
    segment = train.MarketSegment.fillna("__MISSING__").astype(str)
    val_segment = val.MarketSegment.fillna("__MISSING__").astype(str)
    available = train[["LabelAvailableDate", "ExpectedCheckoutDate"]].max(axis=1)
    if not available.ge(train.CutoffDate).all():
        raise ValueError("Historical outcome availability precedes booking cutoff")
    events = pd.DataFrame({
        "available": available, "segment": segment,
        "label": train.IsCanceled.astype(int),
    }).sort_values("available")
    event_rows = list(events.itertuples(index=False))
    counts = defaultdict(int)
    positives = defaultdict(int)
    global_n = global_positive = cursor = 0
    encoded = np.empty(len(train), dtype=float)
    for cutoff, positions in train.groupby("CutoffDate", sort=True).groups.items():
        while cursor < len(event_rows) and event_rows[cursor].available < cutoff:
            event = event_rows[cursor]
            counts[event.segment] += 1
            positives[event.segment] += int(event.label)
            global_n += 1
            global_positive += int(event.label)
            cursor += 1
        historical_rate = global_positive / global_n if global_n else 0.5
        for position in positions:
            name = segment.iloc[position]
            encoded[position] = (
                positives[name] + smoothing * historical_rate
            ) / (counts[name] + smoothing)
    train_global_rate = float(train.IsCanceled.mean())
    final = train.assign(_segment=segment).groupby("_segment").IsCanceled.agg(
        ["sum", "count"]
    )
    mapping = (
        (final["sum"] + smoothing * train_global_rate)
        / (final["count"] + smoothing)
    ).to_dict()
    validation_encoded = val_segment.map(mapping).fillna(train_global_rate).to_numpy()
    return encoded, validation_encoded


def _engineer(
    train: pd.DataFrame,
    val: pd.DataFrame,
    kind: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    """Add one fold-fitted feature intervention to copies of Train and Val."""
    train, val = train.copy(), val.copy()
    added: list[str] = []
    detail: dict = {}
    if kind is None:
        return train, val, added, detail
    train_segment = train.MarketSegment.fillna("__MISSING__").astype(str)
    val_segment = val.MarketSegment.fillna("__MISSING__").astype(str)
    if kind == "FE1":
        medians = train.assign(_segment=train_segment).groupby("_segment").LeadTime.median()
        fallback = float(train.LeadTime.median())
        name = "RelativeLeadTime"
        train[name] = train.LeadTime - train_segment.map(medians).fillna(fallback)
        val[name] = val.LeadTime - val_segment.map(medians).fillna(fallback)
        added = [name]
        detail = {"segment_medians": medians.to_dict(), "fallback": fallback}
    elif kind == "FE2":
        # Category definitions come from this fold's eligible Train only.
        categories = sorted(train_segment.unique())
        for index, category in enumerate(categories):
            name = f"SegmentLeadTime_{index}"
            train[name] = train.LeadTime * train_segment.eq(category).astype(int)
            val[name] = val.LeadTime * val_segment.eq(category).astype(int)
            added.append(name)
        detail = {"interaction_categories": dict(zip(added, categories))}
    elif kind == "FE3":
        for source, name in (
            ("LeadTime", "GroupsLeadTime"),
            ("TotalNights", "GroupsTotalNights"),
            ("PartySize", "GroupsPartySize"),
        ):
            train[name] = train_segment.eq("Groups").astype(int) * train[source]
            val[name] = val_segment.eq("Groups").astype(int) * val[source]
            added.append(name)
        detail = {"group_value": "Groups"}
    elif kind == "FE4":
        name = "HistoricalSegmentCancelRate"
        train[name], val[name] = _chronological_segment_rate(train, val)
        added = [name]
        detail = {
            "smoothing": 20.0,
            "training_history": (
                "CutoffDate, LabelAvailableDate and ExpectedCheckoutDate "
                "strictly earlier than encoded row CutoffDate"
            ),
            "validation_map": "all eligible training labels; unseen=training overall rate",
        }
    else:
        raise ValueError(f"Unknown feature experiment: {kind}")
    return train, val, added, detail


def _catboost_input(
    train: pd.DataFrame,
    val: pd.DataFrame,
    selected_features: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], dict]:
    """Prepare CatBoost's native categorical frame using Train medians only."""
    columns = sum((selected_features[group] for group in ("numeric", "binary", "categorical")), [])
    numeric = selected_features["numeric"] + selected_features["binary"]
    categorical = selected_features["categorical"]
    train_x, val_x = train[columns].copy(), val[columns].copy()
    medians = train_x[numeric].apply(pd.to_numeric, errors="coerce").median().fillna(0)
    for frame in (train_x, val_x):
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce").fillna(medians)
        frame[categorical] = frame[categorical].fillna("__MISSING__").astype(str)
    return train_x, val_x, [columns.index(name) for name in categorical], {
        "numeric_imputation": "eligible Train median, all-missing fallback 0",
        "categorical_missing": "__MISSING__",
        "categorical_handling": "CatBoost native",
    }


def _fit_predict_sklearn(
    train: pd.DataFrame,
    val: pd.DataFrame,
    selected_features: dict,
    feature_contract: dict,
    spec: dict,
    added: list[str],
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit RF/LR/LightGBM/CatBoost and return unweighted probabilities."""
    family = spec["family"]
    params = spec["estimator_params"]
    y_train = train.IsCanceled.astype(int)
    started = time.perf_counter()
    detail = {"convergence_warnings": []}
    if family == "CatBoost":
        from catboost import CatBoostClassifier

        train_x, val_x, cat_indices, preprocessing = _catboost_input(
            train, val, selected_features
        )
        model = CatBoostClassifier(**params)
        model.fit(train_x, y_train, cat_features=cat_indices, verbose=False)
        p_train = model.predict_proba(train_x)[:, 1]
        p_val = model.predict_proba(val_x)[:, 1]
        detail["preprocessing"] = preprocessing
    else:
        if family == "RF":
            estimator = RandomForestClassifier(**params)
            transform_model = "RF"
        elif family == "LR":
            estimator = LogisticRegression(**params)
            transform_model = "LR"
        elif family == "LightGBM":
            from lightgbm import LGBMClassifier

            estimator = LGBMClassifier(**params)
            transform_model = "RF"
        else:
            raise ValueError(f"Unknown sklearn screening family: {family}")
        preprocessor = build_preprocessor(
            selected_features, feature_contract, transform_model,
            engineered_numeric_features=tuple(added),
        )
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)), ("model", estimator)
        ])
        columns = sum((selected_features[group] for group in ("numeric", "binary", "categorical")), [])
        fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(train[columns], y_train, **fit_kwargs)
        detail["convergence_warnings"] = [
            str(item.message) for item in caught
            if issubclass(item.category, ConvergenceWarning)
        ]
        classes = pipeline.named_steps["model"].classes_.tolist()
        if classes != [0, 1]:
            raise ValueError("Probability class order is not [0, 1]")
        p_train = pipeline.predict_proba(train[columns])[:, 1]
        p_val = pipeline.predict_proba(val[columns])[:, 1]
        detail["preprocessing"] = (
            "fold-trained Gold numeric/binary imputation and one-hot categories; "
            + ("LR numeric scaling" if family == "LR" else "unscaled numeric")
        )
        detail["weighted_estimator_fit"] = weights is not None
    detail["training_seconds"] = time.perf_counter() - started
    return p_train, p_val, detail


def _fit_predict_tabm(
    train: pd.DataFrame,
    val: pd.DataFrame,
    selected_features: dict,
    spec: dict,
    fold_number: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train official TabM on one outer fold with train-fitted input mappings.

    Every ensemble member receives its own binary cross-entropy loss. During
    inference sigmoid is applied to each member before probabilities are
    averaged. A fixed epoch budget avoids outer-Validation early stopping.
    """
    import torch
    import torch.nn.functional as F
    from tabm import TabM

    params = spec["training_params"]
    seed = int(spec["seed"]) + fold_number
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    numeric = selected_features["numeric"] + selected_features["binary"]
    categorical = selected_features["categorical"]
    medians = train[numeric].apply(pd.to_numeric, errors="coerce").median().fillna(0)
    scaler = StandardScaler()
    x_train_num = scaler.fit_transform(
        train[numeric].apply(pd.to_numeric, errors="coerce").fillna(medians)
    ).astype("float32")
    x_val_num = scaler.transform(
        val[numeric].apply(pd.to_numeric, errors="coerce").fillna(medians)
    ).astype("float32")
    mappings = []
    for column in categorical:
        observed = train[column].fillna("__MISSING__").astype(str)
        categories = sorted(set(observed) - {"__MISSING__"})
        mappings.append({name: index for index, name in enumerate(["__MISSING__"] + categories)})

    def encode_categories(frame: pd.DataFrame) -> np.ndarray:
        """Map missing/unseen values to index zero inside Train cardinality."""
        return np.column_stack([
            frame[column].fillna("__MISSING__").astype(str)
            .map(mapping).fillna(0).astype("int64").to_numpy()
            for column, mapping in zip(categorical, mappings)
        ])

    x_train_cat = encode_categories(train)
    x_val_cat = encode_categories(val)
    cardinalities = [len(mapping) for mapping in mappings]
    if any(x_train_cat[:, index].max() >= size for index, size in enumerate(cardinalities)):
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
        model.parameters(), lr=params["learning_rate"],
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

    def infer(numeric_array: np.ndarray, categorical_array: np.ndarray) -> np.ndarray:
        """Average sigmoid probabilities, never member logits."""
        model.eval()
        pieces = []
        with torch.no_grad():
            for start in range(0, len(numeric_array), batch_size):
                stop = start + batch_size
                num = torch.from_numpy(numeric_array[start:stop]).to(device)
                cat = torch.from_numpy(categorical_array[start:stop]).to(device)
                logits = model(num, cat).squeeze(-1)
                pieces.append(torch.sigmoid(logits).mean(dim=1).cpu().numpy())
        return np.concatenate(pieces)

    p_train = infer(x_train_num, x_train_cat)
    p_val = infer(x_val_num, x_val_cat)
    detail = {
        "preprocessing": (
            "Train median imputation + StandardScaler for numeric/binary; "
            "Train-only categorical integer mapping; missing/unseen index 0"
        ),
        "categorical_cardinalities": cardinalities,
        "device": str(device), "optimizer": "AdamW",
        "architecture": architecture, "training_params": params,
        "stopping": "fixed epoch budget; no outer-Validation monitoring",
        "train_loss_by_epoch": loss_history,
        "training_seconds": time.perf_counter() - started,
    }
    return p_train, p_val, detail


def _score(
    cohort: pd.DataFrame,
    probabilities: np.ndarray,
    experiment_id: str,
    family: str,
    fold_number: int,
    split_name: str,
) -> dict:
    """Apply the initial experiment's Log Loss and ArrivalDate daily metrics."""
    y = cohort.IsCanceled.astype(int).to_numpy()
    p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid score probabilities")
    row = {
        "experiment_id": experiment_id, "split_id": SPLIT_ID,
        "hotel": cohort.Hotel.iloc[0], "model_name": family,
        "fold": fold_number, "split": split_name,
        "n_bookings": len(cohort), "n_positive": int(y.sum()),
        "n_days": pd.NA, "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": np.nan, "daily_bias_pct": np.nan,
        "daily_nrmse": np.nan, "metric_warning": "",
    }
    if split_name == "val":
        if len(np.unique(y)) == 2:
            row["roc_auc"] = float(roc_auc_score(y, p))
        else:
            row["metric_warning"] = "ROC-AUC undefined: one validation class"
        daily = pd.DataFrame({
            "ArrivalDate": cohort.ArrivalDate.to_numpy(),
            "predicted": p, "actual": y,
        }).groupby("ArrivalDate", as_index=False).sum()
        row["n_days"] = len(daily)
        actual_total = daily.actual.sum()
        if actual_total > 0:
            error = daily.predicted - daily.actual
            row["daily_bias_pct"] = float(100 * error.sum() / actual_total)
            row["daily_nrmse"] = float(
                np.sqrt(np.mean(error**2)) / daily.actual.mean()
            )
        else:
            row["metric_warning"] += "; daily metrics undefined: zero actual cancellations"
    return row


def _verify_saved(
    target: Path,
    request: dict,
    model_data: pd.DataFrame,
    membership: pd.DataFrame,
    base_features: list[str],
) -> dict:
    """Reject incomplete/mismatched runs and recheck OOF labels and metrics."""
    if {path.name for path in target.iterdir()} != ARTIFACT_FILES:
        raise ValueError(f"Incomplete screening artifact: {target}")
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    if config["request"] != request:
        raise ValueError(f"Screening identity changed; use a new ID: {target.name}")
    metrics = pd.read_parquet(target / "metrics.parquet")
    predictions = pd.read_parquet(target / "predictions.parquet")
    hotel = request["hotel"]
    expected = membership.loc[
        membership.Hotel.eq(hotel) & membership.ValFold.notna(),
        KEYS + ["ValFold", "CutoffDate", "ArrivalDate"],
    ].merge(
        model_data.loc[model_data.Hotel.eq(hotel), KEYS + ["IsCanceled"]],
        on=KEYS, validate="one_to_one",
    )
    if len(predictions) != len(expected) or predictions.duplicated(KEYS).any():
        raise ValueError("OOF booking count/uniqueness differs")
    aligned = predictions.merge(expected, on=KEYS, validate="one_to_one", suffixes=("", "_expected"))
    if (
        len(aligned) != len(expected)
        or not aligned.fold.eq(aligned.ValFold).all()
        or not aligned.y_true.eq(aligned.IsCanceled).all()
        or not aligned.CutoffDate.eq(aligned.CutoffDate_expected).all()
        or not aligned.StayDate.eq(aligned.ArrivalDate).all()
        or not predictions.split_id.eq(SPLIT_ID).all()
        or not predictions.Hotel.eq(hotel).all()
        or not predictions.experiment_id.eq(request["experiment_id"]).all()
    ):
        raise ValueError("OOF predictions differ from saved Development membership")
    if len(metrics) != 8 or metrics.groupby("split").size().to_dict() != {"train": 4, "val": 4}:
        raise ValueError("Incomplete fold metrics")
    for fold_number in range(1, 5):
        scored = predictions.loc[predictions.fold.eq(fold_number)]
        metric = metrics.loc[
            metrics.fold.eq(fold_number) & metrics.split.eq("val")
        ].iloc[0]
        assert int(metric.n_bookings) == len(scored)
        assert np.isclose(metric.log_loss, log_loss(scored.y_true, scored.p_cancel, labels=[0, 1]))
        daily = scored.groupby("StayDate").agg(
            predicted=("p_cancel", "sum"), actual=("y_true", "sum")
        )
        error = daily.predicted - daily.actual
        assert np.isclose(metric.daily_bias_pct, 100 * error.sum() / daily.actual.sum())
        assert np.isclose(metric.daily_nrmse, np.sqrt(np.mean(error**2)) / daily.actual.mean())
        assert np.isclose(metric.roc_auc, roc_auc_score(scored.y_true, scored.p_cancel))
    return {"config": config, "metrics": metrics, "predictions": predictions}


def run_screening_experiment(
    *,
    experiment_id: str,
    hotel: str,
    spec: dict,
    model_data: pd.DataFrame,
    membership: pd.DataFrame,
    selected_features: dict,
    feature_contract: dict,
    split_config: dict,
    output_root: Path,
    source_fingerprint: str,
) -> dict:
    """Run or verify one versioned four-fold Development experiment.

    Inputs are Gold Development rows and saved split_v2 Development membership.
    Existing IDs are loaded only after exact request and OOF verification. New
    runs publish config, fold metrics and OOF predictions atomically.
    """
    output_root = Path(output_root)
    if hotel not in {"H1", "H2"} or not experiment_id.startswith(f"{hotel}_"):
        raise ValueError("Hotel and versioned experiment ID differ")
    if not experiment_id.endswith(("_s1", "_c1")):
        raise ValueError("Experiment ID must end with _s1 or _c1")
    base_features = sum((selected_features[group] for group in ("numeric", "binary", "categorical")), [])
    if len(base_features) != 30 or len(set(base_features)) != 30:
        raise ValueError("Expected 30 unique Gold predictors")
    if not membership.Partition.eq("Dev").all():
        raise ValueError("Screening may only access Development membership")
    if not pd.to_datetime(model_data.CutoffDate).lt(
        pd.Timestamp(split_config["test_start"])
    ).all():
        raise ValueError("Screening model data must exclude Test rows")
    request = {
        "experiment_id": experiment_id, "hotel": hotel,
        "split_id": SPLIT_ID, "source_fingerprint": source_fingerprint,
        "split_calendar": split_config, "base_features": selected_features,
        "spec": spec,
    }
    target = output_root / experiment_id
    if target.exists():
        result = _verify_saved(target, request, model_data, membership, base_features)
        result["status"] = "loaded and verified"
        return result

    metric_rows = []
    prediction_parts = []
    fold_details = []
    for fold_number, fold in enumerate(split_config["folds"], start=1):
        train, val, fit_date = _cohorts(
            model_data, membership, hotel, base_features, fold_number, fold
        )
        train, val, added, feature_detail = _engineer(train, val, spec.get("feature_experiment"))
        roles = {
            "numeric": selected_features["numeric"] + added,
            "binary": selected_features["binary"],
            "categorical": selected_features["categorical"],
        }
        weight_detail = None
        weights = None
        if spec.get("half_life_days") is not None:
            age_days = (fit_date - train.CutoffDate).dt.days.to_numpy()
            weights = np.power(0.5, age_days / spec["half_life_days"])
            weights = weights / weights.mean()
            weight_detail = {
                "train_n": len(weights), "min_weight": float(weights.min()),
                "max_weight": float(weights.max()),
                "mean_weight": float(weights.mean()),
                "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
            }
        if spec["family"] == "TabM":
            p_train, p_val, fit_detail = _fit_predict_tabm(
                train, val, roles, spec, fold_number
            )
        else:
            p_train, p_val, fit_detail = _fit_predict_sklearn(
                train, val, roles, feature_contract, spec, added, weights
            )
        metric_rows.extend([
            _score(train, p_train, experiment_id, spec["family"], fold_number, "train"),
            _score(val, p_val, experiment_id, spec["family"], fold_number, "val"),
        ])
        predictions = val[KEYS + ["CutoffDate", "ArrivalDate"]].rename(
            columns={"ArrivalDate": "StayDate"}
        ).copy()
        predictions["fold"] = fold_number
        predictions["y_true"] = val.IsCanceled.astype(int).to_numpy()
        predictions["p_cancel"] = p_val
        predictions["experiment_id"] = experiment_id
        predictions["model_name"] = spec["family"]
        predictions["split_id"] = SPLIT_ID
        prediction_parts.append(predictions)
        fold_details.append({
            "fold": fold_number, "train_n": len(train), "val_n": len(val),
            "fit_date": str(fit_date.date()), "added_features": added,
            "feature_detail": feature_detail, "weights": weight_detail,
            "fit_detail": fit_detail,
        })

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    config = {
        "request": request,
        "fold_details": fold_details,
        "package_versions": {
            name: _version(name) for name in
            ("numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "torch", "tabm")
        },
        "training_rule": (
            "Dev; CutoffDate < FitDate; LabelAvailableDate < FitDate; "
            "ExpectedCheckoutDate < FitDate"
        ),
        "metric_rule": "unweighted; daily groups are ArrivalDate within each ValFold",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{experiment_id}.{uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        (temporary / "config.json").write_text(
            json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        metrics.to_parquet(temporary / "metrics.parquet", index=False)
        predictions.to_parquet(temporary / "predictions.parquet", index=False)
        os.rename(temporary, target)
    except Exception:
        # Retain incomplete temporary evidence for diagnosis; never replace an ID.
        raise
    result = _verify_saved(target, request, model_data, membership, base_features)
    result["status"] = "created and verified"
    return result
