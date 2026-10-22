"""H2 feature upper-bound, label, and cohort-shift diagnostics on development only."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
from utils.model_pipeline import build_pipeline  # noqa: E402

RUN_ID = "h2_three_diagnostics_v1"
OUT = ROOT / "reports" / RUN_ID
GOLD_ROOT = PROJECT / "Data pipeline" / "hotel_booking_pipeline" / "datamart"
SPLIT = ROOT / "splits" / "split_v1" / "H2" / "cv_membership.csv"
REFERENCE = ROOT / "reports" / "feature_ablation_round1_v1" / "run_config.json"
ADVERSARIAL_SEEDS = (17, 42, 73)


def columns(config):
    return [name for group in config.values() for name in group]


def add_features(base, numeric=(), categorical=()):
    updated = {group: list(names) for group, names in base.items()}
    updated["numeric"].extend(numeric)
    updated["categorical"].extend(categorical)
    return updated


def metrics(y, probability):
    return {
        "Rows": len(y),
        "Positives": int(y.sum()),
        "ActualRate": float(y.mean()),
        "MeanProbability": float(np.mean(probability)),
        "ROC_AUC": float(roc_auc_score(y, probability)),
        "AveragePrecision": float(average_precision_score(y, probability)),
        "LogLoss": float(log_loss(y, probability, labels=[0, 1])),
        "BrierScore": float(brier_score_loss(y, probability)),
    }


def get_cohort(membership, gold, fold, role):
    keys = membership.loc[
        membership.Fold.eq(fold) & membership.Role.eq(role), ["Hotel", "Booking_ID"]
    ]
    result = keys.merge(gold, on=["Hotel", "Booking_ID"], validate="one_to_one")
    assert len(result) == len(keys) and result.IsCanceled.notna().all()
    return result


def label_audit(gold):
    labels = pd.read_parquet(GOLD_ROOT / "gold" / "labels" / "gold_label_store_H2.parquet")
    silver = pd.read_parquet(GOLD_ROOT / "silver" / "hotels" / "silver_H2.parquet")
    gold_dates = gold[["Booking_ID", "ArrivalDate", "CutoffDate"]]
    labels = labels.merge(gold_dates, on="Booking_ID", validate="one_to_one")
    for frame in (labels, silver):
        for name in ("ArrivalDate", "CutoffDate", "ReservationStatusDate"):
            frame[name] = pd.to_datetime(frame[name])
        frame["DaysBeforeArrivalAtStatus"] = (
            frame.ArrivalDate - frame.ReservationStatusDate
        ).dt.days

    positives = labels.loc[labels.IsCanceled.eq(1)]
    audit = {
        "gold_rows": len(labels),
        "status_by_target": pd.crosstab(labels.ReservationStatus, labels.IsCanceled).to_dict(),
        "positive_status_timing": {
            "before_arrival_1_to_13_days": int(positives.DaysBeforeArrivalAtStatus.between(1, 13).sum()),
            "on_arrival_day": int(positives.DaysBeforeArrivalAtStatus.eq(0).sum()),
            "on_or_before_cutoff": int((positives.ReservationStatusDate <= positives.CutoffDate).sum()),
        },
    }
    source_scope = silver.loc[silver.LeadTime.ge(14) & silver.OvernightScope]
    terminal = source_scope.loc[source_scope.ReservationStatus.isin(["Canceled", "No-Show"])]
    audit["cutoff_boundary"] = {
        str(days): {
            "eligible": int(terminal.loc[terminal.DaysBeforeArrivalAtStatus.eq(days), "EligibleT14"].sum()),
            "excluded": int((~terminal.loc[terminal.DaysBeforeArrivalAtStatus.eq(days), "EligibleT14"]).sum()),
        }
        for days in (13, 14, 15)
    }
    early = terminal.loc[terminal.ReservationStatusDate.le(terminal.CutoffDate)]
    audit["early_terminal_scope_rows"] = len(early)
    audit["early_terminal_retained_in_gold"] = int(early.Booking_ID.isin(set(labels.Booking_ID)).sum())
    return audit


def univariate_shift(source, future, feature_config, fold, variant):
    records = []
    for feature in columns(feature_config):
        if feature in feature_config["categorical"]:
            a = source[feature].fillna("__MISSING__").astype(str).value_counts(normalize=True)
            b = future[feature].fillna("__MISSING__").astype(str).value_counts(normalize=True)
            levels = a.index.union(b.index)
            distance = float(0.5 * (a.reindex(levels, fill_value=0) - b.reindex(levels, fill_value=0)).abs().sum())
            measure = "total_variation"
        else:
            a = pd.to_numeric(source[feature], errors="coerce").dropna()
            b = pd.to_numeric(future[feature], errors="coerce").dropna()
            distance = float(ks_2samp(a, b).statistic)
            measure = "KS"
        records.append({"Fold": fold, "Variant": variant, "Feature": feature,
                        "Measure": measure, "Distance": distance})
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(REFERENCE.read_text(encoding="utf-8"))["protocol"]
    f1 = protocol["feature_sets"]["F1"]
    booking = add_features(f1,
        numeric=("BookingChanges", "DaysInWaitingList", "ADR", "TotalOfSpecialRequests"),
        categorical=("DepositType",))
    history = add_features(f1,
        numeric=("PreviousCancellations", "PreviousBookingsNotCanceled"))
    trends = add_features(f1,
        numeric=("HotelSearchInterest", "FlightSearchInterest"))
    all_others = add_features(booking,
        numeric=("PreviousCancellations", "PreviousBookingsNotCanceled",
                 "HotelSearchInterest", "FlightSearchInterest"))
    feature_sets = {"F1_core17": f1, "plus_booking_state": booking,
                    "plus_history": history, "plus_trends": trends,
                    "plus_all_others": all_others}
    assert [len(columns(c)) for c in feature_sets.values()] == [17, 22, 19, 19, 26]

    gold = pd.read_parquet(GOLD_ROOT / "gold" / "model_dataset" / "gold_model_dataset_H2.parquet")
    assert gold.Booking_ID.is_unique and gold.Hotel.eq("H2").all()
    gold["CutoffDate"] = pd.to_datetime(gold.CutoffDate)
    membership = pd.read_csv(SPLIT)
    params = protocol["model_params"]["H2"]
    assert params["max_depth"] == 6 and params["min_samples_leaf"] == 50
    label_result = label_audit(gold)
    (OUT / "label_audit.json").write_text(json.dumps(label_result, indent=2), encoding="utf-8")
    print("Label audit saved", flush=True)

    upper_rows = []
    for fold in range(1, 5):
        train = get_cohort(membership, gold, fold, "Train")
        future = get_cohort(membership, gold, fold, "Validation")
        assert not set(train.Booking_ID) & set(future.Booking_ID)
        for name, config in feature_sets.items():
            x = columns(config)
            model = build_pipeline("random_forest", config, params)
            start = perf_counter()
            model.fit(train[x], train.IsCanceled)
            train_p = model.predict_proba(train[x])[:, 1]
            future_p = model.predict_proba(future[x])[:, 1]
            upper_rows.append({"Hotel": "H2", "Fold": fold, "FeatureSet": name,
                               "InputColumns": len(x), "FitSeconds": perf_counter() - start,
                               "TrainROC_AUC": float(roc_auc_score(train.IsCanceled, train_p)),
                               **metrics(future.IsCanceled, future_p)})
            print(f"Upper bound fold {fold} {name}", flush=True)
        pd.DataFrame(upper_rows).to_csv(OUT / "upper_bound_fold_metrics.csv", index=False)

    upper = pd.DataFrame(upper_rows)
    upper_summary = upper.groupby("FeatureSet", as_index=False).agg(
        InputColumns=("InputColumns", "first"),
        LogLoss=("LogLoss", "mean"), BrierScore=("BrierScore", "mean"),
        ROC_AUC=("ROC_AUC", "mean"), AveragePrecision=("AveragePrecision", "mean"),
        TrainROC_AUC=("TrainROC_AUC", "mean"),
    )
    baseline = upper_summary.loc[upper_summary.FeatureSet.eq("F1_core17")].iloc[0]
    upper_summary["AUC_GainVsF1"] = upper_summary.ROC_AUC - baseline.ROC_AUC
    upper_summary.to_csv(OUT / "upper_bound_summary.csv", index=False)

    no_calendar = {group: [name for name in names if name not in {
        "ArrivalDateDayOfMonth", "ArrivalDayOfWeek", "IsPortugalHoliday"}]
        for group, names in f1.items()}
    adv_rows = []
    shift_rows = []
    for fold in range(1, 5):
        eligible = get_cohort(membership, gold, fold, "Train")
        future = get_cohort(membership, gold, fold, "Validation")
        fit_date = pd.Timestamp(membership.loc[membership.Fold.eq(fold), "FitDate"].iloc[0])
        all_candidates = gold.loc[gold.CutoffDate.lt(fit_date)].copy()
        variants = [("eligible_vs_future", eligible, f1),
                    ("all_candidates_vs_future", all_candidates, f1)]
        if fold == 1:
            variants += [("eligible_no_calendar", eligible, no_calendar),
                         ("all_candidates_no_calendar", all_candidates, no_calendar)]
        for variant, source, config in variants:
            x = columns(config)
            shift_rows.extend(univariate_shift(source, future, config, fold, variant))
            combined = pd.concat([source.assign(Source=0), future.assign(Source=1)], ignore_index=True)
            for seed in ADVERSARIAL_SEEDS:
                fit, holdout = train_test_split(combined, test_size=0.25,
                                                random_state=seed, stratify=combined.Source)
                classifier = build_pipeline("random_forest", config, {
                    "n_estimators": 200, "max_depth": 6, "min_samples_leaf": 50,
                    "max_features": "sqrt", "class_weight": "balanced",
                    "random_state": seed, "n_jobs": 2,
                })
                classifier.fit(fit[x], fit.Source)
                probability = classifier.predict_proba(holdout[x])[:, 1]
                adv_rows.append({"Hotel": "H2", "Fold": fold, "Variant": variant,
                                 "Seed": seed, "SourceRows": len(source),
                                 "FutureRows": len(future), "InputColumns": len(x),
                                 "AdversarialAUC": float(roc_auc_score(holdout.Source, probability))})
                print(f"Adversarial fold {fold} {variant} seed {seed}", flush=True)
            pd.DataFrame(adv_rows).to_csv(OUT / "adversarial_seed_metrics.csv", index=False)
    pd.DataFrame(shift_rows).to_csv(OUT / "feature_shift.csv", index=False)
    pd.DataFrame(adv_rows).groupby(["Hotel", "Fold", "Variant"], as_index=False).agg(
        Seeds=("Seed", "nunique"), SourceRows=("SourceRows", "first"),
        FutureRows=("FutureRows", "first"), InputColumns=("InputColumns", "first"),
        AdversarialAUC=("AdversarialAUC", "mean"),
        AdversarialAUC_SD=("AdversarialAUC", "std"),
    ).to_csv(OUT / "adversarial_summary.csv", index=False)
    (OUT / "run_config.json").write_text(json.dumps({
        "run_id": RUN_ID, "hotel": "H2", "split": "split_v1",
        "feature_sets": feature_sets, "cancellation_model_params": params,
        "adversarial_seeds": ADVERSARIAL_SEEDS,
        "adversarial_target": "source cohort, not cancellation outcome",
        "adversarial_excluded": ["Booking_ID", "Hotel", "ArrivalDate", "CutoffDate",
                                 "IsCanceled", "ReservationStatus", "ReservationStatusDate"],
        "final_test_used_for_model_metrics": False,
        "label_audit_scope": "all H2 Gold rows, including final-test-period rows",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }, indent=2), encoding="utf-8")
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
