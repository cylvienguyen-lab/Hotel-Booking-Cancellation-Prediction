# Deployment Handoff — Hotel Cancellation Prediction System

**For:** Aditya (deployment / MLOps)
**From:** Section 4 / 5 (model evaluation + business action)
**Date:** 2026-10-27
**Branch:** `evaluation-&-biz-actions`

A single read-through of this file plus the two reference READMEs
should be enough to pick up the work. If anything's unclear, the two
links in "Where to go next" at the bottom point to the detailed
sources.

---

## TL;DR (one minute)

- The system **scores one cohort of bookings per day** at T-14 (= 14
  days before each booking's arrival) and produces **two batch files**
  for the ops dashboard: a per-booking risk shortlist and a per-night
  hotel-level forecast.
- Models are **already fitted and persisted** under
  `Model pipeline/models/`. The scoring interface is one function:
  `utils.model_scoring.score_bookings(model, bookings_df, as_of_date)`.
- Need to wire the daily batch, feed the two outputs into the
  ops dashboard, and set up monitoring. 
- **Hard pin:** `scikit-learn==1.6.1` in production — matches the
  version that saved the pickles. The project `Dockerfile` already
  pins this; use it or an equivalent image.

---

## 1. What the system delivers (business framing)

Two batch outputs, refreshed daily, feeding one unified ops dashboard:

### Deliverable A — Booking-level risk shortlist

> "Here are the top-K bookings per day that are most likely to cancel.
> Operations contacts them to intervene (wellness-check call, reminder,
> flexibility offer)."

- **Who uses it:** operations / reservations team
- **How often:** daily — one new cohort per day
- **Cohort size:** K per hotel (default **K=10 for H1, K=20 for H2** —
  tune to actual ops contact capacity)
- **Each booking appears on exactly ONE day's list** — the day of its
  T-14 cutoff. There is no re-scoring as T drops toward 0.
- **Columns**: `Hotel`, `Booking_ID`, `CutoffDate`, `ArrivalDate`,
  `ADR`, `TotalNights`, `EstimatedRoomCount`, `p_cancel`,
  `expected_lost_revenue`, `risk_rank`, `model_version`,
  `prediction_as_of_date`, `generated_at_utc`.
- **Example output file (backtest):**
  [`Evaluation pipeline/outputs/booking_shortlist_operational.parquet`](Evaluation%20pipeline/outputs/booking_shortlist_operational.parquet)


### Deliverable B — Hotel-level forecast

> "For the next 2 weeks, here are the expected check-ins and expected
> room-nights occupied per night, per hotel. Housekeeping schedules
> staff and F&B plans menus from this."

- **Who uses it:** housekeeping, F&B, revenue management
- **How often (model):** daily refresh (because new T-14 cohorts arrive
  each day)
- **How often (consumption):** typically weekly or ad-hoc — the
  dashboard rolls up daily files into weekly summaries. Housekeeping
  and F&B don't need to check daily, can check every week or on ad-hocs.
- **Two companion tables:**
  - `arrivals`: per-night check-in count estimate.
  - `occupancy`: per-night room-nights estimate (weights by
    `EstimatedRoomCount`, expands across `TotalNights`).
- **Example output files (backtest):**
  [`Evaluation pipeline/outputs/hotel_forecast_arrivals.parquet`](Evaluation%20pipeline/outputs/hotel_forecast_arrivals.parquet),
  [`Evaluation pipeline/outputs/hotel_forecast_occupancy.parquet`](Evaluation%20pipeline/outputs/hotel_forecast_occupancy.parquet)

### What the system does NOT do

- No direct overbooking policy / automated oversell decisions. The
  dataset lacks a true per-hotel capacity figure. Overbooking is
  scoped as future work (phase 2, with intervention outcomes as new
  features).
- No real-time scoring. Model is T-14 batch only; bookings are not
  re-scored as arrival approaches.

---

## 2. Operational semantics — how our output fuses with PMS reality

**Our forecast is initial 2-week planning, not real-time truth.** It
covers about **89% of future occupancy** (long-lead bookings known at
T-14). The remaining 11% is short-lead/walk-in bookings that didn't
exist when the model scored. Also, we don't re-score as T drops from
14 to 0 — cancellations after T-14 and new short-lead arrivals happen
after our snapshot.

Almost all hotels rely on a PMS (Property Management System) to manage daily bookings, inventory, and guest check-ins. In our solution, PMS carries the real-time operational state, while our model provides the initial predictive baseline. By integrating our model's predictions with the hotel's PMS data, the operational dashboard fuses both sources for the operations user

The operational dashboard combines both:

```
operational_dashboard  =  model_forecast             (ours — ~89% of future planning)
                       +  short_lead_bookings        (PMS: lead < 14 days)
                       +  walk_ins                   (PMS: same-day arrivals)
                       +  current_in_house_guests    (PMS: already checked in)
                       -  cancellations_since_T14    (PMS: happened after our snapshot)
```

Treat our model as **initial planning**; PMS carries real-time state;
the dashboard fuses them for the operations user.

---

## 3. What you receive — file inventory

All paths are relative to the repo root
(`Hotel-Booking-Cancellation-Prediction/`).

### Model registry (what you deploy)

Already fitted on all eligible development data (train + validation
with `FinalRefit == 1`, 15,998 H1 bookings + 29,431 H2 bookings),
seeded for reproducibility.

```
Model pipeline/models/
├── H1-E2_v1_20170418/              <-- H1 finalist bundle
│   ├── metadata.json               <-- manifest: components, SHA-256, library versions, scoring instructions
│   ├── RF30_pipeline.pkl           <-- sklearn Pipeline (RF 300 trees)
│   └── XGB30_pipeline.pkl          <-- sklearn Pipeline (XGBoost)
└── H2-E1_v1_20170418/              <-- H2 finalist bundle
    ├── metadata.json
    ├── C2_pipeline.pkl             <-- sklearn Pipeline (RF + recency weighting)
    ├── C2_fe4_mapping.json         <-- MarketSegment -> historical cancel rate dict
    ├── TabM-E8_state_dict.pt       <-- torch state dict
    ├── TabM-E8_preprocess.json     <-- preprocessing bundle (medians, scaler params, mappings)
    └── TabM-E8_spec.json           <-- architecture + training params
```

Each hotel's ensemble is an **equal-weight average** of its two
components' `p_cancel`.

### Scoring interface (what you call)

[`Model pipeline/utils/model_scoring.py`](Model%20pipeline/utils/model_scoring.py)
— two functions:

- `load_hotel_model(hotel, models_root)` — loads a hotel's complete
  ensemble (sklearn pickles + TabM state_dict rebuilt into a torch
  module) and returns a `LoadedHotelModel` object.
- `score_bookings(model, bookings_df, as_of_date=None)` — returns a
  DataFrame with per-booking `p_cancel` (equal-weight ensemble
  average), audit fields (`model_version`, `prediction_as_of_date`,
  `generated_at_utc`), and per-component `p_<name>` columns for audit.

### Output-generation functions (what builds the dashboards)

[`Model pipeline/utils/model_outputs.py`](Model%20pipeline/utils/model_outputs.py)
— two functions:

- `aggregate_hotel_forecast(scored_bookings)` — returns `{"arrivals":
  ..., "occupancy": ...}` DataFrames for Deliverable B.
- `generate_booking_shortlist(scored_bookings, k_per_hotel, rank_by,
  include_y_true=False)` — returns the top-K daily shortlist for
  Deliverable A. **Set `include_y_true=False` in production** (y_true
  is a backtest-only field).

### Refit code (if you ever need to retrain)

[`Model pipeline/model_refit.ipynb`](Model%20pipeline/model_refit.ipynb)
+ [`Model pipeline/utils/model_refit.py`](Model%20pipeline/utils/model_refit.py).

Refits both finalists from scratch on all `FinalRefit == 1` bookings
and re-publishes the model registry. ~20 minutes CPU (TabM dominates).
You only need this when the training distribution has drifted enough
to require a refresh — see the "Monitoring" section.

### Data pipeline (feature engineering)

Produces the Gold feature frames that the models
consume. Scoring at inference time reads directly from Gold
(`datamart/gold/model_dataset/gold_model_dataset_{H1,H2}.parquet`).
See [`Data pipeline/hotel_booking_pipeline/README.ipynb`](Data%20pipeline/hotel_booking_pipeline/README.ipynb).

### Reference documentation to read next

- [`Evaluation pipeline/README.md`](Evaluation%20pipeline/README.md) —
  full project overview, finalists, metric rationale, deployment
  handoff section (what you receive, how to score, operational
  semantics, monitoring hooks, sklearn-version requirement).
- [`Evaluation pipeline/evaluation_notebook.ipynb`](Evaluation%20pipeline/evaluation_notebook.ipynb)
  — section 12 has the business-action summary; sections 10 and 11
  are the reference implementations of the two output generators.

### Example outputs (backtest, for schema reference)

Everything the production dashboard will consume, computed on the Test
cohort during evaluation:

- Deliverable A: [`Evaluation pipeline/outputs/booking_shortlist_operational.parquet`](Evaluation%20pipeline/outputs/booking_shortlist_operational.parquet) (what ops gets),
  [`booking_shortlist_eval.parquet`](Evaluation%20pipeline/outputs/booking_shortlist_eval.parquet)
  (same + y_true, backtest only).
- Deliverable B: [`hotel_forecast_arrivals.parquet`](Evaluation%20pipeline/outputs/hotel_forecast_arrivals.parquet),
  [`hotel_forecast_occupancy.parquet`](Evaluation%20pipeline/outputs/hotel_forecast_occupancy.parquet),
  [`hotel_forecast_coverage.parquet`](Evaluation%20pipeline/outputs/hotel_forecast_coverage.parquet)
  (actual-vs-predicted coverage per hotel).
- Model quality on Test: [`layer_a_test.parquet`](Evaluation%20pipeline/outputs/layer_a_test.parquet).
- Feature importance (for monitoring): [`feature_importance.parquet`](Evaluation%20pipeline/outputs/feature_importance.parquet).

---

## 4. How to deploy 

### Suggested daily batch pattern

```python
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from utils.model_scoring import load_hotel_model, score_bookings
from utils.model_outputs import (
    aggregate_hotel_forecast,
    generate_booking_shortlist,
)

today = pd.Timestamp.utcnow().normalize()
MODELS_ROOT = Path("Model pipeline/models")

for hotel in ("H1", "H2"):
    # 1. Load the current registered finalist
    model = load_hotel_model(hotel, MODELS_ROOT)

    # 2. Pull today's T-14 cohort from Gold (CutoffDate == today, active)
    gold = pd.read_parquet(
        f"Data pipeline/hotel_booking_pipeline/datamart/gold/model_dataset/gold_model_dataset_{hotel}.parquet"
    )
    bookings_today = gold[(gold.Hotel == hotel) & (gold.CutoffDate == today)]
    if bookings_today.empty:
        continue  # no new T-14 cohort for this date (weekend/holiday pattern)

    # 3. Score — ensemble average p_cancel per booking
    scored = score_bookings(model, bookings_today, as_of_date=today)

    # 4. Build the two deliverables
    forecasts = aggregate_hotel_forecast(scored)
    shortlist = generate_booking_shortlist(
        scored,
        k_per_hotel={"H1": 10, "H2": 20},
        rank_by="p_cancel",         # or "expected_lost_revenue" if business asks
        include_y_true=False,
    )

    # 5. Write to the dashboard store (format of your choice)
    forecasts["arrivals"].to_parquet(f"dashboard/{today:%Y-%m-%d}_{hotel}_arrivals.parquet")
    forecasts["occupancy"].to_parquet(f"dashboard/{today:%Y-%m-%d}_{hotel}_occupancy.parquet")
    shortlist.to_parquet(f"dashboard/{today:%Y-%m-%d}_{hotel}_shortlist.parquet")

    # 6. Log every prediction row for monitoring (preserve model_version,
    #    prediction_as_of_date, generated_at_utc as audit keys).
```

### Dashboard composition

- **Operations shortlist view:** read today's `_shortlist.parquet` for
  both hotels. Join with the PMS or CRM to show contact state.
- **Hotel-level view:** aggregate the last N days of `_arrivals.parquet`
  and `_occupancy.parquet` into a 2-week rolling forward window per
  hotel, with weekly summaries and ad-hoc drill-down on specific
  nights. Join with the PMS to add short-lead bookings, walk-ins, and
  in-house guests (see "Operational semantics" above).

### Infrastructure requirements

- **Python 3.12** (project's Dockerfile base).
- **Hard-pinned libraries** (see [`Model pipeline/requirements.txt`](Model%20pipeline/requirements.txt)):
  - `scikit-learn==1.6.1` — this is a hard requirement; different
    sklearn versions can't unpickle the saved pipelines.
  - `xgboost-cpu==3.0.5` (XGBoost 3.x)
  - `torch==2.8.0` + `tabm==0.0.3` (TabM-E8 scoring for H2)
  - `pandas==2.2.3`, `numpy==2.2.2`, `pyarrow==26.0.0`
- The project `Dockerfile` already assembles this. The simplest deploy
  is a thin image built from the same base.


---

## 5. Monitoring hooks (ship with the service)

Watch these over the first 30 days of production and alert on drift:

### Data quality
- Feature-column availability on every batch (no missing columns, no
  silent type changes in Gold).
- Daily booking volume (sudden drops mean upstream pipeline problem).
- `MarketSegment` distribution (C2's FE4 feature maps from this; new
  segments fall through to the fallback rate).

### Model quality (compute weekly, once outcomes are observed)
- **Prevalence drift** — mean actual cancellation rate per hotel per
  week. Training-time was 7-13%; the Test window already showed drift
  (H2 train 13% → test 8%). Alert if `|drift|` > 3pp.
- **Daily nRMSE** on realised outcomes — compare to the Test baseline
  (H1 ≈ 0.83, H2 ≈ 0.70). Alert if nRMSE > 1.5× baseline.
- **Daily Bias %** — alert if `|bias|` > 15% for two weeks running.
- **AP on the shortlist outcome** — compute monthly. Expected
  ~0.19 (H1) and ~0.28 (H2). Alert if AP drops below 0.10.
- **Shortlist Precision@K** at the deployed K. Expected 13% (H1 @ K=10),
  14% (H2 @ K=20) on test. Alert if precision drops below 8%.

### Coverage drift
- Monitor the Gold/Silver ratio per hotel. Training-time this was
  ~89%. If the share of short-lead bookings grows, our forecast's
  slice of total occupancy shrinks, and the PMS integration adjustment
  needs re-tuning.

### Top-feature population drift
- `Evaluation pipeline/outputs/feature_importance.parquet` lists the
  top features per sklearn component. Monitor their population
  statistics (mean, variance, missingness) in production. If a top
  feature's distribution shifts materially, flag for retraining.

---

## 6. Known limitations and caveats

- **sklearn version pin.** The sklearn pipelines are tied to 1.6.1 by
  pickle format. Upgrading sklearn requires a refit. This is a
  documented hard requirement, not a bug.
- **Short-lead bookings (lead < 14 days)** are not scored by this
  model. They're ~11% of actual occupancy on average and must be
  handled by the PMS integration on the dashboard side.
- **TabM-E8's reproducibility assumes torch 2.8.0.** A future torch
  upgrade might break state_dict loading; if so, refit from the saved
  spec. The spec includes the exact architecture and seed.
- **No automated overbooking policy.** Operations uses our outputs to
  inform decisions, not to automate them. The 2nd-stage overbooking
  model is scoped as future work (phase 2, once outreach outcomes have
  accumulated as features).
- **Model is a snapshot per inference day.** Cancellations after T-14
  and bookings added after T-14 are not reflected in our output; they
  live in the PMS.

---

## 7. Where to go next

- [`Evaluation pipeline/README.md`](Evaluation%20pipeline/README.md) —
  deep-dive on finalists, metrics, calibration, operational semantics,
  and the full deployment handoff section.
- [`Evaluation pipeline/evaluation_notebook.ipynb`](Evaluation%20pipeline/evaluation_notebook.ipynb)
  — the end-to-end evaluation that produced the model choices. Sections
  10-12 are the reference implementation of the two deliverables.
- [`Model pipeline/model_refit.ipynb`](Model%20pipeline/model_refit.ipynb)
  — the refit flow (how the model registry was produced).
- Model registry: [`Model pipeline/models/`](Model%20pipeline/models/).

