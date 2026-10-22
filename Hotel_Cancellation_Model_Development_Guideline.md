# Hotel Cancellation Model Development: Step-by-Step Guideline

**Working version:** 20 October 2026  
**Scope:** Cylvie and Allyson's model development and evaluation work, from Gold handoff to a scoring package for Aditya.  
**Status:** Proposed workflow. Split dates, final feature sets, model family, calibration design, business review capacity, and the final inference feature contract still require decisions based on the complete Gold data.

## 0. ML problem and prediction contract

### What are we trying to predict?

At the **end of the day 14 days before scheduled arrival (T−14)**, score each booking that exists and is still active. The target is whether that booking **cancels after T−14 or becomes a no-show**, versus completing the stay.

- **Unit of prediction:** one eligible booking at its `CutoffDate = ArrivalDate − 14 days`.
- **Input `X`:** booking and contextual features that could be known at `CutoffDate` and reproduced when scoring a future booking.
- **Historical label `y = IsCanceled`:** `1` for later cancellation/no-show; `0` for completed stay. The label is used for training and evaluation, never supplied with a new booking for inference.
- **Model output:** a calibrated probability or risk score for each eligible booking, such as `P(later cancellation or no-show | information at T−14) = 0.32`.
- **Next business layer:** a separately agreed decision rule (for example top-*k* bookings for review or a threshold), followed by hotel action and outcome measurement. A score does not establish that contacting a guest will prevent cancellation.

**Example:** a booking arrives on 20 October, so its `CutoffDate` is 6 October. Cancellation on 17 October gives label 1. Cancellation on 4 October makes the booking ineligible because it is already gone at scoring time. A completed stay gives label 0 when the outcome is known. For a completed stay, label availability may be later than arrival.

**Critical limitation:** the source records contain the booking's eventual state, not genuine historical snapshots at T−14. A reconstructed T−14 row can therefore contain a value changed after the cutoff. Feature availability must be assessed separately from predictive power. The Gold feature contract lists **30 candidate predictors**, not 30 automatically approved final predictors.

### Two clocks to keep distinct

| Clock | Field or rule | Use |
|---|---|---|
| Prediction time | `CutoffDate` | Determine which booking is scored on a given day and which chronological period its row belongs to. |
| Outcome availability | When cancellation/no-show/completed stay became known, checked using the outcome/audit dates | Determine whether a historical label was available by a simulated model fit date. |

Split by **prediction time**. At each simulated fit date, also require **outcomes already known**. `ArrivalDate` is always 14 days after `CutoffDate` for this sample, so their order is identical with shifted boundaries, but `CutoffDate` expresses the operational prediction time. Do not use the outcome date as a predictor.

### Scope of this guideline

The sequence below covers the **training and evaluation workflow**. Aditya's daily batch inference loads the resulting fitted package and scores new, unlabeled, eligible bookings; it does not retrain or recalibrate the model. The current supplied ZIP includes the feature contract, README, raw/Bronze inputs, and a Gold feature directory, but **does not include the complete H1/H2 Gold model datasets described in the README**. Obtain the complete latest handoff before calculating splits or metrics. README counts are reference checks, not substitutes for the handed-off tables.

---

## Workflow at a glance

1. **Receive and audit Gold** → a checked modelling input and feature eligibility register.
2. **Design experiments and temporal splits** → frozen development/test boundaries, feature sets, baseline, metrics.
3. **Run candidate experiments** → reproducible model, feature-set, preprocessing, and tuning comparisons.
4. **Review errors and revise features where justified** → shortlist or versioned feature change and rerun.
5. **Select single model or ensemble** → one frozen solution specification.
6. **Refit and calibrate** → one fitted chain that returns calibrated probabilities.
7. **Final Test (OOT)** → independent evaluation and promote/hold recommendation.
8. **Package and hand off** → runnable inference artifact, contract, examples, and model report.

Two loops are allowed **before the final test**: trying another experiment, and updating a feature following a specific development-data finding. Do not optimize repeatedly on the same final test.

---

## Stage 1 — Receive and audit Gold

**Input:** H1/H2 Gold features and labels (or already joined Gold modelling datasets), `feature_contract.json`, processing rules, version/run information.

### 1.1 Establish the handoff

- [ ] Locate the complete H1/H2 Gold tables and record their version, location, row counts, and schema.
- [ ] Identify whether features and labels are separate. If a joined model dataset already exists, inspect it instead of joining again. Otherwise join on `Hotel + Booking_ID`.
- [ ] Assert unique keys in each input, a one-to-one join, no unintended lost or multiplied bookings, and no missing labels in joined rows.

**Result:** versioned H1/H2 modelling tables plus a join reconciliation. A key/join failure stops training; do not repair it by arbitrary deduplication.

### 1.2 Verify prediction population and label

- [ ] Confirm `CutoffDate = ArrivalDate − 14 days`, `LeadTime ≥ 14`, `PartySize > 0`, and `TotalNights > 0` in the Gold scope.
- [ ] Confirm early cancellations/no-shows at or before the cutoff are excluded. Confirm retained late cancellations/no-shows are label 1 and completed stays label 0 using label/audit information where available.
- [ ] Confirm `IsCanceled` contains only 0/1 and is not a predictor. Keep `ReservationStatus` and `ReservationStatusDate` only for outcome/scope auditing, never in `X`.
- [ ] Compare actual counts with the README's reported H1 20,140 rows/1,518 positives and H2 38,233 rows/4,536 positives. Investigate any difference as a version or processing issue; never force counts to match.

**Result:** a pass/fail scope-and-label report. Do not infer label timing from `CutoffDate` alone.

### 1.3 Audit schema and basic quality

- [ ] Separate predictors, target, and metadata. Check each column against the 23 numeric + 7 categorical candidate predictors, 10 metadata fields, and target in the contract.
- [ ] Profile data types, null counts, category levels/rare or unexpected values, numerical ranges, and H1/H2 differences. The Gold contract deliberately retains documented missing values and categorical text.
- [ ] Summarize booking volume, positive rate, and `CutoffDate` range by hotel and month. Flag unusual early H2 months documented in EDA for sensitivity checks.
- [ ] Record preprocessing needs (for example imputation or encoding) without fitting an imputer/scaler/encoder on the full Gold dataset.

**Result:** schema/quality profile and a list of modelling preprocessing needs. This verifies the interface; it does not redo Bronze/Silver cleaning.

### 1.4 Create a feature eligibility register

For **every candidate feature**, document: meaning, source, whether its value was known at T−14, whether the value in Gold really represents that time, whether Aditya can generate it again, missingness, and a decision with a reason.

Use statuses such as **verified/defensible**, **uncertain—investigate or sensitivity experiment**, **exclude**, and **inference reproduction unresolved**. Pay particular attention to `BookingChanges`, `DaysInWaitingList`, `ADR`, `TotalOfSpecialRequests`, and the retrospective scaling/availability of Google Trends. Source status is determined by timing and provenance, not by correlation or feature importance.

**Gate to Stage 2:** input keys, labels, population, and schema are sound; open issues are recorded; candidate features have explicit eligibility decisions. Unresolved risky features may be explored for sensitivity but cannot silently become production inputs.

**Suggested work products:** `audit_summary.md`, `feature_register.csv`, `gold_contract_checks.json`, a dataset version/hash or pipeline run ID.

---

## Stage 2 — Design experiments and temporal evaluation

**Input:** audited Gold, feature register, monthly booking/positive counts.

### 2.1 Define feature sets with hypotheses

- [ ] Start with a defensible set of features available and reproducible at T−14. Call it the **core set**; it is a timing/reproducibility classification, not a feature-importance ranking.
- [ ] Add only purposeful alternatives: core + verified external variables; core + one group whose availability is under investigation; or a smaller/simple set if complexity and robustness merit comparison.
- [ ] For every set, specify its question, columns, timing assumptions, and whether it could actually be deployed. If all 30 are verified, an all-features set is a legitimate baseline; do not invent an “other” group merely to have one.
- [ ] Treat a set containing an unresolved point-in-time feature as **diagnostic** until availability is established. A high score does not prove it is safe.

### 2.2 Choose chronological boundaries, then check outcome maturity

Provisional periods: **Train → Validation–selection → Validation–calibration → Final Test (OOT)**. They are ordered by `CutoffDate`. Exact dates are not specified until row and positive counts per hotel/month are checked, along with seasonality and the early H2 anomaly. The final period is a **single out-of-time test** in this proposed project design. Lecture 4 depicts separate Test and successive OOT periods as a general pattern; add distinct periods only if the course rubric requires them or the available data supports a separate, meaningful evaluation question.

For every simulated training/refit boundary, include only historical rows whose **outcomes were known by that date**. For example, a booking with `CutoffDate = 26 Sep`, arrival on 10 Oct, and cancellation on 5 Oct belongs to the earlier period by prediction date but **could not have trained a model on 1 Oct**. Exclude it for that fit or move the fit boundary. A blanket `ArrivalDate ≤ boundary` is not an equivalent label-availability test, especially for completed stays.

- [ ] Freeze split boundaries and the list of usable labelled rows at each fit date.
- [ ] Ensure enough bookings and positives in Train, Validation–selection, Validation–calibration, and Final Test for each hotel.
- [ ] If the separate calibration period is too small or unrepresentative, revisit calibration design before model selection: a shared validation set with a simpler calibration model has reuse/optimism risk; properly out-of-fold calibration within development data is another option.
- [ ] Record seed, data/contract version, split dates, exclusions due to unresolved labels, and row/label counts. No random row split.

### 2.3 Predeclare baselines and criteria

- [ ] Start with a **constant-risk baseline per hotel** (training positive rate); compare a simple Logistic Regression and at least one suitable tree-based candidate. Separate H1/H2 models are the initial route; compare a pooled model only if there is a clear question and consistent evaluation. If a pooled model uses `Hotel` as an input, update the input contract explicitly because `Hotel` is currently metadata.
- [ ] Set **PR-AUC** as a primary imbalanced-class ranking metric, and report precision/recall at plausible top-*k* review capacities. Until a hotel capacity is known, show several *k* values rather than asserting an operating threshold.
- [ ] Report ROC-AUC/Gini for course comparability, plus probability quality via Brier score/reliability plots after calibration. Metrics must be shown by H1/H2 and temporal slice, not only pooled.
- [ ] Decide *before seeing Final Test* how a materially useful gain versus baseline and a simpler model will be judged. There is no universal PR-AUC increment that makes an ensemble worth deploying.

**Gate to Stage 3:** boundary and label-maturity rules are fixed; Test remains unseen; each candidate set has a rationale; metrics/baselines and an experiment logging scheme are specified.

**Suggested work products:** `split_manifest.json`, `feature_sets.yaml`, `experiment_plan.md`, baseline results on development periods only.

---

## Stage 3 — Run candidate experiments

**Input:** frozen development splits, feature sets, baseline/metrics, model/hyperparameter ranges.

One trial is **(feature set, model family, preprocessing recipe, hyperparameter configuration)**. Model parameters are learned during each fit; hyperparameters are chosen by comparing trials. Changing a hyperparameter requires a new fit. Avoid a huge exhaustive cross-product at the start: establish simple baselines, test purposeful feature groups/model families, then refine promising settings.

### 3.1 Implement one candidate as a single fitted chain

- [ ] Select only that trial's allowed columns. Keep `y` and IDs/dates out of the predictor matrix except explicitly approved features.
- [ ] Put learned preprocessing (imputation, category encoding, scaling if relevant) in the model pipeline. Fit it **with the estimator**, on training data only. Apply the same fitted rules to validation rows.
- [ ] Handle unseen inference categories and required/missing inputs deliberately. One-hot encoding and imputation are not assumed to have been completed by the Gold pipeline.
- [ ] Fit the estimator; generate **scores/probabilities**, not only hard predicted classes; calculate the chosen metrics.

### 3.2 Time-aware CV for tuning, if used

Within the Train period, create forward-moving training/validation folds. At each fold: use earlier rows with labels known by that fold's fit date → fit preprocessing and estimator → score the next later block. Refit **both** preprocessing and model for every fold and hyperparameter choice. A five-fold time-aware evaluation means five valid later validation windows, not rotating one of five blocks into validation while the other four (including future blocks) train.

Aggregate fold metrics and inspect their variation. Refit promising configurations on the eligible Train rows, then compare them on **the same Validation–selection bookings**. If CV proves unnecessary for the available period/data, a fixed temporal Train/Validation comparison is acceptable; document that choice. Never use Final Test in search.

### 3.3 Record trials and shortlist

For each trial log: feature-set ID, model family, preprocessing, hyperparameters, data/split version, fitted period, fold definitions, number of bookings/positives, metrics, training duration, and location of development predictions. Keep comparable validation rows/metrics across candidates.

**Gate to Stage 4:** baseline and at least a small, interpretable candidate set run reproducibly; no fold fits preprocessing on its future validation rows; results are comparable across H1/H2 and time.

**Suggested work products:** runnable training script/notebook for exploration, `experiments.csv`, development predictions, fitted trial checkpoints as needed (not yet the deployment artifact).

---

## Stage 4 — Review errors and revise features when justified

**Input:** trial logs and development predictions, true outcomes, feature register.

- [ ] Compare with the baseline and inspect false negatives/positives and ranking at top-*k* by hotel, month/season, channel/market segment, lead-time band, and other business-relevant slices with enough observations.
- [ ] Run **feature-set sensitivity comparisons** on the same split and model where useful: what changes when a group is added/removed? Check both gain and temporal stability. This is a sensitivity experiment, not proof that the group is causally useful or deployable.
- [ ] Use importance/permutation explanations to create hypotheses; correlated features and importance instability prevent automatic “low importance → drop” rules. Eligibility at T−14 overrides apparent predictive value.
- [ ] If there is a specific recurring failure, write a feature hypothesis: what mechanism it captures, its data source, whether it was available at T−14, how Aditya will compute it, and what improvement would count as useful.
- [ ] If justified, update/version the upstream feature logic and contract, rerun the necessary Stage 1 checks and all affected development experiments on the **same frozen split**. Rebuilding unchanged raw/Bronze/Silver stages is unnecessary if the new feature can be derived without changing them.

**Decision:** another purposeful development experiment? → Stage 3. New justified and reproducible feature? → version feature pipeline → Stage 1. Otherwise shortlist and proceed. **Do not** open Test to invent features from its errors and then call the reused Test untouched.

**Suggested work products:** error-analysis note, feature-set comparison, feature-change request or shortlisted configurations.

---

## Stage 5 — Select single model or ensemble

**Input:** shortlisted configurations and predictions on the same Validation–selection bookings.

- [ ] Identify the best **single-model** candidate under the predeclared metrics and constraints.
- [ ] Check whether strong candidates make different errors. If so, make an **ensemble candidate**, starting with a fixed, simple combination of predictions. If learning blend weights or a stacking model, use out-of-fold predictions where each base model had not trained on the row it predicts.
- [ ] Compare single versus ensemble on identical eligible bookings, H1/H2, and time periods. Inspect PR-AUC, precision/recall at review capacity, eventual calibration feasibility, and inference/runtime complexity.
- [ ] Translate a difference at top-*k* into additional **correctly identified high-risk bookings**, not “prevented cancellations.” If the gain is small relative to temporal variability, limited to one period/hotel, or operationally unhelpful, keep the simpler single model.
- [ ] Freeze selected features, preprocessing recipe, model family/components, hyperparameters, and ensemble rule before calibration and Final Test.

**Output:** a single selected solution specification. Neither Validation–calibration nor Final Test is used to choose the winner.

---

## Stage 6 — Refit and calibrate

**Input:** frozen solution specification, eligible development rows, held-out Validation–calibration rows.

- [ ] Refit the selected preprocessing and base model(s) on the development rows available with resolved labels **before** the calibration period. If using Train + Validation–selection, verify those outcomes were known by the simulated refit date.
- [ ] Generate uncalibrated scores on Validation–calibration; the base estimator(s) and preprocessing must not learn from those rows.
- [ ] Fit a calibration mapping from score to actual label on that held-out period. Begin with a low-complexity approach when positive cases are limited; check reliability/Brier and rank metrics by hotel and time. Calibration changes probability interpretation; it is not evidence an intervention will work.
- [ ] Ensure any separate hotel models have an explicit routing rule and appropriate calibration. Freeze the entire chain: **final feature selector/schema → fitted preprocessing → fitted model or ensemble → fitted calibrator → probability**.

**Why hold out calibration data?** Many candidate choices were already made using Validation–selection labels. Fitting calibration on the same rows would reuse those labels; the final test would still be independent, but validation calibration estimates may look optimistic. A separate period provides a cleaner estimate if its sample size and timing are adequate.

**Gate to Stage 7:** one serializable, callable frozen chain exists, its inputs are specified, and it has not seen Final Test labels or fitted on Final Test features.

---

## Stage 7 — Final Test (OOT) and release recommendation

**Input:** frozen chain and the later, untouched out-of-time test period.

- [ ] Apply the already fitted preprocessing and calibrator to Final Test features; do not fit anything or change the selected solution based on test rows.
- [ ] Report sample sizes/positive rates; PR-AUC and ROC-AUC/Gini; precision/recall at fixed top-*k*; Brier/reliability; comparisons with hotel baseline; H1/H2 and time slices; uncertainty where useful.
- [ ] Report feature-availability assumptions and sensitivity between deployable versus diagnostic feature sets. Distinguish failures caused by ranking, calibration, input shift, and provenance limitations.
- [ ] Compare the frozen solution with predeclared criteria; mark **handoff** or **hold**. A held version can motivate a new development run, but repeated redesign using the same final holdout invalidates a claim of untouched evaluation. A future holdout or explicitly exploratory reporting is then needed.

**OOT meaning:** `Test` is the role of this held-out dataset; `OOT` says it is later in time. A single later test covers future generalization for this project. It does not guarantee permanent stability: rolling historical checks and production monitoring provide further evidence over time. The lecture shows additional OOT windows as a general monitoring/evaluation design, not proof that this short project must reserve several extra final partitions.

**Output:** evaluation report, reproducible prediction file for audit, promote/hold decision, documented limitations.

---

## Stage 8 — Package and hand off to Aditya

**Input:** accepted frozen chain, feature contract, evaluation results.

Deliver **runnable components**, not just a notebook or one AUC value:

| Deliverable | Example form | Purpose |
|---|---|---|
| Fitted scoring chain | `model_v1.joblib` or equivalent, possibly one artifact per hotel | Contains selected preprocessing, estimator/ensemble, and calibrator. Loaded for scoring; nothing refits at inference. |
| Scoring code | `score_batch.py` or a `predict_batch()` module | Reads unlabeled eligible Gold features, validates inputs, routes H1/H2, loads artifact, returns scores. |
| Input/output contract | `feature_schema.json` or YAML + sample records | Final feature names/types/order, category/missing behavior, eligible T−14 population, output fields, and feature contract version. |
| Version and evaluation record | `model_card.md`, `metrics.json`, `split_manifest.json` | Training/refit cutoff, data/run/version IDs, decisions, metrics by hotel/time, assumptions, limitations. |
| Reproducible environment and example | requirements/lockfile + sample input/output and a one-command smoke run | Ensures Aditya obtains the same prediction from the same sample. |

Minimal output record: `Booking_ID`, `Hotel`, `CutoffDate`, **calibrated** `risk_score`, `model_version`, `scored_at`. `CutoffDate` is the prediction-as-of date; `scored_at` is the actual job execution timestamp. A separately agreed top-*k*/threshold policy may be handed off as a **decision rule**, not silently baked into the meaning of the model score.

Before claiming the package can score future data, verify that **every final feature** can be recreated on the daily cutoff. The current Google Trends export has a documented scale/vintage limitation; resolve it or exclude that feature from the deployed solution. Test that a batch can be scored with no `IsCanceled` column and that unseen categories or missing optional values behave as documented.

**Output:** a versioned package Aditya can load for daily batch scoring and an explicit set of known limits.

---

## Execution order for the first working session

1. Obtain the **complete latest** H1/H2 Gold feature and label/model datasets. Record their data pipeline version.
2. Run Stage 1's schema, key, population, label, and feature-availability checks. Produce a small feature register; escalate only specific discrepancies to the data team.
3. Count booking rows and positive labels by `CutoffDate` month and hotel, plus outcome-availability delays. Only then propose exact split dates and test calibration sample size.
4. Freeze a **minimal defensible feature set**, a constant baseline, and a simple train/validation experiment before broad tuning or ensemble work.
5. Build the experiment log and time-aware evaluation loop; review development results before touching Final Test.

## Decision log to fill during work

| Decision | Current status | Evidence required |
|---|---|---|
| Complete Gold delivery and data version | Open | Actual H1/H2 tables and reconciliation with README counts. |
| Final deployable feature set | Open | Point-in-time provenance and future reproducibility, then development comparison. |
| Exact chronological boundaries | Open | Monthly counts, positives, seasonality, label maturity, enough calibration/test cases. |
| Separate versus pooled hotel model | Separate first; compare pooled if justified | Same splits, eligible rows, per-hotel metrics and reproducibility. |
| CV design and search scope | Time-aware if used | Number and size of valid forward folds, positive cases, computational budget. |
| Calibration strategy | Held-out later validation proposed | Enough positive cases by hotel and stable calibration across periods. |
| Single versus ensemble | Open | Consistent useful improvement on the same validation rows versus added complexity. |
| Review capacity / decision threshold | Open with hotel/business owner | Actual daily capacity or costs/benefits; until then report several top-*k* values. |
| Separate in-time Test in addition to OOT | Not proposed by default | Course rubric or a distinct evaluation purpose and enough historical data. |

## References used

- Supplied hotel booking pipeline ZIP: `README.ipynb` and `datamart/gold/feature_contract.json` (T−14 rules, candidate columns, documented limitations and reference counts).
- Supplied `EDA_Hotel_Booking.ipynb` (T−14 population, feature leakage review, H1/H2 time patterns).
- Supplied `10_5 meeting.docx` (team ownership, initial chronological validation direction and metrics). The meeting's earlier D−7 framing has been superseded by the current pipeline's T−14 choice.
- Supplied model development lecture (split, tuning, calibration and OOT concepts) and model deployment lecture (batch inference and model artefacts). The lab ZIP was not required to derive this project-specific procedure.
