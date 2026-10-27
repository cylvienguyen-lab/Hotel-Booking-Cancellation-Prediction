"""Standalone output-generation functions for the production batch job.

Separates the business-logic aggregations (hotel-level forecast + daily
risk shortlist) out of the evaluation notebook so the deployment /
MLOps team can call them from a batch orchestrator without re-deriving
the logic.

Both functions take a *scored bookings* DataFrame (already has
``p_cancel`` per booking from ``model_scoring.score_bookings``) and
return the final business-facing DataFrames. No model loading happens
inside this module — that's ``model_scoring.load_hotel_model``.

Typical daily batch structure::

    from utils.model_scoring import load_hotel_model, score_bookings
    from utils.model_outputs import (
        aggregate_hotel_forecast, generate_booking_shortlist,
    )

    today = pd.Timestamp.utcnow().normalize()
    for hotel in ("H1", "H2"):
        model = load_hotel_model(hotel, Path("models"))
        bookings_today = <pull Gold feature frame for CutoffDate == today>
        scored = score_bookings(model, bookings_today, as_of_date=today)

        shortlist = generate_booking_shortlist(
            scored, k_per_hotel={"H1": 10, "H2": 20},
            rank_by="p_cancel",
        )
        forecasts = aggregate_hotel_forecast(scored)

        <write shortlist and forecasts to the ops dashboard>
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Deliverable 1 — hotel-level forecast
# -----------------------------------------------------------------------------
def aggregate_hotel_forecast(
    scored_bookings: pd.DataFrame,
    room_count_col: str = "EstimatedRoomCount",
) -> dict[str, pd.DataFrame]:
    """Build the nightly expected-arrivals and expected-occupancy frames.

    ``scored_bookings`` must have:
        Hotel, Booking_ID, ArrivalDate, TotalNights, p_cancel,
        EstimatedRoomCount (or whatever ``room_count_col`` is).

    Returns a dict with two frames:
        - "arrivals":  one row per (hotel, arrival_date):
            n_bookings, E_arrivals (= sum(1 − p_cancel))
        - "occupancy": one row per (hotel, night_date):
            E_room_nights (= sum((1 − p_cancel) × rooms) over bookings
            whose stay covers that night), n_bookings_covering

    The occupancy forecast expands each booking across its stay
    (ArrivalDate … ArrivalDate + TotalNights − 1) and weights by room
    count. This is what the hotel operations dashboard would consume
    for staffing / F&B planning.

    If actual outcomes (``y_true``) are present, the function also
    returns ``actual_arrivals`` and ``actual_room_nights`` columns for
    backtesting; otherwise those are left out.
    """
    if "ArrivalDate" not in scored_bookings.columns:
        raise ValueError("scored_bookings must have ArrivalDate")
    if room_count_col not in scored_bookings.columns:
        raise ValueError(f"scored_bookings must have {room_count_col!r}")

    df = scored_bookings.copy()
    df["ArrivalDate"] = pd.to_datetime(df["ArrivalDate"])
    df["TotalNights"] = df["TotalNights"].astype(int)
    df["_rooms"] = df[room_count_col].fillna(1).astype(int)
    df["E_arrival_prob"] = 1.0 - df["p_cancel"]

    has_truth = "y_true" in df.columns
    if has_truth:
        df["_actual_arrival"] = (df["y_true"] == 0).astype(int)

    # ------- arrivals (one check-in per booking) -------
    arrivals_agg = {
        "n_bookings": ("Booking_ID", "count"),
        "E_arrivals": ("E_arrival_prob", "sum"),
    }
    if has_truth:
        arrivals_agg["actual_arrivals"] = ("_actual_arrival", "sum")
    arrivals = (
        df.groupby(["Hotel", "ArrivalDate"], as_index=False)
        .agg(**arrivals_agg)
        .rename(columns={"ArrivalDate": "arrival_date"})
    )

    # ------- occupancy (expand across stay nights, weight by rooms) -------
    max_nights = int(df["TotalNights"].max())
    chunks = []
    for d in range(max_nights):
        active = df[df["TotalNights"] > d].copy()
        active["NightDate"] = active["ArrivalDate"] + pd.Timedelta(days=d)
        keep = ["Hotel", "Booking_ID", "NightDate", "_rooms", "E_arrival_prob"]
        if has_truth:
            keep.append("_actual_arrival")
        chunks.append(active[keep])
    expanded = pd.concat(chunks, ignore_index=True)
    expanded["E_room_nights"] = expanded["E_arrival_prob"] * expanded["_rooms"]
    if has_truth:
        expanded["actual_room_nights"] = expanded["_actual_arrival"] * expanded["_rooms"]

    occ_agg = {
        "E_room_nights": ("E_room_nights", "sum"),
        "n_bookings_covering": ("Booking_ID", "count"),
    }
    if has_truth:
        occ_agg["actual_room_nights"] = ("actual_room_nights", "sum")
    occupancy = (
        expanded.groupby(["Hotel", "NightDate"], as_index=False)
        .agg(**occ_agg)
        .rename(columns={"NightDate": "night_date"})
    )

    return {"arrivals": arrivals, "occupancy": occupancy}


# -----------------------------------------------------------------------------
# Deliverable 2 — booking-level risk shortlist
# -----------------------------------------------------------------------------
def generate_booking_shortlist(
    scored_bookings: pd.DataFrame,
    k_per_hotel: dict[str, int],
    rank_by: Literal["p_cancel", "expected_lost_revenue"] = "p_cancel",
    include_y_true: bool = False,
) -> pd.DataFrame:
    """Build the daily top-K-per-cutoff risk shortlist for operations.

    ``scored_bookings`` must have:
        Hotel, Booking_ID, CutoffDate, ArrivalDate, p_cancel,
        model_version, prediction_as_of_date, generated_at_utc,
        ADR, TotalNights, EstimatedRoomCount.

    Parameters
    ----------
    k_per_hotel : {"H1": 10, "H2": 20} or similar
        How many bookings per cutoff date to flag per hotel. Operations
        picks this from their actual contact-budget capacity.
    rank_by : "p_cancel" | "expected_lost_revenue"
        - "p_cancel" maximises recall of cancellations.
        - "expected_lost_revenue" prioritises highest-dollar bookings
          (p_cancel × ADR × TotalNights × EstimatedRoomCount).
    include_y_true : bool
        True = include y_true for backtesting / Section 4 report. False
        = ops-facing table (no leakage, no realised outcome).

    Returns
    -------
    DataFrame with one row per flagged booking, sorted by
    (Hotel, CutoffDate, risk_rank). Columns:
        Hotel, CutoffDate, ArrivalDate, Booking_ID,
        p_cancel, ADR, TotalNights, EstimatedRoomCount,
        expected_lost_revenue, risk_rank,
        model_version, prediction_as_of_date, generated_at_utc,
        [y_true if include_y_true]
    """
    required = {
        "Hotel", "Booking_ID", "CutoffDate", "ArrivalDate", "p_cancel",
        "ADR", "TotalNights", "EstimatedRoomCount",
        "model_version", "prediction_as_of_date", "generated_at_utc",
    }
    missing = required - set(scored_bookings.columns)
    if missing:
        raise ValueError(f"scored_bookings missing required columns: {sorted(missing)}")

    df = scored_bookings.copy()
    df["expected_lost_revenue"] = (
        df["p_cancel"] * df["ADR"] * df["TotalNights"] * df["EstimatedRoomCount"]
    )
    if rank_by not in ("p_cancel", "expected_lost_revenue"):
        raise ValueError(f"rank_by must be p_cancel or expected_lost_revenue, got {rank_by!r}")

    df["risk_rank"] = (
        df.groupby(["Hotel", "CutoffDate"])[rank_by]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    frames = []
    for hotel, k in k_per_hotel.items():
        frames.append(df[(df.Hotel == hotel) & (df.risk_rank <= k)])
    shortlist = pd.concat(frames, ignore_index=True) if frames else df.iloc[:0].copy()

    cols = [
        "Hotel", "CutoffDate", "ArrivalDate", "Booking_ID",
        "p_cancel", "ADR", "TotalNights", "EstimatedRoomCount",
        "expected_lost_revenue", "risk_rank",
        "model_version", "prediction_as_of_date", "generated_at_utc",
    ]
    if include_y_true and "y_true" in shortlist.columns:
        cols.append("y_true")
    shortlist = (
        shortlist[cols]
        .sort_values(["Hotel", "CutoffDate", "risk_rank"])
        .reset_index(drop=True)
    )
    return shortlist
