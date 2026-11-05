"""Rebuild the hotel Bronze, Silver, and Gold tables from supplied sources.

Run from any working directory with ``python run_data.py``. The script reuses
the processing functions from the course notebook and exits nonzero if an
input or the resulting Gold contract is invalid.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyspark
import pyspark.sql.functions as F

from utils import data_processing_bronze_features as bronze_features
from utils import data_processing_bronze_table as bronze_table
from utils import data_processing_gold_features as gold_features
from utils import data_processing_gold_model_dataset as gold_model
from utils import data_processing_gold_table as gold_table
from utils import data_processing_silver_features as silver_features
from utils import data_processing_silver_table as silver_table


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
EXPECTED = {"H1": (20_140, 1_518), "H2": (38_233, 4_536)}
REQUIRED_INPUTS = (
    "data/raw/H1.csv",
    "data/raw/H2.csv",
    "data/external/holidays/holidays_PT_2015.json",
    "data/external/holidays/holidays_PT_2016.json",
    "data/external/holidays/holidays_PT_2017.json",
    "data/external/exchange_rates/ecb_eurusd_2014_2017.csv",
    "data/external/google_trends/google_trends_hotels_flights_weekly_2015_2017.csv",
    "config/feature_contract.json",
)


def _paths() -> dict[str, str]:
    return {
        "raw": str(DATA_ROOT / "raw"),
        "holidays": str(DATA_ROOT / "external/holidays"),
        "exchange_rates": str(DATA_ROOT / "external/exchange_rates"),
        "trends": str(DATA_ROOT / "external/google_trends"),
        "bronze_hotels": str(DATA_ROOT / "datamart/bronze/hotels"),
        "bronze_external": str(DATA_ROOT / "datamart/bronze/external"),
        "silver_hotels": str(DATA_ROOT / "datamart/silver/hotels"),
        "silver_external": str(DATA_ROOT / "datamart/silver/external"),
        "silver_scoring": str(DATA_ROOT / "datamart/silver/scoring"),
        "gold_features": str(DATA_ROOT / "datamart/gold/features"),
        "gold_labels": str(DATA_ROOT / "datamart/gold/labels"),
        "gold_model": str(DATA_ROOT / "datamart/gold/model_dataset"),
    }


def _check_inputs() -> dict:
    missing = [name for name in REQUIRED_INPUTS if not (PROJECT_ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))

    contract = json.loads((PROJECT_ROOT / "config/feature_contract.json").read_text(encoding="utf-8"))
    expected_groups = {
        "numeric_predictors": gold_features.NUMERIC_COLUMNS,
        "categorical_predictors": gold_features.CATEGORICAL_COLUMNS,
        "metadata_only": gold_features.METADATA_COLUMNS,
    }
    for name, columns in expected_groups.items():
        if contract.get(name) != columns:
            raise ValueError(f"feature_contract.json disagrees with {name} in Gold code")
    if contract.get("target") != "IsCanceled":
        raise ValueError("feature_contract.json has the wrong target")
    return contract


def _check_gold(spark, paths: dict[str, str]) -> list[dict]:
    expected_features = set(
        gold_features.METADATA_COLUMNS
        + gold_features.NUMERIC_COLUMNS
        + gold_features.CATEGORICAL_COLUMNS
    )
    summary = []
    for hotel, (expected_rows, expected_positive) in EXPECTED.items():
        feature = spark.read.parquet(str(Path(paths["gold_features"]) / f"gold_feature_store_{hotel}.parquet"))
        label = spark.read.parquet(str(Path(paths["gold_labels"]) / f"gold_label_store_{hotel}.parquet"))
        model = spark.read.parquet(str(Path(paths["gold_model"]) / f"gold_model_dataset_{hotel}.parquet"))

        if set(feature.columns) != expected_features:
            raise ValueError(f"{hotel}: Gold feature columns do not match the contract")
        if set(model.columns) != expected_features | {"IsCanceled"}:
            raise ValueError(f"{hotel}: Gold model dataset columns do not match the contract")
        if set(label.columns) != {"Booking_ID", "Hotel", "IsCanceled", "ReservationStatus", "ReservationStatusDate"}:
            raise ValueError(f"{hotel}: Gold label columns do not match the contract")

        counts = model.agg(
            F.count("*").alias("rows"),
            F.countDistinct("Booking_ID").alias("unique_bookings"),
            F.sum("IsCanceled").alias("positives"),
        ).first()
        rows, unique, positives = (int(counts[x]) for x in ("rows", "unique_bookings", "positives"))
        if (rows, unique, positives) != (expected_rows, expected_rows, expected_positive):
            raise ValueError(f"{hotel}: unexpected Gold counts {(rows, unique, positives)}")
        if feature.count() != rows or label.count() != rows:
            raise ValueError(f"{hotel}: feature/label row counts do not match model dataset")
        if model.filter(F.col("IsCanceled").isNull() | ~F.col("IsCanceled").isin(0, 1)).limit(1).count():
            raise ValueError(f"{hotel}: invalid cancellation target")
        in_scope = (
            (F.col("LeadTime") >= 14)
            & (F.col("PartySize") > 0)
            & (F.col("TotalNights") > 0)
            & (F.col("CutoffDate") == F.date_sub(F.col("ArrivalDate"), 14))
        )
        if model.filter(~F.coalesce(in_scope, F.lit(False))).limit(1).count():
            raise ValueError(f"{hotel}: a Gold booking violates the T-14 scope")
        summary.append({"hotel": hotel, "rows": rows, "positive_labels": positives})
    return summary


def main() -> None:
    contract = _check_inputs()
    paths = _paths()
    (DATA_ROOT / "datamart/gold/data_build_audit.json").unlink(missing_ok=True)
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("hotel_booking_data")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        for hotel in ("H1", "H2"):
            bronze_table.process_bronze_table(hotel, paths["raw"], paths["bronze_hotels"], spark)
        bronze_features.process_bronze_holidays(paths["holidays"], paths["bronze_external"], spark)
        bronze_features.process_bronze_exchange_rates(paths["exchange_rates"], paths["bronze_external"], spark)
        bronze_features.process_bronze_google_trends(paths["trends"], paths["bronze_external"], spark)

        for hotel in ("H1", "H2"):
            silver_table.process_silver_table(hotel, paths["bronze_hotels"], paths["silver_hotels"], spark)
        silver_features.process_silver_holidays(paths["bronze_external"], paths["silver_external"], spark)
        silver_features.process_silver_exchange_rates(paths["bronze_external"], paths["silver_external"], spark)
        silver_features.process_silver_google_trends(paths["bronze_external"], paths["silver_external"], spark)
        for hotel in ("H1", "H2"):
            silver_features.process_silver_scoring_google_trends(
                hotel, paths["silver_hotels"], paths["silver_external"], paths["silver_scoring"], spark
            )
            silver_features.process_silver_scoring_exchange_rates(
                hotel, paths["silver_hotels"], paths["silver_external"], paths["silver_scoring"], spark
            )

        for hotel in ("H1", "H2"):
            gold_features.process_features_gold_table(
                hotel, paths["silver_hotels"], paths["silver_external"],
                paths["silver_scoring"], paths["gold_features"], spark,
            )
            gold_table.process_labels_gold_table(hotel, paths["silver_hotels"], paths["gold_labels"], spark)
            gold_model.process_model_dataset_gold_table(
                hotel, paths["gold_features"], paths["gold_labels"], paths["gold_model"], spark
            )

        summary = _check_gold(spark, paths)
        gold_root = DATA_ROOT / "datamart/gold"
        gold_root.mkdir(parents=True, exist_ok=True)
        (gold_root / "feature_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        (gold_root / "data_build_audit.json").write_text(
            json.dumps({"status": "passed", "hotels": summary}, indent=2) + "\n", encoding="utf-8"
        )
        print("Data build passed:", json.dumps(summary))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
