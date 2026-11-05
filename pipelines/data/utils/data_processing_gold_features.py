import os

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType


# Explicit predictor list: metadata and final outcomes must never enter model inputs.
CATEGORICAL_COLUMNS = [
    'ArrivalDateMonth', 'Meal', 'MarketSegment', 'DistributionChannel',
    'ReservedRoomType', 'DepositType', 'CustomerType',
]
NUMERIC_COLUMNS = [
    'LeadTime', 'ArrivalDateYear', 'ArrivalDateWeekNumber', 'ArrivalDateDayOfMonth',
    'StaysInWeekendNights', 'StaysInWeekNights', 'Adults', 'Children', 'Babies',
    'IsRepeatedGuest', 'PreviousCancellations', 'PreviousBookingsNotCanceled',
    'BookingChanges', 'DaysInWaitingList', 'ADR', 'TotalOfSpecialRequests',
    'ArrivalDayOfWeek', 'TotalNights', 'PartySize', 'IsPortugalHoliday', 'EURUSD',
    'HotelSearchInterest', 'FlightSearchInterest',
]
METADATA_COLUMNS = [
    'Booking_ID', 'Hotel', 'ArrivalDate', 'BookingCreatedDate', 'CutoffDate',
    'FXObservationDate', 'EstimatedRoomCount',
    'TrendWeekStart', 'TrendWeekEnd', 'TrendAvailableDate',
]


def process_features_gold_table(hotel, silver_hotel_directory, silver_external_directory,
                                silver_scoring_directory, gold_feature_store_directory, spark):
    # connect to silver tables
    filepath = os.path.join(silver_hotel_directory, 'silver_' + hotel + '.parquet')
    booking_df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', booking_df.count())
    holidays_df = spark.read.parquet(os.path.join(silver_external_directory, 'silver_holidays.parquet'))
    rates_df = spark.read.parquet(os.path.join(silver_scoring_directory, 'silver_scoring_exchange_rates_' + hotel + '.parquet'))
    trends_df = spark.read.parquet(os.path.join(silver_scoring_directory, 'silver_scoring_google_trends_' + hotel + '.parquet'))

    # keep active bookings at T-14 within the overnight-stay scope
    df = booking_df.filter(col('EligibleT14') & col('OvernightScope'))

    # join the known holiday calendar and prior-observation-day exchange rate
    df = df.join(holidays_df, (df['ArrivalDate'] == holidays_df['HolidayDate'])
                 & (holidays_df['HolidayKnownFrom'] <= df['CutoffDate']), how='left')
    df = df.withColumn('IsPortugalHoliday', F.when(col('HolidayDate').isNotNull(), 1).otherwise(0).cast(IntegerType()))
    df = df.join(rates_df, on='CutoffDate', how='left')
    df = df.join(trends_df, on='CutoffDate', how='left')

    # select the agreed predictors and reporting metadata
    # Keep categories as text and preserve documented missing values.
    df = df.select(*(METADATA_COLUMNS + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS))

    # save gold feature store
    os.makedirs(gold_feature_store_directory, exist_ok=True)
    filepath = os.path.join(gold_feature_store_directory, 'gold_feature_store_' + hotel + '.parquet')
    df.write.mode('overwrite').parquet(filepath)
    print('saved to:', filepath, 'row count:', df.count())
    return df
