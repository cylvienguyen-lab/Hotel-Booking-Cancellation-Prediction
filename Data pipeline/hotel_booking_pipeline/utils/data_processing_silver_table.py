import os

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, DoubleType, DateType


def process_silver_table(hotel, bronze_directory, silver_directory, spark):
    # connect to bronze table
    filepath = os.path.join(bronze_directory, 'bronze_' + hotel + '.csv')
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: remove whitespace and source missing-value tokens
    for column in df.columns:
        df = df.withColumn(column, F.trim(col(column)))
    for column in ['Agent', 'Company']:
        df = df.withColumn(column, F.when(col(column).isNull() | col(column).isin('NULL', ''), 'None').otherwise(col(column)))
    for column in [c for c in df.columns if c not in ['Agent', 'Company']]:
        df = df.withColumn(column, F.when(col(column).isin('NULL', 'NA', ''), F.lit(None)).otherwise(col(column)))

    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        'IsCanceled': IntegerType(), 'LeadTime': IntegerType(),
        'ArrivalDateYear': IntegerType(), 'ArrivalDateMonth': StringType(),
        'ArrivalDateWeekNumber': IntegerType(), 'ArrivalDateDayOfMonth': IntegerType(),
        'StaysInWeekendNights': IntegerType(), 'StaysInWeekNights': IntegerType(),
        'Adults': IntegerType(), 'Children': IntegerType(), 'Babies': IntegerType(),
        'Meal': StringType(), 'Country': StringType(), 'MarketSegment': StringType(),
        'DistributionChannel': StringType(), 'IsRepeatedGuest': IntegerType(),
        'PreviousCancellations': IntegerType(), 'PreviousBookingsNotCanceled': IntegerType(),
        'ReservedRoomType': StringType(), 'AssignedRoomType': StringType(),
        'BookingChanges': IntegerType(), 'DepositType': StringType(),
        'Agent': StringType(), 'Company': StringType(), 'DaysInWaitingList': IntegerType(),
        'CustomerType': StringType(), 'ADR': DoubleType(),
        'RequiredCarParkingSpaces': IntegerType(), 'TotalOfSpecialRequests': IntegerType(),
        'ReservationStatus': StringType(), 'ReservationStatusDate': DateType(),
        'SourceRowNumber': IntegerType(),
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: identifiers and dates; full dates are metadata for filters/joins
    df = df.withColumn('Hotel', F.lit(hotel))
    df = df.withColumn('Booking_ID', F.concat(F.lit(hotel + '-'), F.format_string('%06d', col('SourceRowNumber'))))
    date_text = F.concat_ws('-', col('ArrivalDateYear'), col('ArrivalDateMonth'), col('ArrivalDateDayOfMonth'))
    df = df.withColumn('ArrivalDate', F.to_date(date_text, 'yyyy-MMMM-d'))
    df = df.withColumn('BookingCreatedDate', F.date_sub(col('ArrivalDate'), col('LeadTime')))
    df = df.withColumn('CutoffDate', F.date_sub(col('ArrivalDate'), 14))
    df = df.withColumn('ArrivalDayOfWeek', (F.pmod(F.dayofweek(col('ArrivalDate')) + 5, F.lit(7)) + 1).cast(IntegerType()))
    df = df.withColumn('TotalNights', col('StaysInWeekNights') + col('StaysInWeekendNights'))
    df = df.withColumn('PartySize', col('Adults') + col('Children') + col('Babies'))

    # clean ADR: negative becomes null; zero and high values are retained for review
    df = df.withColumn('ADRReview', (col('ADR') < 0) | (col('ADR') > 1000))
    df = df.withColumn('ADR', F.when(col('ADR') >= 0, col('ADR')).otherwise(F.lit(None)))

    # augment data: estimated rooms are a scenario, not an observed room count
    capacity = {'H1': 2, 'H2': 4}[hotel]
    df = df.withColumn('EstimatedRoomCount', F.when(col('PartySize') > 0, F.ceil(col('PartySize') / capacity)).cast(IntegerType()))

    # flag differences for review; do not rewrite final-status dates or exclude by outcome
    df = df.withColumn('DateReview',
        ((col('ReservationStatus') == 'No-Show') & (col('ReservationStatusDate') != col('ArrivalDate')))
        | ((col('ReservationStatus') == 'Check-Out')
           & (col('ReservationStatusDate') != F.date_add(col('ArrivalDate'), col('TotalNights')))))
    df = df.withColumn('OvernightScope', (col('TotalNights') > 0) & (col('PartySize') > 0))
    df = df.withColumn('EligibleT14', (col('LeadTime') >= 14)
        & ~((col('ReservationStatus').isin('Canceled', 'No-Show'))
            & (col('ReservationStatusDate') <= col('CutoffDate'))))

    # save silver table; keep all bookings so exclusions remain inspectable
    os.makedirs(silver_directory, exist_ok=True)
    filepath = os.path.join(silver_directory, 'silver_' + hotel + '.parquet')
    df.write.mode('overwrite').parquet(filepath)
    print('saved to:', filepath)

    # save review rows separately; these flags are not model predictors
    review = df.filter(col('ADRReview') | col('DateReview') | ~F.coalesce(col('OvernightScope'), F.lit(False)))
    filepath = os.path.join(silver_directory, 'silver_review_' + hotel + '.parquet')
    review.write.mode('overwrite').parquet(filepath)
    print('saved to:', filepath, 'review row count:', review.count())
    return df
