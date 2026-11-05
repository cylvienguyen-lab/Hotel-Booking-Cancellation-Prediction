import os

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import DateType, DoubleType
from pyspark.sql.window import Window


def process_silver_google_trends(bronze_directory, silver_directory, spark):
    # connect to bronze table
    filepath = os.path.join(bronze_directory, 'bronze_google_trends.csv')
    source_df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', source_df.count())

    # map each hotel's two search terms; all four share one exported comparison scale
    tables = []
    terms = {'H1': ('Algarve hotels', 'flights to Faro'),
             'H2': ('Lisbon hotels', 'flights to Lisbon')}
    for hotel, (hotel_term, flight_term) in terms.items():
        df = source_df.select(F.lit(hotel).alias('Hotel'),
            col('Week').cast(DateType()).alias('TrendWeekStart'),
            col(hotel_term + ': (Worldwide)').cast(DoubleType()).alias('HotelSearchInterest'),
            col(flight_term + ': (Worldwide)').cast(DoubleType()).alias('FlightSearchInterest'))
        df = df.withColumn('TrendWeekEnd', F.date_add(col('TrendWeekStart'), 6))
        # exclude weeks only partly covered by the requested export period
        df = df.filter((col('TrendWeekStart') >= F.lit('2015-01-01').cast(DateType()))
                       & (col('TrendWeekEnd') <= F.lit('2017-08-31').cast(DateType())))
        # offline assumption: seven-day buffer after week end, not a verified release date
        df = df.withColumn('TrendAvailableDate', F.date_add(col('TrendWeekEnd'), 7))
        tables.append(df)
    df = tables[0].unionByName(tables[1])

    # save silver table
    os.makedirs(silver_directory, exist_ok=True)
    filepath = os.path.join(silver_directory, 'silver_google_trends.parquet')
    df.write.mode('overwrite').parquet(filepath)
    print('saved to:', filepath, 'row count:', df.count())
    return df


def process_silver_scoring_google_trends(hotel, silver_hotel_directory, silver_external_directory, silver_scoring_directory, spark):
    # connect to silver tables; match once per cutoff day
    filepath = os.path.join(silver_hotel_directory, 'silver_' + hotel + '.parquet')
    days_df = spark.read.parquet(filepath).select('CutoffDate').distinct()
    filepath = os.path.join(silver_external_directory, 'silver_google_trends.parquet')
    trends_df = spark.read.parquet(filepath).filter(col('Hotel') == hotel).drop('Hotel')

    # choose the latest eligible completed week without future backfill
    df = days_df.join(trends_df, trends_df['TrendAvailableDate'] <= days_df['CutoffDate'], how='left')
    latest_first = Window.partitionBy('CutoffDate').orderBy(col('TrendAvailableDate').desc_nulls_last())
    df = df.withColumn('row_number', F.row_number().over(latest_first))
    df = df.filter(col('row_number') == 1).drop('row_number')

    # save the scoring-date table
    os.makedirs(silver_scoring_directory, exist_ok=True)
    filepath = os.path.join(silver_scoring_directory, 'silver_scoring_google_trends_' + hotel + '.parquet')
    df.write.mode('overwrite').parquet(filepath)
    print('saved to:', filepath, 'row count:', df.count())
    return df


def process_silver_holidays(bronze_directory, silver_directory, spark):
    # connect to bronze table
    filepath = os.path.join(bronze_directory, 'bronze_holidays.csv')
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: retain national mandatory dates; four holidays were suspended in 2015
    standard = ["New Year's Day", 'Good Friday', 'Easter Sunday', 'Freedom Day', 'Labour Day',
                'National Day', 'Assumption Day', 'Immaculate Conception', 'Christmas Day']
    restored = ['Corpus Christi', 'Republic Day', 'All Saints Day', 'Restoration of Independence']
    df = df.withColumn('HolidayDate', col('date').cast(DateType()))
    df = df.filter((col('global') == 'true') & (col('countryCode') == 'PT'))
    df = df.filter(col('name').isin(standard) | (col('name').isin(restored) & (F.year('HolidayDate') >= 2016)))
    df = df.withColumn('HolidayKnownFrom', F.when(col('name').isin(restored), F.lit('2016-04-02'))
                       .otherwise(F.lit('2012-06-25')).cast(DateType()))
    df = df.select('HolidayDate', col('name').alias('HolidayName'), 'HolidayKnownFrom')

    # save silver table
    os.makedirs(silver_directory, exist_ok=True)
    filepath = os.path.join(silver_directory, 'silver_holidays.parquet')
    df.write.mode('overwrite').parquet(filepath)
    print('saved to:', filepath, 'row count:', df.count())
    return df


def process_silver_exchange_rates(bronze_directory, silver_directory, spark):
    # connect to bronze table
    filepath = os.path.join(bronze_directory, 'bronze_exchange_rates.csv')
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce the selected currency pair and data types
    df = df.filter((col('CURRENCY') == 'USD') & (col('CURRENCY_DENOM') == 'EUR'))
    df = df.select(col('TIME_PERIOD').cast(DateType()).alias('FXObservationDate'),
                   col('OBS_VALUE').cast(DoubleType()).alias('EURUSD'))

    # save silver table
    os.makedirs(silver_directory, exist_ok=True)
    filepath = os.path.join(silver_directory, 'silver_exchange_rates.parquet')
    df.write.mode('overwrite').parquet(filepath)
    print('saved to:', filepath, 'row count:', df.count())
    return df


def process_silver_scoring_exchange_rates(hotel, silver_hotel_directory, silver_external_directory, silver_scoring_directory, spark):
    # connect to silver tables; use distinct cutoff days to avoid a booking x rates join
    filepath = os.path.join(silver_hotel_directory, 'silver_' + hotel + '.parquet')
    days_df = spark.read.parquet(filepath).select('CutoffDate').distinct()
    filepath = os.path.join(silver_external_directory, 'silver_exchange_rates.parquet')
    rates_df = spark.read.parquet(filepath)

    # choose the latest published-day observation before scoring, as in Assignment 1's as-of join
    df = days_df.join(rates_df, rates_df['FXObservationDate'] < days_df['CutoffDate'], how='left')
    latest_first = Window.partitionBy('CutoffDate').orderBy(col('FXObservationDate').desc_nulls_last())
    df = df.withColumn('row_number', F.row_number().over(latest_first))
    df = df.filter(col('row_number') == 1).drop('row_number')

    # save the scoring-date table; Gold joins it by CutoffDate
    os.makedirs(silver_scoring_directory, exist_ok=True)
    filepath = os.path.join(silver_scoring_directory, 'silver_scoring_exchange_rates_' + hotel + '.parquet')
    df.write.mode('overwrite').parquet(filepath)
    print('saved to:', filepath, 'row count:', df.count())
    return df
