import os

from pyspark.sql.functions import col


def process_labels_gold_table(hotel, silver_hotel_directory, gold_label_store_directory, spark):
    # connect to silver table
    filepath = os.path.join(silver_hotel_directory, 'silver_' + hotel + '.parquet')
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # use the same T-14 population as the feature store and the existing outcome label
    df = df.filter(col('EligibleT14') & col('OvernightScope'))
    # Status/date are label audit fields, never predictors. Date supports outcome maturity checks.
    df = df.select('Booking_ID', 'Hotel', 'IsCanceled', 'ReservationStatus', 'ReservationStatusDate')

    # save gold label store
    os.makedirs(gold_label_store_directory, exist_ok=True)
    filepath = os.path.join(gold_label_store_directory, 'gold_label_store_' + hotel + '.parquet')
    df.write.mode('overwrite').parquet(filepath)
    print('saved to:', filepath, 'row count:', df.count())
    return df
