import csv
import os

from pyspark.sql.types import StructType, StructField, StringType, IntegerType


def process_bronze_table(hotel, source_directory, bronze_directory, spark):
    # prepare arguments
    if hotel not in ['H1', 'H2']:
        raise ValueError('hotel must be H1 or H2')
    csv_file_path = os.path.join(source_directory, hotel + '.csv')

    # load data; preserve source row order for a repeatable Booking_ID in Silver
    # The source has no booking key. csv.reader also handles quoted CSV fields.
    with open(csv_file_path, encoding='utf-8-sig', newline='') as source:
        reader = csv.reader(source)
        columns = [column.strip() for column in next(reader)]
        rows = [row + [number] for number, row in enumerate(reader, start=1)]
    schema = StructType([StructField(column, StringType()) for column in columns]
                        + [StructField('SourceRowNumber', IntegerType(), False)])
    df = spark.createDataFrame(rows, schema)
    print('loaded from:', csv_file_path, 'row count:', df.count())

    # save bronze table; values stay unchanged and identical bookings are retained
    os.makedirs(bronze_directory, exist_ok=True)
    filepath = os.path.join(bronze_directory, 'bronze_' + hotel + '.csv')
    df.orderBy('SourceRowNumber').toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)
    return df
