import csv
import json
import os


def process_bronze_google_trends(source_directory, bronze_directory, spark):
    # connect to the fixed four-term export; skip the two Google metadata lines
    filepath = os.path.join(source_directory, 'google_trends_hotels_flights_weekly_2015_2017.csv')
    with open(filepath, encoding='utf-8-sig', newline='') as source:
        reader = csv.reader(source)
        next(reader)
        next(reader)
        columns = next(reader)
        rows = [row for row in reader if row]
    expected = ['Week', 'Algarve hotels: (Worldwide)', 'Lisbon hotels: (Worldwide)',
                'flights to Faro: (Worldwide)', 'flights to Lisbon: (Worldwide)']
    if columns != expected:
        raise ValueError('Expected the agreed four-term Google Trends export')
    df = spark.createDataFrame(rows, columns)
    print('loaded from:', filepath, 'row count:', df.count())

    # save bronze table; preserve source names and values
    os.makedirs(bronze_directory, exist_ok=True)
    filepath = os.path.join(bronze_directory, 'bronze_google_trends.csv')
    df.orderBy('Week').toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)
    return df


def process_bronze_holidays(source_directory, bronze_directory, spark):
    # connect to source files; flatten the API response without correcting dates yet
    rows = []
    for year in [2015, 2016, 2017]:
        filepath = os.path.join(source_directory, 'holidays_PT_' + str(year) + '.json')
        with open(filepath, encoding='utf-8') as source:
            for row in json.load(source):
                rows.append((row['date'], row['name'], row['localName'],
                             str(row['global']).lower(), row['countryCode']))
    df = spark.createDataFrame(rows, 'date string, name string, localName string, global string, countryCode string')
    print('loaded from:', source_directory, 'row count:', df.count())

    # save bronze table
    os.makedirs(bronze_directory, exist_ok=True)
    filepath = os.path.join(bronze_directory, 'bronze_holidays.csv')
    df.orderBy('date', 'name').toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)
    return df


def process_bronze_exchange_rates(source_directory, bronze_directory, spark):
    # connect to source file
    filepath = os.path.join(source_directory, 'ecb_eurusd_2014_2017.csv')
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', df.count())

    # save bronze table; retain the downloaded ECB columns
    os.makedirs(bronze_directory, exist_ok=True)
    filepath = os.path.join(bronze_directory, 'bronze_exchange_rates.csv')
    df.orderBy('TIME_PERIOD').toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)
    return df
