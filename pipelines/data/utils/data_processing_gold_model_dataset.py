import os


def process_model_dataset_gold_table(hotel, gold_feature_store_directory, gold_label_store_directory,
                                     gold_model_dataset_directory, spark):
    # connect to gold feature and label tables
    feature_filepath = os.path.join(gold_feature_store_directory, 'gold_feature_store_' + hotel + '.parquet')
    label_filepath = os.path.join(gold_label_store_directory, 'gold_label_store_' + hotel + '.parquet')
    feature_df = spark.read.parquet(feature_filepath)
    label_df = spark.read.parquet(label_filepath).select('Booking_ID', 'IsCanceled')

    # combine features and the existing label for each booking
    df = feature_df.join(label_df, on='Booking_ID', how='inner')

    # save the final joined Gold dataset
    os.makedirs(gold_model_dataset_directory, exist_ok=True)
    filepath = os.path.join(gold_model_dataset_directory, 'gold_model_dataset_' + hotel + '.parquet')
    df.write.mode('overwrite').parquet(filepath)
    print('saved to:', filepath, 'row count:', df.count())
    return df
