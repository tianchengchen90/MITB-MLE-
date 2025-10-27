import argparse
import os
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_gold_table

# to call this script: python gold_feature_store.py --snapshotdate "2023-01-01"
# This combines all silver features into gold feature store

def main(snapshotdate):
    print('\n\n---starting job: gold_feature_store---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate

    # create gold datalake directories
    gold_feature_store_directory = "datamart/gold/feature_store/"

    if not os.path.exists(gold_feature_store_directory):
        os.makedirs(gold_feature_store_directory)

    # Define all silver directories
    silver_dirs = {
        'loan_daily': "datamart/silver/loan_daily/",
        'attributes': "datamart/silver/attributes/",
        'financials': "datamart/silver/financials/",
        'clickstream': "datamart/silver/clickstream/"
    }

    # run data processing
    utils.data_processing_gold_table.process_features_gold_table(
        date_str, silver_dirs, gold_feature_store_directory, spark
    )

    # end spark session
    spark.stop()

    print('\n\n---completed job: gold_feature_store---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)