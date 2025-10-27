import argparse
import os
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_silver_table

# to call this script: python silver_label_store.py --snapshotdate "2023-01-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate

    # create silver datalake directories
    bronze_lms_directory = "datamart/bronze/lms/"
    silver_loan_daily_directory = "datamart/silver/loan_daily/"
    base_datamart_dir = "datamart/"

    if not os.path.exists(silver_loan_daily_directory):
        os.makedirs(silver_loan_daily_directory)

    # run data processing
    utils.data_processing_silver_table.process_silver_table_legacy(
        date_str, bronze_lms_directory, silver_loan_daily_directory, spark, base_datamart_dir
    )

    # end spark session
    spark.stop()

    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)