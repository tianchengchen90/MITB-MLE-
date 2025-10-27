import argparse
import os
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table

# to call this script: python bronze_table_3.py --snapshotdate "2023-01-01"
# For processing of feature_clickstream.csv.csv into bronze layer

def main(snapshotdate):
    print('\n\n---starting job: bronze_table_3 (clickstream)---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate

    # create bronze datalake
    bronze_clickstream_directory = "datamart/bronze/features/clickstream/"

    if not os.path.exists(bronze_clickstream_directory):
        os.makedirs(bronze_clickstream_directory)

    # source data file (relative to /opt/airflow/scripts/)
    csv_file = "../data/feature_clickstream.csv"

    # clickstream file does not have snapshot_date column, so process once without date filter
    utils.data_processing_bronze_table.process_bronze_table(
        csv_file,
        date_str,
        bronze_clickstream_directory,
        "bronze_feature_clickstream",
        spark,
        date_filter_column=None,
    )

    # end spark session
    spark.stop()

    print('\n\n---completed job: bronze_table_3 (clickstream)---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)