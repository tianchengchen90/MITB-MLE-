import argparse
import os
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table

# to call this script: python bronze_table_1.py --snapshotdate "2023-01-01"
# This processes features_attributes.csv into bronze layer

def main(snapshotdate):
    print('\n\n---starting job: bronze_table_1 (attributes)---\n\n')

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
    bronze_attributes_directory = "datamart/bronze/features/attributes/"

    if not os.path.exists(bronze_attributes_directory):
        os.makedirs(bronze_attributes_directory)

    # source data file (relative to /opt/airflow/scripts/)
    csv_file = "data/features_attributes.csv"

    # attributes file does not have snapshot_date column, so process once without date filter
    utils.data_processing_bronze_table.process_bronze_table(
        csv_file,
        date_str,
        bronze_attributes_directory,
        "bronze_features_attributes",
        spark,
        date_filter_column=None,
    )

    # end spark session
    spark.stop()

    print('\n\n---completed job: bronze_table_1 (attributes)---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)