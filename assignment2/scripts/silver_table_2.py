import argparse
import os
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_silver_table

# to call this script: python silver_table_2.py --snapshotdate "2023-01-01"
# This processes clickstream from bronze to silver layer

def main(snapshotdate):
    print('\n\n---starting job: silver_table_2 (clickstream)---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate
    date_str_formatted = date_str.replace('-', '_')

    # create silver datalake directories
    bronze_clickstream_directory = "datamart/bronze/features/clickstream/"
    silver_clickstream_directory = "datamart/silver/clickstream/"
    base_datamart_dir = "datamart/"

    if not os.path.exists(silver_clickstream_directory):
        os.makedirs(silver_clickstream_directory)

    # Process clickstream
    print(f"  Processing clickstream for {date_str}...")
    bronze_clickstream_filepath = f"{bronze_clickstream_directory}bronze_feature_clickstream_{date_str_formatted}.csv"
    silver_clickstream_filepath = f"{silver_clickstream_directory}silver_clickstream_{date_str_formatted}.parquet"

    if os.path.exists(bronze_clickstream_filepath):
        utils.data_processing_silver_table.process_silver_table(
            'clickstream', bronze_clickstream_filepath, silver_clickstream_filepath, spark, base_datamart_dir
        )
    else:
        print(f"  Warning: Bronze clickstream file not found at {bronze_clickstream_filepath}")

    # end spark session
    spark.stop()

    print('\n\n---completed job: silver_table_2 (clickstream)---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)