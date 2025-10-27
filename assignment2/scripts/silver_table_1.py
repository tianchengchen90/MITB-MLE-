import argparse
import os
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_silver_table

# to call this script: python silver_table_1.py --snapshotdate "2023-01-01"
# This processes attributes and financials from bronze to silver layer

def main(snapshotdate):
    print('\n\n---starting job: silver_table_1 (attributes + financials)---\n\n')

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
    bronze_attributes_directory = "datamart/bronze/features/attributes/"
    bronze_financials_directory = "datamart/bronze/features/financials/"
    silver_attributes_directory = "datamart/silver/attributes/"
    silver_financials_directory = "datamart/silver/financials/"
    base_datamart_dir = "datamart/"

    for directory in [silver_attributes_directory, silver_financials_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Process attributes
    print(f"  Processing attributes for {date_str}...")
    bronze_attributes_filepath = f"{bronze_attributes_directory}bronze_features_attributes_{date_str_formatted}.csv"
    silver_attributes_filepath = f"{silver_attributes_directory}silver_attributes_{date_str_formatted}.parquet"

    if os.path.exists(bronze_attributes_filepath):
        utils.data_processing_silver_table.process_silver_table(
            'attributes', bronze_attributes_filepath, silver_attributes_filepath, spark, base_datamart_dir
        )
    else:
        print(f"  Warning: Bronze attributes file not found at {bronze_attributes_filepath}")

    # Process financials
    print(f"  Processing financials for {date_str}...")
    bronze_financials_filepath = f"{bronze_financials_directory}bronze_features_financials_{date_str_formatted}.csv"
    silver_financials_filepath = f"{silver_financials_directory}silver_financials_{date_str_formatted}.parquet"

    if os.path.exists(bronze_financials_filepath):
        utils.data_processing_silver_table.process_silver_table(
            'financials', bronze_financials_filepath, silver_financials_filepath, spark, base_datamart_dir
        )
    else:
        print(f"  Warning: Bronze financials file not found at {bronze_financials_filepath}")

    # end spark session
    spark.stop()

    print('\n\n---completed job: silver_table_1 (attributes + financials)---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)