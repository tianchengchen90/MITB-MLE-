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

    # --- CORRECTED SECTION ---

    # Define the BASE directories, which is what process_silver_table expects
    bronze_base_directory = "datamart/bronze/"
    silver_base_directory = "datamart/silver/"


    if not os.path.exists(silver_base_directory):
        os.makedirs(silver_base_directory)

    # run data processing
    # 1. Fixed variable names to match what is defined
    # 2. Using the base directories as required by the util function
    utils.data_processing_silver_table.process_silver_table(
        date_str, 
        bronze_base_directory, 
        silver_base_directory, 
        spark
    )

    # --- END CORRECTED SECTION ---

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