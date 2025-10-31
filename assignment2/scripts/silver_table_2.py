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

    if not os.path.exists(silver_clickstream_directory):
        os.makedirs(silver_clickstream_directory)

    # Process clickstream
    print(f"  Processing clickstream for {date_str}...")
    
    # --- CORRECTED SECTION ---
    
    # Fixed typo: was "bronze_feature_clickstream"
    bronze_clickstream_filepath = f"{bronze_clickstream_directory}bronze_features_clickstream_{date_str_formatted}.csv"
    silver_clickstream_filepath = f"{silver_clickstream_directory}silver_clickstream_{date_str_formatted}.parquet"

    if os.path.exists(bronze_clickstream_filepath):
        # 1. Load the bronze data
        df_clk = spark.read.csv(bronze_clickstream_filepath, header=True, inferSchema=True)
        # Convert columns to lowercase
        df_clk = df_clk.select([col(c).alias(c.lower()) for c in df_clk.columns])

        # 2. Call the CORRECT util function
        df_clk_processed = utils.data_processing_silver_table.process_df_clickstream(df_clk)
        
        # 3. Save the silver data
        df_clk_processed.write.mode("overwrite").parquet(silver_clickstream_filepath)
        print(f"  Saved clickstream to: {silver_clickstream_filepath}")
        
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