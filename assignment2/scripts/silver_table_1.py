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

    # --- Define Directories ---
    bronze_attributes_directory = "datamart/bronze/features/attributes/"
    bronze_financials_directory = "datamart/bronze/features/financials/"
    silver_attributes_directory = "datamart/silver/attributes/"
    silver_financials_directory = "datamart/silver/financials/"
    
    # This is the base silver directory, needed by the financials processor
    silver_base_directory = "datamart/silver/"

    for directory in [silver_attributes_directory, silver_financials_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # --- Process attributes ---
    print(f"  Processing attributes for {date_str}...")
    bronze_attributes_filepath = f"{bronze_attributes_directory}bronze_features_attributes_{date_str_formatted}.csv"
    silver_attributes_filepath = f"{silver_attributes_directory}silver_attributes_{date_str_formatted}.parquet"

    if os.path.exists(bronze_attributes_filepath):
        # 1. Load the bronze data
        df_attr = spark.read.csv(bronze_attributes_filepath, header=True, inferSchema=True)
        # Convert columns to lowercase to match util function expectations
        df_attr = df_attr.select([col(c).alias(c.lower()) for c in df_attr.columns])
        
        # 2. Call the CORRECT util function
        df_attr_processed = utils.data_processing_silver_table.process_df_attributes(df_attr)
        
        # 3. Save the silver data
        df_attr_processed.write.mode("overwrite").parquet(silver_attributes_filepath)
        print(f"  Saved attributes to: {silver_attributes_filepath}")
    else:
        print(f"  Warning: Bronze attributes file not found at {bronze_attributes_filepath}")

    # --- Process financials ---
    print(f"  Processing financials for {date_str}...")
    bronze_financials_filepath = f"{bronze_financials_directory}bronze_features_financials_{date_str_formatted}.csv"
    silver_financials_filepath = f"{silver_financials_directory}silver_financials_{date_str_formatted}.parquet"

    if os.path.exists(bronze_financials_filepath):
        # 1. Load the bronze data
        df_fin = spark.read.csv(bronze_financials_filepath, header=True, inferSchema=True)
        # Convert columns to lowercase
        df_fin = df_fin.select([col(c).alias(c.lower()) for c in df_fin.columns])
        
        # 2. Call the CORRECT util function
        #    (It needs the base silver path to save the 'loan_type' table)
        df_fin_processed = utils.data_processing_silver_table.process_df_financials(
            df_fin, 
            silver_base_directory, 
            date_str
        )
        
        # 3. Save the silver data
        df_fin_processed.write.mode("overwrite").parquet(silver_financials_filepath)
        print(f"  Saved financials to: {silver_financials_filepath}")
    else:
        print(f"  Warning: Bronze financials file not found at {bronze_financials_filepath}")

    # end spark session
    spark.stop()

    print('\n\n---completed job: silver_table_1 (attributes + financials)---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-D")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)