import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

# Explicitly import PySpark functions used
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, silver_loan_daily_directory, spark):
    """
    Reads the bronze clickstream data, enforces schema and naming conventions, 
    joins with loan data to assign a loan_id (only keeping records that match the loan start event), 
    and writes the result to a silver Parquet table.
    """
    # prepare arguments
    # Use .date() for compatibility with DateType
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    
    # --- 1. CONNECT TO BRONZE CLICKSTREAM TABLE ---
    
    partition_name = "bronze_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded bronze clickstream from:', filepath, 'row count:', df.count())

    # Rename columns to snake_case for consistency (only Customer_ID needs renaming here)
    if "Customer_ID" in df.columns:
        df = df.withColumnRenamed("Customer_ID", "customer_id")
    
    # Add the snapshot_date column
    df = df.withColumn("snapshot_date", F.lit(snapshot_date))
    
    # --- 2. CONNECT TO SILVER LOAN TABLE AND PREPARE KEYS ---
    
    loan_partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    loan_filepath = silver_loan_daily_directory + loan_partition_name
    
    try:
        df_loan = spark.read.parquet(loan_filepath)
        print('loaded silver loan data from:', loan_filepath, 'row count:', df_loan.count())
    except Exception as e:
        print(f"Error loading silver loan data: {e}. Returning original clickstream data for debugging.")
        # If the dependency is critical, you might want to raise the exception instead of returning.
        return df.withColumn("loan_id", F.lit(None).cast(StringType())) 

    # Extract unique loan identifiers: loan_id, customer_id, and loan_start_date
    df_loan_keys = df_loan.select(
        "loan_id", 
        "customer_id", 
        "loan_start_date"
    ).distinct()
    print('Extracted unique loan keys:', df_loan_keys.count())
    
    # --- 3. INNER JOIN CLICKSTREAM WITH LOAN KEYS (Matching loan start events only) ---

    # Alias for clarity in the join condition
    loan_keys = df_loan_keys.alias("lk")
    clickstream = df.alias("cs")
    
    # Perform an INNER JOIN: Only keep records where both clickstream and loan_keys match the conditions.
    # This filters the clickstream data to only include records associated with a loan start on that date.
    df_joined = clickstream.join(
        loan_keys,
        (col("cs.customer_id") == col("lk.customer_id")) & 
        (col("cs.snapshot_date") == col("lk.loan_start_date")),
        how='inner' # 'inner' to ensure only matches in both sides, so no leakage 
    )
    
    # Select final columns, taking all clickstream columns and adding the matching loan_id
    select_cols = [col("cs." + c) for c in clickstream.columns] + [col("lk.loan_id")]
    
    df = df_joined.select(select_cols)
    print(f"Joined clickstream records (after INNER JOIN): {df.count()}")


    # --- 4. ENFORCE SCHEMA ---

    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "fe_1": IntegerType(), "fe_2": IntegerType(), "fe_3": IntegerType(),
        "fe_4": IntegerType(), "fe_5": IntegerType(), "fe_6": IntegerType(),
        "fe_7": IntegerType(), "fe_8": IntegerType(), "fe_9": IntegerType(),
        "fe_10": IntegerType(), "fe_11": IntegerType(), "fe_12": IntegerType(),
        "fe_13": IntegerType(), "fe_14": IntegerType(), "fe_15": IntegerType(),
        "fe_16": IntegerType(), "fe_17": IntegerType(), "fe_18": IntegerType(),
        "fe_19": IntegerType(), "fe_20": IntegerType(),
        "customer_id": StringType(), 
        "snapshot_date": DateType(),
        "loan_id": StringType(), # New column added via join
    }

    # Final explicit casting to enforce schema
    for column, new_type in column_type_map.items():
        # Only attempt to cast columns that actually exist in the DataFrame
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))

    # --- 5. SAVE SILVER TABLE ---

    final_partition_name = "silver_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    final_filepath = silver_clickstream_directory + final_partition_name
    
    df.write.mode("overwrite").parquet(final_filepath)
    
    print('saved silver clickstream data to:', final_filepath)
    
    return df