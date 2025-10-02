import os
import glob
import pandas as pd # Kept for local reference, but not used for PySpark ops
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
# Removed numpy import as it's not compatible with PySpark column operations (e.g., np.nan)


def process_silver_table_features_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    """
    Reads the bronze attributes CSV, cleans the data using PySpark native operations,
    enforces the schema, and writes the result to a silver Parquet table.
    """
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date() # Use .date() for compatibility with DateType
    
    # connect to bronze table
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # --- Start PySpark-Native Data Cleaning and Renaming ---
    
    # Rename columns to snake_case first
    rename_map = {
        "Customer_ID": "customer_id",
        "Name": "name",
        "Age": "age",
        "SSN": "ssn",
        "Occupation": "occupation",
    }

    for old_name, new_name in rename_map.items():
        # Safely rename columns that exist in the loaded CSV (before inferring schema)
        if old_name in df.columns:
            df = df.withColumnRenamed(old_name, new_name)
    
    # 1. Clean data: enforce schema / data type using PySpark functions
    
    # Remove '_' from age column and cast it to string first
    df = df.withColumn('age', F.regexp_replace(F.col('age').cast('string'), '_', ''))
    
    # NEW: Clean occupation string by removing all characters that are NOT alphabets (a-z, A-Z).
    # This uses regex r'[^a-zA-Z]' to replace symbols, numbers, and spaces with an empty string.
    df = df.withColumn('occupation', 
                       F.regexp_replace(F.col('occupation').cast('string'), r'[^a-zA-Z]', ''))

    # Replace ssn to missing if it does not follow the format (XXX-XX-XXXX)
    ssn_regex = r'^\d{3}-\d{2}-\d{4}$'
    df = df.withColumn('ssn',
        # Keep ssn if it matches the regex, otherwise set to None (NULL)
        F.when(F.col('ssn').rlike(ssn_regex), F.col('ssn'))
        .otherwise(F.lit(None))
    )
    
    # Replace blank whitespace or empty strings (r'^\s*$') with None (NULL)
    # Note: If 'occupation' contained only non-alphabetic characters (e.g., '_______'), 
    # the previous step made it an empty string '', which this step handles by setting to None.
    blank_regex = r'^\s*$'
    
    # Note: Column names updated to snake_case here
    for col_name in ['age', 'ssn', 'occupation']:
        # Check if the string column (after converting to string) matches the blank regex
        # If it matches, set it to None, otherwise keep the original value
        df = df.withColumn(
            col_name,
            F.when(
                F.col(col_name).cast('string').rlike(blank_regex),
                F.lit(None)
            ).otherwise(F.col(col_name))
        )
        
    # Add the snapshot_date column
    df = df.withColumn("snapshot_date", F.lit(snapshot_date))

    # --- End PySpark-Native Data Cleaning ---

    # Dictionary specifying columns and their desired datatypes
    # Keys updated to snake_case
    column_type_map = {
        "customer_id": StringType(),
        "name": StringType(),
        "age": IntegerType(),
        "ssn": StringType(),
        "occupation": StringType(),
        "snapshot_date": DateType(),
    }

    # Final explicit casting to enforce schema
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))


    # save silver table - IRL connect to database to write
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    
    # Increase the number of output files (optional, but better for Parquet)
    # df.coalesce(1).write.mode("overwrite").parquet(filepath) 
    
    df.write.mode("overwrite").parquet(filepath)
    
    print('saved to:', filepath)
    
    return df