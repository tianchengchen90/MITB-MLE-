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


def process_silver_table(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    """
    Reads the bronze clickstream data, enforces schema and naming conventions, 
    and writes the result to a silver Parquet table.
    """
    # prepare arguments
    # Use .date() for compatibility with DateType
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    
    # connect to bronze table
    partition_name = "bronze_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # --- Start PySpark-Native Data Cleaning and Renaming ---
    
    # 1. Rename columns to snake_case for consistency (only Customer_ID needs renaming here)
    if "Customer_ID" in df.columns:
        df = df.withColumnRenamed("Customer_ID", "customer_id")
    
    # 2. Add the snapshot_date column
    df = df.withColumn("snapshot_date", F.lit(snapshot_date))

    # --- End PySpark-Native Data Cleaning ---

    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "fe_1": IntegerType(),
        "fe_2": IntegerType(),
        "fe_3": IntegerType(),
        "fe_4": IntegerType(),
        "fe_5": IntegerType(),
        "fe_6": IntegerType(),
        "fe_7": IntegerType(),
        "fe_8": IntegerType(),
        "fe_9": IntegerType(),
        "fe_10": IntegerType(),
        "fe_11": IntegerType(),
        "fe_12": IntegerType(),
        "fe_13": IntegerType(),
        "fe_14": IntegerType(),
        "fe_15": IntegerType(),
        "fe_16": IntegerType(),
        "fe_17": IntegerType(),
        "fe_18": IntegerType(),
        "fe_19": IntegerType(),
        "fe_20": IntegerType(),
        "customer_id": StringType(), # Updated to use snake_case
        "snapshot_date": DateType(),
    }

    # Final explicit casting to enforce schema
    for column, new_type in column_type_map.items():
        # Only attempt to cast columns that actually exist in the DataFrame
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    
    df.write.mode("overwrite").parquet(filepath)
    
    print('saved to:', filepath)
    
    return df