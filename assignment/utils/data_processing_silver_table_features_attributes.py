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

from pyspark.sql.functions import col, lit, when
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table_features_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    """
    Reads the bronze attributes CSV, cleans the data using PySpark native operations,
    one-hot encodes the 'occupation' field, enforces the schema, 
    and writes the result to a silver Parquet table.
    """
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date() # Use .date() for compatibility with DateType
    
    # connect to bronze table
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = os.path.join(bronze_attributes_directory, partition_name)
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
    df = df.withColumn('age', F.regexp_replace(col('age').cast('string'), '_', ''))
    
    # Clean occupation string by removing all characters that are NOT alphabets (a-z, A-Z).
    df = df.withColumn('occupation', 
                       F.regexp_replace(col('occupation').cast('string'), r'[^a-zA-Z]', ''))

    # Replace ssn to missing if it does not follow the format (XXX-XX-XXXX)
    ssn_regex = r'^\d{3}-\d{2}-\d{4}$'
    df = df.withColumn('ssn',
        # Keep ssn if it matches the regex, otherwise set to None (NULL)
        when(col('ssn').rlike(ssn_regex), col('ssn'))
        .otherwise(lit(None))
    )
    
    # Replace blank whitespace or empty strings (r'^\s*$') with None (NULL)
    blank_regex = r'^\s*$'
    
    for col_name in ['age', 'ssn', 'occupation']:
        df = df.withColumn(
            col_name,
            when(
                col(col_name).cast('string').rlike(blank_regex),
                lit(None)
            ).otherwise(col(col_name))
        )
        
    # Add the snapshot_date column
    df = df.withColumn("snapshot_date", lit(snapshot_date))

    # --- NEW: One-Hot Encode Occupation ---
    
    # 1. Get a distinct list of non-null occupations to create columns from.
    # This ensures the schema is consistent.
    occupation_list = [
        row.occupation for row in df.select('occupation').distinct().filter(col('occupation').isNotNull()).collect()
    ]
    print(f"Found {len(occupation_list)} unique occupations to one-hot encode.")

    # 2. Group by all identifying columns and pivot on 'occupation'
    # The pivot operation transforms rows into columns. We fill with 1 if the occupation matches, otherwise it will be null.
    grouping_cols = [c for c in df.columns if c != 'occupation']
    
    if occupation_list: # Only pivot if there are occupations to encode
        df = (
            df.groupBy(*grouping_cols)
              .pivot('occupation', occupation_list)
              .agg(lit(1)) # Place a 1 in the cell for the customer's occupation
              .fillna(0) # Fill all nulls (non-matching occupations) with 0
        )

        # 3. Rename the new columns with a prefix for clarity and drop the original column
        for occ in occupation_list:
            df = df.withColumnRenamed(occ, f"occupation_{occ.lower()}")
            
    # The original 'occupation' column is automatically removed by the groupBy/pivot operation.
    
    # --- End PySpark-Native Data Cleaning & Encoding ---

    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "customer_id": StringType(),
        "name": StringType(),
        "age": IntegerType(),
        "ssn": StringType(),
        "snapshot_date": DateType(),
    }

    # Add the new one-hot encoded occupation columns to the schema map
    if occupation_list:
        for occ in occupation_list:
            column_type_map[f"occupation_{occ.lower()}"] = IntegerType()


    # Final explicit casting to enforce schema
    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))


    # save silver table
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = os.path.join(silver_attributes_directory, partition_name)
    
    df.write.mode("overwrite").parquet(filepath)
    
    print('saved to:', filepath)
    print("Final schema after one-hot encoding:")
    df.printSchema()
    
    return df