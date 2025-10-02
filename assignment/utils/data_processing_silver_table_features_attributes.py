import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table_features_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_features_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # remove _ from Age
    df['Age'].astype(str).str.replace('_', '', regex=False)
    # replace SSN to missing if does not follow format
    ssn_regex = r'^\d{3}-\d{2}-\d{4}$'
    df['SSN'].where(df['SSN'].str.match(ssn_regex), np.nan)
    # replace ______ occupation to NA
    df['Occupation'].astype(str).str.replace('_______', '', regex=False)
    # replace blank whitespace or empty strings with np.nan
    blank_regex = r'^\s*$'
    df['Age'].astype(str).replace(blank_regex, np.nan, regex=True)
    df['SSN'].astype(str).replace(blank_regex, np.nan, regex=True)
    df['Occupation'].replace(blank_regex, np.nan, regex=True)

    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))


    # save silver table - IRL connect to database to write
    partition_name = "silver_features_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df