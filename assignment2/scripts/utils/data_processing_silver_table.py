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
# --- NEW IMPORTS ---
from collections import Counter
from pyspark.sql.types import MapType


# --- NEW HELPER FUNCTIONS ---
# These functions appear correct as provided.

############################
# Attributes
############################
def process_df_attributes(df):
    """
    Function to process attributes table
    """
    numeric_regex = r'([-+]?\d*\.?\d+)'
    
    # Extract numeric part from string in 'age' column
    df = df.withColumn("age", F.regexp_extract(col("age"), numeric_regex, 1))

    # Define column data types
    columns = {
        'customer_id': StringType(),
        'name': StringType(),
        'age': IntegerType(),
        'ssn': StringType(),
        'occupation': StringType(),
        'snapshot_date': DateType()
    }

    # Cast columns to the proper data type
    for column, new_type in columns.items():
        # Check if column exists before casting
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))

    # Enforce valid age constraints
    df = df.withColumn(
        "age",
        F.when((col("age") >= 0) & (col("age") <= 120), col("age"))  # keep valid
        .otherwise(None)  # redact invalid
    ) 

    # Enforce valid SSN
    df = df.withColumn(
        "ssn",
        F.regexp_extract(col("ssn"), r'^(\d{3}-\d{2}-\d{4})$', 1)
    )
    df = df.withColumn(
        "ssn",
        F.when(col("ssn") == "", None).otherwise(col("ssn"))
    )

    # Null empty occupation
    df = df.withColumn(
        "occupation",
        F.when(col("occupation") == "_______", None).otherwise(col("occupation"))
    )
    
    # Drop PII 'name' column, but keep validated 'ssn' for potential joins
    if 'name' in df.columns:
        df = df.drop('name')
        
    return df

############################
# Clickstream
############################
def process_df_clickstream(df):
    """
    Function to process clickstream table
    """
    # Define column data types
    columns = {
        **{f'fe_{i}': IntegerType() for i in range(1, 21)},
        'customer_id': StringType(),
        'snapshot_date': DateType()
    }

    # Cast columns to the proper data type
    for column, new_type in columns.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))
    return df

############################
# Financials
############################
def split_loan_type(loan_type):
    """
    Utility function to split loan type into frequency table
    """
    if not isinstance(loan_type, str):
        return {}
    
    loans_list = loan_type.replace(' and ', ',').split(',')

    cleaned = [item.strip().replace(' ', '_').lower() for item in loans_list if item.strip() != '']

    return dict(Counter(cleaned))

def process_df_financials(df, silver_db, snapshot_date_str):
    """
    Function to process financials table
    """
    numeric_regex = r'([-+]?\d*\.?\d+)'
    
    columns = {
        'annual_income': FloatType(),
        'monthly_inhand_salary': FloatType(),
        'num_bank_accounts': IntegerType(),
        'num_credit_card': IntegerType(),
        'interest_rate': IntegerType(),
        'num_of_loan': IntegerType(),
        'delay_from_due_date': IntegerType(),
        'num_of_delayed_payment': IntegerType(),
        'changed_credit_limit': FloatType(),
        'num_credit_inquiries': FloatType(),
        'outstanding_debt': FloatType(),
        'credit_utilization_ratio': FloatType(),
        'total_emi_per_month': FloatType(),
        'amount_invested_monthly': FloatType(),
        'monthly_balance': FloatType()
    }

    # Cast columns to the proper data type
    for col_name, dtype in columns.items():
        if col_name in df.columns:
            df = df.withColumn(col_name, F.regexp_extract(col(col_name), numeric_regex, 1))
            df = df.withColumn(col_name, col(col_name).cast(dtype))

    # Split credit history age
    if 'credit_history_age' in df.columns:
        df = df.withColumn("credit_history_age_year",
                           F.regexp_extract(col('credit_history_age'), r'(\d+)\s+Year', 1))
        df = df.withColumn("credit_history_age_year", col("credit_history_age_year").cast(IntegerType()))
        df = df.withColumn("credit_history_age_month",
                           F.regexp_extract(col('credit_history_age'), r'(\d+)\s+Month', 1))
        df = df.withColumn("credit_history_age_month", col("credit_history_age_month").cast(IntegerType()))
        # Drop the original string column
        df = df.drop('credit_history_age')


    # Remove negative values from columns that should not have it
    for column_name in ['num_of_loan', 'delay_from_due_date', 'num_of_delayed_payment']:
        if column_name in df.columns:
            df = df.withColumn(
                column_name,
                F.when(col(column_name) >= 0, col(column_name))  # keep valid
                .otherwise(None)  # redact invalid
            ) 
    
    # Clip outliers to 97th percentile (as per your logic)
    for column_name in ['num_bank_accounts', 'num_credit_card', 'interest_rate', 'num_of_loan', 'num_of_delayed_payment']:
        if column_name in df.columns:
            percentile_value = df.approxQuantile(column_name, [0.97], 0.01)[0]
            # Ensure percentile_value is not None before comparing
            if percentile_value is not None:
                df = df.withColumn(
                    column_name,
                    F.when(col(column_name) > percentile_value, percentile_value)
                    .otherwise(col(column_name))
                )

    # Split payment behaviour
    if 'payment_behaviour' in df.columns:
        payment_behaviour_regex = r'(Low|High)_spent_(Small|Medium|Large)_value'
        df = df.withColumn(
            'payment_behaviour_spent',
            F.regexp_extract(col('payment_behaviour'), payment_behaviour_regex, 1)
        )
        df = df.withColumn(
            'payment_behaviour_spent',
            F.when(col('payment_behaviour_spent') != '', col('payment_behaviour_spent'))
            .otherwise(None)
        )
        df = df.withColumn(
            'payment_behaviour_value',
            F.regexp_extract(col('payment_behaviour'), payment_behaviour_regex, 2)
        )
        df = df.withColumn(
            'payment_behaviour_value',
            F.when(col('payment_behaviour_value') != '', col('payment_behaviour_value'))
            .otherwise(None)
        )
        # Drop the original string column
        df = df.drop('payment_behaviour')


    # Null empty credit_mix
    if 'credit_mix' in df.columns:
        df = df.withColumn(
            "credit_mix",
            F.when(col("credit_mix") == "_", None).otherwise(col("credit_mix"))
        )
    
    ######################################
    # Split loan type into its own table
    ######################################
    if 'type_of_loan' in df.columns:
        df_loan_type = df.select('customer_id', 'snapshot_date', 'type_of_loan')

        # Register helper function as a udf
        split_loan_type_udf = F.udf(split_loan_type, MapType(StringType(), IntegerType()))

        # Apply UDF to column
        df_loan_type = df_loan_type.withColumn("loan_type_counts", split_loan_type_udf(col("type_of_loan")))
        
        # Get all unique loan types present in this batch
        all_keys = (
            df_loan_type.select("loan_type_counts")
            .rdd.flatMap(lambda row: row["loan_type_counts"].keys() if row["loan_type_counts"] else [])
            .distinct()
            .collect()
        )

        # Create individual columns for each loan type
        for key in all_keys:
            df_loan_type = df_loan_type.withColumn(
                key,
                F.coalesce(col("loan_type_counts").getItem(key), F.lit(0))
            )

        # Drop intermedate columns
        df_loan_type = df_loan_type.drop("loan_type_counts", "type_of_loan")
        
        # Save new table
        # We create a specific directory for this new table
        loan_type_dir = os.path.join(silver_db, 'loan_type')
        os.makedirs(loan_type_dir, exist_ok=True)
        
        partition_name = f"silver_loan_type_{snapshot_date_str.replace('-','_')}.parquet"
        filepath = os.path.join(loan_type_dir, partition_name)
        df_loan_type.write.mode("overwrite").parquet(filepath)
        print(f"Saved separate loan_type table to: {filepath}")

        # Drop the original column from the main financials df
        df = df.drop('type_of_loan')

    return df

############################
# LMS
############################
def process_df_lms(df):
    """
    Function to process LMS table
    """
    column_type_map = {
        "loan_id": StringType(),
        "customer_id": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    # Cast columns to proper data type
    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))

    # Add "month on book" column
    if 'installment_num' in df.columns:
        df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # Add "days past due" column
    if 'overdue_amt' in df.columns and 'due_amt' in df.columns and 'snapshot_date' in df.columns:
        df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType()))
        df = df.fillna({"installments_missed": 0})
        df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
        df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))
    
    return df


# --- CORRECTED MAIN FUNCTION ---
# This function's pathing logic has been fixed to be robust.
# The arguments are now 'bronze_base_dir' and 'silver_base_dir'
# to reflect that they should be the root 'datamart/bronze/'
# and 'datamart/silver/' directories.

def process_silver_table(snapshot_date_str, bronze_base_dir, silver_base_dir, spark):
    """
    Processes all bronze files for a given snapshot date and
    transforms them into silver parquet files.
    """
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    date_str_formatted = snapshot_date_str.replace('-', '_')

    # This dictionary maps a dataset name to its path and file components
    # 'key': (bronze_subfolder, bronze_prefix, silver_subfolder, silver_prefix)
    datasets = {
        "loan_daily": (
            "lms", 
            "bronze_lms_loan_daily", 
            "loan_daily", 
            "silver_loan_daily"
        ),
        "features_clickstream": (
            "features/clickstream", 
            "bronze_features_clickstream", 
            "clickstream", 
            "silver_clickstream"
        ),
        "features_attributes": (
            "features/attributes", 
            "bronze_features_attributes", 
            "attributes", 
            "silver_attributes"
        ),
        "features_financials": (
            "features/financials", 
            "bronze_features_financials", 
            "financials", 
            "silver_financials"
        )
    }

    results = {}
    
    # connect to bronze table
    for name, (bronze_sub, bronze_prefix, silver_sub, silver_prefix) in datasets.items():
        
        bronze_filename = f"{bronze_prefix}_{date_str_formatted}.csv"
        filepath = os.path.join(bronze_base_dir, bronze_sub, bronze_filename)

        if not os.path.exists(filepath):
            print(f"Skipping {name}, no Bronze file for {snapshot_date_str} at {filepath}")
            continue
        
        df = spark.read.csv(filepath, header=True, inferSchema=True)
        print('loaded from:', filepath, 'row count:', df.count())

        if df.count() == 0:
            print(f"Skipping {name}, empty dataset for {snapshot_date}")
            continue
    
    
        # Convert all columns to lowercase to match the new processing functions
        df = df.select([col(c).alias(c.lower()) for c in df.columns])
        
        # --- CLEANING LOGIC ---
        
        if name == 'loan_daily':
            df = process_df_lms(df)
        
        elif name == "features_clickstream":
            df = process_df_clickstream(df)
            
        elif name == "features_attributes":
            df = process_df_attributes(df)
            
        elif name == "features_financials":
            # This function also saves the 'loan_type' table as a side-effect
            # It needs the *base* silver directory
            df = process_df_financials(df, silver_base_dir, snapshot_date_str)

        
        # save silver table
        dataset_dir = os.path.join(silver_base_dir, silver_sub)
        os.makedirs(dataset_dir, exist_ok=True)
        outname = f"{silver_prefix}_{date_str_formatted}.parquet"
        outpath = os.path.join(dataset_dir, outname)
        
        df.write.mode("overwrite").parquet(outpath)
        print('saved to:', outpath)

        results[name] = df
        
    return results