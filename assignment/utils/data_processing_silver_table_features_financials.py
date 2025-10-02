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

from pyspark.sql.functions import col, lit, when, split, explode, regexp_extract
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    """
    Reads the bronze loan data CSV, cleans the data using PySpark native operations,
    performs outlier detection and imputation on numerical features, 
    transforms categorical/ordinal features, and writes the result to a silver Parquet table.
    """
    # prepare arguments
    # Use .date() for compatibility with DateType
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # --- Start PySpark-Native Data Cleaning and Renaming ---
    
    # 1. Rename columns to snake_case first for consistency
    rename_map = {
        "Customer_ID": "customer_id",
        "Annual_Income": "annual_income",
        "Monthly_Inhand_Salary": "monthly_inhand_salary",
        "Num_Bank_Accounts": "num_bank_accounts",
        "Num_Credit_Card": "num_credit_card",
        "Interest_Rate": "interest_rate",
        "Num_of_Loan": "num_of_loan",
        "Type_of_Loan": "type_of_loan",
        "Delay_from_due_date": "delay_from_due_date",
        "Num_of_Delayed_Payment": "num_of_delayed_payment",
        "Changed_Credit_Limit": "changed_credit_limit",
        "Num_Credit_Inquiries": "num_credit_inquiries",
        "Credit_Mix": "credit_mix",
        "Outstanding_Debt": "outstanding_debt",
        "Credit_Utilization_Ratio": "credit_utilization_ratio",
        "Credit_History_Age": "credit_history_age",
        "Payment_of_Min_Amount": "payment_of_min_amount",
        "Total_EMI_per_month": "total_emi_per_month",
        "Amount_invested_monthly": "amount_invested_monthly",
        "Payment_Behaviour": "payment_behaviour",
        "Monthly_Balance": "monthly_balance",
    }

    for old_name, new_name in rename_map.items():
        if old_name in df.columns:
            df = df.withColumnRenamed(old_name, new_name)

    # 2. Clean data: use F.regexp_replace for string cleaning (removing '_')
    cols_to_clean_underscore = [
        "annual_income", 
        "num_of_delayed_payment", 
        "changed_credit_limit", 
        "credit_mix", 
        "outstanding_debt", 
        "amount_invested_monthly", 
        "monthly_balance"
    ]
    for col_name in cols_to_clean_underscore:
        df = df.withColumn(
            col_name, 
            F.regexp_replace(F.col(col_name).cast('string'), '_', '')
        )
        
    # Remove 'danger' string from payment_behaviour
    df = df.withColumn(
        'payment_behaviour',
        F.regexp_replace(F.col('payment_behaviour').cast('string'), '!@9#%8', '')
    )

    # 3. Replace blank whitespace or empty strings (r'^\s*$') with None (NULL)
    blank_regex = r'^\s*$'
    cols_to_clean_blanks = ['type_of_loan', 'credit_mix', 'changed_credit_limit']
    
    for col_name in cols_to_clean_blanks:
        df = df.withColumn(
            col_name,
            F.when(
                F.col(col_name).cast('string').rlike(blank_regex),
                F.lit(None)
            ).otherwise(F.col(col_name))
        )
    
    # 4. Numerical Feature Validation, Outlier Clipping, and Median Imputation
    
    numerical_cols = [
        "annual_income", "monthly_inhand_salary", "num_bank_accounts", 
        "num_credit_card", "interest_rate", "num_of_loan", "delay_from_due_date", 
        "num_of_delayed_payment", "changed_credit_limit", "num_credit_inquiries", 
        "outstanding_debt", "credit_utilization_ratio", "total_emi_per_month", 
        "amount_invested_monthly", "monthly_balance"
    ]
    
    # Calculate median (50th percentile) and high threshold (99.9th percentile)
    stats_exprs = [F.percentile_approx(F.col(c).cast(FloatType()), F.array(F.lit(0.5))).alias(f"{c}_median") for c in numerical_cols]
    stats_exprs += [F.percentile_approx(F.col(c).cast(FloatType()), F.array(F.lit(0.999))).alias(f"{c}_p999") for c in numerical_cols]

    # Collect statistics (action)
    try:
        stats_row = df.agg(*stats_exprs).collect()[0]
    except IndexError:
        print("Warning: Could not calculate statistics for loan features. Using default imputation values.")
        stats_row = {}

    # Extract collected stats into simple dictionaries
    median_map = {c: stats_row[f"{c}_median"][0] if f"{c}_median" in stats_row and stats_row[f"{c}_median"] is not None else 0.0 for c in numerical_cols}
    p999_map = {c: stats_row[f"{c}_p999"][0] if f"{c}_p999" in stats_row and stats_row[f"{c}_p999"] is not None else 1e9 for c in numerical_cols}

    # Validation, Outlier Clipping, and Imputation Loop
    for col_name in numerical_cols:
        median_val = median_map.get(col_name, 0.0)
        p999_val = p999_map.get(col_name, 1e9)

        # Ensure the column is float for robust operations
        df = df.withColumn(col_name, F.col(col_name).cast(FloatType()))
        
        # Clip Anomalously Large Values (> P99.9) and Negative Values (< 0) to NULL
        df = df.withColumn(col_name,
            F.when(F.col(col_name) > F.lit(p999_val), F.lit(None).cast(FloatType()))  # Outlier clip
            .when(F.col(col_name) < F.lit(0), F.lit(None).cast(FloatType()))         # Negative check
            .otherwise(F.col(col_name))
        )

        # Impute NULL values (original missing, negative, or clipped outliers) with Median
        df = df.withColumn(col_name,
            F.when(F.col(col_name).isNull(), F.lit(median_val))
            .otherwise(F.col(col_name))
        )


    # 5. Change nominal to numerical (One-Hot Encoding for Loan Type)
    loan_type_list = [
        "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan", 
        "Mortgage Loan", "Not Specified", "Payday Loan", "Personal Loan", "Student Loan"
    ]
    
    # Split the comma-separated string into an array
    df_split = df.withColumn(
        "loan_array",
        F.split(F.col("type_of_loan"), ",\s*")
    )

    # Explode the array so each loan type gets its own row
    df_exploded = df_split.withColumn("loan_type_clean",
        F.explode(F.col("loan_array"))
    )

    # Add a column with value 1 to count occurrences in the pivot
    df_exploded = df_exploded.withColumn("count", F.lit(1))

    # Group by all original columns (excluding the temporary/transformed ones)
    grouping_cols = [c for c in df_exploded.columns if c not in ["type_of_loan", "loan_array", "loan_type_clean", "count"]]

    df = (
        df_exploded
        # Group by all relevant columns
        .groupBy(*grouping_cols)
        # Pivot the 'loan_type_clean' column using the pre-defined list
        .pivot("loan_type_clean", loan_type_list)
        # Sum the 'count' column (the 1s we added)
        .sum("count")
    ).fillna(0) # Fill NaN values created by the pivot with 0

    # 6. Validation Check: Ensure sum of OHE loan columns equals num_of_loan
    
    # Note: Loan type list columns must exist after the pivot in Step 5
    loan_type_cols = loan_type_list
    
    # Calculate the sum of the one-hot encoded columns
    # We use sum() on a list of PySpark Column objects
    sum_loan_types_expr = sum([F.col(c) for c in loan_type_cols])
    
    # Add a column for the calculated sum (cast to Integer for comparison)
    df = df.withColumn("loan_type_count_sum", sum_loan_types_expr.cast(IntegerType()))
    
    # Add a flag to check consistency: 1 if consistent, 0 if inconsistent
    df = df.withColumn("loan_count_is_valid", 
        F.when(
            F.col("loan_type_count_sum") == F.col("num_of_loan"), 
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    # 7. Change ordinal data to scale for credit_mix
    df = df.withColumn("credit_mix_encoded",
        F.when(F.col("credit_mix") == "Good", 3)      # Highest score for 'Good'
        .when(F.col("credit_mix") == "Standard", 2) # Mid score for 'Standard'
        .when(F.col("credit_mix") == "Bad", 1)      # Lowest score for 'Bad'
        .otherwise(F.lit(None)) # Handle NULL/missing values
    )


    # 8. Augment and adds new columns for credit_history
    df = (
        df
        # Extract the years
        .withColumn(
            "credit_history_years",
            F.regexp_extract(F.col("credit_history_age"), r"(\d+)\sYears", 1).cast("integer")
        )
        # Extract the months
        .withColumn(
            "credit_history_months",
            F.regexp_extract(F.col("credit_history_age"), r"(\d+)\sMonths", 1).cast("integer")
        )
    )
    
    # 9. Encode spending levels
    df = df.withColumn(
        "spending_level_encoded",
        F.when(F.col("payment_behaviour").like("High_spent%"), 2)  # Assign 2 to High_spent
        .when(F.col("payment_behaviour").like("Low_spent%"), 1)   # Assign 1 to Low_spent
        .otherwise(F.lit(None))
    )
    
    # 10. Encode payment value
    df = df.withColumn(
        "transaction_value_encoded",
        F.when(F.col("payment_behaviour").contains("Large_value"), 3)
        .when(F.col("payment_behaviour").contains("Medium_value"), 2)
        .when(F.col("payment_behaviour").contains("Small_value"), 1)
        .otherwise(F.lit(None))
    )
    
    # Add the snapshot_date column
    df = df.withColumn("snapshot_date", F.lit(snapshot_date))

    # --- End PySpark-Native Data Cleaning ---

    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "customer_id": StringType(),
        "annual_income": FloatType(),
        "monthly_inhand_salary": FloatType(),
        "num_bank_accounts": IntegerType(),
        "num_credit_card": IntegerType(),
        "interest_rate": IntegerType(),
        "num_of_loan": IntegerType(),
        "type_of_loan": StringType(), 
        "delay_from_due_date": IntegerType(),
        "num_of_delayed_payment": IntegerType(),
        "changed_credit_limit": FloatType(),
        "num_credit_inquiries": IntegerType(),
        "credit_mix": StringType(),
        "outstanding_debt": FloatType(),
        "credit_utilization_ratio": FloatType(),
        "credit_history_age": StringType(),
        "payment_of_min_amount": StringType(),
        "total_emi_per_month": FloatType(),
        "amount_invested_monthly": FloatType(),
        "payment_behaviour": StringType(),
        "monthly_balance": FloatType(),
        "snapshot_date": DateType(),
        "credit_mix_encoded": IntegerType(),
        "credit_history_years": IntegerType(),
        "credit_history_months": IntegerType(),
        "spending_level_encoded": IntegerType(),
        "transaction_value_encoded": IntegerType(),
        "loan_type_count_sum": IntegerType(), # New validation column
        "loan_count_is_valid": IntegerType(), # New validation flag
    }
    
    # Add pivoted loan columns to the type map (set to IntegerType, 0 or 1)
    for loan_type in loan_type_list:
        column_type_map[loan_type] = IntegerType() 

    # Final explicit casting to enforce schema
    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    
    df.write.mode("overwrite").parquet(filepath)
    
    print('saved to:', filepath)
    
    return df