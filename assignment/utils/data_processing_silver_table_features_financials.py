import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse
import re

from pyspark.sql.functions import col, lit, when, split, explode, regexp_extract
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    """
    Reads the bronze loan data CSV, cleans the data using PySpark native operations,
    performs outlier detection and imputation on numerical features, 
    transforms categorical/ordinal features, and writes the result to a silver Parquet table.
    """
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    
    # connect to bronze table
    partition_name = "bronze_features_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
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
        
    df = df.withColumn(
        'payment_behaviour',
        F.regexp_replace(F.col('payment_behaviour').cast('string'), '!@9#%8', '')
    )

    # 3. Replace blank whitespace or empty strings with None (NULL)
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
    cols_for_percentile_clip = [
        "annual_income", "monthly_inhand_salary", 
        "changed_credit_limit", "outstanding_debt", 
        "credit_utilization_ratio", "total_emi_per_month", 
        "amount_invested_monthly", "monthly_balance"
    ]
    custom_clip_thresholds = {
        'interest_rate': 40,
        'num_bank_accounts': 15,
        'num_credit_card': 20,
        'num_credit_inquiries': 30,
        'num_of_delayed_payment': 30,
        'delay_from_due_date': 100
    }
    cols_for_custom_clip = list(custom_clip_thresholds.keys())
    all_numerical_cols_for_stats = cols_for_percentile_clip + cols_for_custom_clip
    stats_exprs = [F.percentile_approx(F.col(c).cast(FloatType()), 0.5).alias(f"{c}_median") for c in all_numerical_cols_for_stats]
    stats_exprs += [F.percentile_approx(F.col(c).cast(FloatType()), 0.999).alias(f"{c}_p999") for c in cols_for_percentile_clip]

    try:
        stats_row = df.agg(*stats_exprs).collect()[0]
    except IndexError:
        print("Warning: Could not calculate statistics. Using default imputation values.")
        stats_row = {}
        
    stats_dict = stats_row.asDict()
    median_map = {c: stats_dict.get(f"{c}_median", 0.0) for c in all_numerical_cols_for_stats}
    p999_map = {c: stats_dict.get(f"{c}_p999", 1e9) for c in cols_for_percentile_clip}
    
    print("\n--- 99.9th Percentile (Outlier Thresholds) for EDA ---")
    pprint.pprint(p999_map)
    print("-----------------------------------------------------\n")
    
    print("Applying custom clipping and validation rules...")
    for col_name, upper_bound in custom_clip_thresholds.items():
        median_val = median_map.get(col_name, 0.0)
        df = df.withColumn(col_name, F.col(col_name).cast(FloatType()))
        df = df.withColumn(col_name, F.when(F.col(col_name).isNull(), F.lit(median_val)).otherwise(F.col(col_name)))
        df = df.withColumn(col_name, F.when(F.col(col_name) < 0, F.lit(0)).otherwise(F.col(col_name)))
        df = df.withColumn(col_name, F.when(F.col(col_name) > F.lit(upper_bound), F.lit(upper_bound)).otherwise(F.col(col_name)))

    print("Applying 99.9th percentile clipping...")
    for col_name in cols_for_percentile_clip:
        median_val = median_map.get(col_name, 0.0)
        p999_val = p999_map.get(col_name, 1e9)
        df = df.withColumn(col_name, F.col(col_name).cast(FloatType()))
        df = df.withColumn(col_name, F.when(F.col(col_name).isNull(), F.lit(median_val)).otherwise(F.col(col_name)))
        df = df.withColumn(col_name, F.when(F.col(col_name) < 0, F.lit(0)).otherwise(F.col(col_name)))
        df = df.withColumn(col_name, F.when(F.col(col_name) > F.lit(p999_val), F.lit(p999_val)).otherwise(F.col(col_name)))

    # 5. Change nominal to numerical (One-Hot Encoding for Loan Type)
    loan_type_list = [
        "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan", 
        "Mortgage Loan", "Not Specified", "Payday Loan", "Personal Loan", "Student Loan"
    ]
    df_split = df.withColumn("loan_array", F.split(F.col("type_of_loan"), r",\s*"))
    df_exploded = df_split.withColumn("loan_type_clean", F.explode(F.col("loan_array")))
    df_exploded = df_exploded.withColumn("count", F.lit(1))
    grouping_cols = [c for c in df_exploded.columns if c not in ["type_of_loan", "loan_array", "loan_type_clean", "count"]]

    df = (
        df_exploded
        .groupBy(*grouping_cols)
        .pivot("loan_type_clean", loan_type_list)
        .sum("count")
    ).fillna(0)

    # ##################################################################
    # ## ✨ NEW: Convert dummified loan columns to snake_case         ##
    # ##################################################################
    def to_snake_case(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        return s2.replace('-', '_').replace(' ', '_')

    loan_type_cols_snake = [to_snake_case(c) for c in loan_type_list]
    
    print("Renaming loan type columns to snake_case...")
    for original_col, new_col in zip(loan_type_list, loan_type_cols_snake):
        if original_col in df.columns:
            df = df.withColumnRenamed(original_col, new_col)

    # 6. Correct num_of_loan based on the one-hot encoded loan types
    # **MODIFIED**: Use the new snake_cased column names
    sum_loan_types_expr = sum([F.col(c) for c in loan_type_cols_snake])
    
    df = df.withColumn("loan_type_count_sum", sum_loan_types_expr.cast(IntegerType()))
    
    print("Overwriting 'num_of_loan' with calculated count from loan types.")
    df = df.withColumn("num_of_loan", F.col("loan_type_count_sum"))

    # 7. Change ordinal data to scale for credit_mix
    df = df.withColumn("credit_mix_encoded",
        F.when(F.col("credit_mix") == "Good", 3)
        .when(F.col("credit_mix") == "Standard", 2)
        .when(F.col("credit_mix") == "Bad", 1)
        .otherwise(F.lit(None))
    )

    # 8. Augment and adds new columns for credit_history
    df = (
        df
        .withColumn("credit_history_years", F.regexp_extract(F.col("credit_history_age"), r"(\d+)\sYears", 1).cast("integer"))
        .withColumn("credit_history_months", F.regexp_extract(F.col("credit_history_age"), r"(\d+)\sMonths", 1).cast("integer"))
    )
    df = df.withColumn("credit_history_months_total", (F.col("credit_history_years") * 12 + F.col("credit_history_months")).cast(IntegerType()))
    
    # 9. Encode spending levels
    df = df.withColumn("spending_level_encoded",
        F.when(F.col("payment_behaviour").like("High_spent%"), 2)
        .when(F.col("payment_behaviour").like("Low_spent%"), 1)
        .otherwise(F.lit(None))
    )
    
    # 10. Encode payment value
    df = df.withColumn("transaction_value_encoded",
        F.when(F.col("payment_behaviour").contains("Large_value"), 3)
        .when(F.col("payment_behaviour").contains("Medium_value"), 2)
        .when(F.col("payment_behaviour").contains("Small_value"), 1)
        .otherwise(F.lit(None))
    )
    
    df = df.withColumn("snapshot_date", F.lit(snapshot_date))

    # --- End PySpark-Native Data Cleaning ---

    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "customer_id": StringType(), "annual_income": FloatType(), "monthly_inhand_salary": FloatType(),
        "num_bank_accounts": IntegerType(), "num_credit_card": IntegerType(), "interest_rate": IntegerType(),
        "num_of_loan": IntegerType(), "type_of_loan": StringType(), "delay_from_due_date": IntegerType(),
        "num_of_delayed_payment": IntegerType(), "changed_credit_limit": FloatType(), "num_credit_inquiries": IntegerType(),
        "credit_mix": StringType(), "outstanding_debt": FloatType(), "credit_utilization_ratio": FloatType(),
        "credit_history_age": StringType(), "payment_of_min_amount": StringType(), "total_emi_per_month": FloatType(),
        "amount_invested_monthly": FloatType(), "payment_behaviour": StringType(), "monthly_balance": FloatType(),
        "snapshot_date": DateType(), "credit_mix_encoded": IntegerType(), "credit_history_years": IntegerType(),
        "credit_history_months": IntegerType(), "credit_history_months_total": IntegerType(), 
        "spending_level_encoded": IntegerType(), "transaction_value_encoded": IntegerType(),
        "loan_type_count_sum": IntegerType(),
    }
    
    # **MODIFIED**: Add new snake_cased pivoted loan columns to the type map
    for loan_type in loan_type_cols_snake:
        column_type_map[loan_type] = IntegerType() 

    # Final explicit casting to enforce schema
    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))

    # save silver table
    partition_name = "silver_features_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    
    df.write.mode("overwrite").parquet(filepath)
    
    print('saved to:', filepath)
    
    return df