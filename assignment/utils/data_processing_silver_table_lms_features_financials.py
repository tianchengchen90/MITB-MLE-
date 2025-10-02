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


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # remove _ from Annual income, Num_of_Delayed_Payment, Changed_Credit_Limit, Credit_Mix, Outstanding_Debt, Amount_invested_monthly, Monthly_Balance
    df['Annual_Income'].astype(str).str.replace('_', '', regex=False)
    df['Num_of_Delayed_Payment'].astype(str).str.replace('_', '', regex=False)
    df['Changed_Credit_Limit'].astype(str).str.replace('_', '', regex=False)
    df['Credit_Mix'].astype(str).str.replace('_', '', regex=False)
    df['Outstanding_Debt'].astype(str).str.replace('_', '', regex=False)
    df['Amount_invested_monthly'].astype(str).str.replace('_', '', regex=False)
    df['Monthly_Balance'].astype(str).str.replace('_', '', regex=False)
    df['Payment_Behaviour'].astype(str).str.replace('!@9#%8', '', regex=False)

    # replace blank space with np.nan for Type_of_Loan, Credit_Mix, Changed_Credit_Limit, 
    blank_regex = r'^\s*$'
    df['Type_of_Loan'].astype(str).replace(blank_regex, np.nan, regex=True)
    df['Credit_Mix'].astype(str).replace(blank_regex, np.nan, regex=True)
    df['Changed_Credit_Limit'].astype(str).replace(blank_regex, np.nan, regex=True)

    # Change nominal to numerical (UNSURE HOW TO HANDLE TYPE OF LOAN)
    loan_type_list = [
    "Auto Loan",
    "Credit-Builder Loan",
    "Debt Consolidation Loan",
    "Home Equity Loan",
    "Mortgage Loan",
    "Not Specified",
    "Payday Loan",
    "Personal Loan",
    "Student Loan"
]
    df_split = df.withColumn(
    "Loan_Array",
    split(col("Type_of_Loan"), ",\s*")
)

# 2. Explode the array so each loan type gets its own row
    df = df_split.withColumn("Loan_Type_Clean",
    explode(col("Loan_Array"))
    ).select(
    # Keep the original primary key/ID and the new clean loan type column
    df.columns + ["Loan_Type_Clean"]
)

# 3. Add a column with value 1 to count occurrences in the pivot
    df = df.withColumn("count", lit(1))

# The original columns (excluding the exploded ones) serve as the grouping key
    grouping_cols = [c for c in df.columns if c != "Type_of_Loan"]

    df = (
    df
    # Group by all original columns (except the one being transformed)
    .groupBy(*grouping_cols)
    # Pivot the 'Loan_Type_Clean' column using the pre-defined list
    .pivot("Loan_Type_Clean", loan_type_list)
    # Sum the 'count' column (the 1s we added)
    .sum("count")
)

    # Change ordinal data to scale
    df = df.withColumn("Credit_Mix_Encoded",
    when(col("Credit_Mix") == "Good", 3)      # Highest score for 'Good'
    .when(col("Credit_Mix") == "Standard", 2) # Mid score for 'Standard'
    .when(col("Credit_Mix") == "Bad", 1)       # Lowest score for 'Bad'
)


    # Augment and adds new columns for Credit_History
    df = (
    df
    # Extract the years
    .withColumn(
        "Credit_History_Years",
        regexp_extract(col("Credit_History_Age"), r"(\d+)\sYears", 1).cast("integer")
    )
    # Extract the months
    .withColumn(
        "Credit_History_Months",
        regexp_extract(col("Credit_History_Age"), r"(\d+)\sMonths", 1).cast("integer")
    )
)
    # Encode spending levels
    df = df.withColumn(
    "Spending_Level_Encoded",
    when(col("Payment_behaviour").like("High_spent%"), 2)  # Assign 2 to High_spent
    .when(col("Payment_behaviour").like("Low_spent%"), 1)   # Assign 1 to Low_spent
)
    # Encode payment value
    df = df.withColumn(
    "Transaction_Value_Encoded",
    when(col("Payment_behaviour").contains("Large_value"), 3)
    .when(col("Payment_behaviour").contains("Medium_value"), 2)
    .when(col("Payment_behaviour").contains("Small_value"), 1)
)

    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
        "Credit_Mix_Encoded": IntegerType(),
        "Credit_History_Years": IntegerType(),
        "Credit_History_Months": IntegerType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df