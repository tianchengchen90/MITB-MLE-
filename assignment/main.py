# %%
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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table_feature_clickstream
import utils.data_processing_bronze_table_features_attributes
import utils.data_processing_bronze_table_features_financials
import utils.data_processing_bronze_table_lms

import utils.data_processing_silver_table_feature_clickstream
import utils.data_processing_silver_table_features_attributes
import utils.data_processing_silver_table_lms
import utils.data_processing_silver_table_features_financials
import utils.data_processing_silver_table_aggregated

import utils.data_processing_gold_table_lms
import utils.data_processing_gold_table_features_aggregated



# %% [markdown]
# ## set up pyspark session

# %%
# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# %% [markdown]
# ## set up config

# %%
# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# %%
# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
dates_str_lst

# %% [markdown]
# ## Build Bronze Tables

# %% [markdown]
# ### Loan

# %%
# create bronze datalake for loan
bronze_lms_directory = "datamart/bronze/lms/"

if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

# %%
# run bronze backfill for loan
for date_str in dates_str_lst:
    utils.data_processing_bronze_table_lms.process_bronze_table_lms(date_str, bronze_lms_directory, spark)


# %%
# inspect output for loan
utils.data_processing_bronze_table_lms.process_bronze_table_lms(date_str, bronze_lms_directory, spark).toPandas()

# %% [markdown]
# ### Feature Clickstream

# %%
# create bronze datalake for feature clickstream
bronze_clickstream_directory = "datamart/bronze/feature_clickstream/"

if not os.path.exists(bronze_clickstream_directory):
    os.makedirs(bronze_clickstream_directory)

# %%
# run bronze backfill for feature clickstream
for date_str in dates_str_lst:
    utils.data_processing_bronze_table_feature_clickstream.process_bronze_table_feature_clickstream(date_str, bronze_clickstream_directory, spark)

# %%
# inspect output for feature clickstream
utils.data_processing_bronze_table_feature_clickstream.process_bronze_table_feature_clickstream(date_str, bronze_clickstream_directory, spark).toPandas()

# %% [markdown]
# ### Features attributes

# %%
# create bronze datalake for feature attributes
bronze_attributes_directory = "datamart/bronze/features_attributes/"

if not os.path.exists(bronze_attributes_directory):
    os.makedirs(bronze_attributes_directory)

# %%
# run bronze backfill for feature attributes
for date_str in dates_str_lst:
    utils.data_processing_bronze_table_features_attributes.process_bronze_table_fa(date_str, bronze_attributes_directory, spark)

# %%
# inspect output for feature attributes
utils.data_processing_bronze_table_features_attributes.process_bronze_table_fa(date_str, bronze_attributes_directory, spark).toPandas()

# %% [markdown]
# ### Features financials

# %%
# create bronze datalake for feature financials
bronze_financials_directory = "datamart/bronze/features_financials/"

if not os.path.exists(bronze_financials_directory):
    os.makedirs(bronze_financials_directory)

# %%
# run bronze backfill for feature financials
for date_str in dates_str_lst:
    utils.data_processing_bronze_table_features_financials.process_bronze_table_features_financials(date_str, bronze_financials_directory, spark)

# %%
# inspect output for feature clickstream
utils.data_processing_bronze_table_features_financials.process_bronze_table_features_financials(date_str, bronze_clickstream_directory, spark).toPandas()

# %% [markdown]
# ## Build Silver Table

# %% [markdown]
# ### Loan

# %%
# create silver datalake
silver_loan_daily_directory = "datamart/silver/loan_daily/"

if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)

# %%
# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table_lms.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)


# %%
utils.data_processing_silver_table_lms.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark).toPandas()

# %% [markdown]
# ## Loan EDA on credit labels

# %%
# set dpd label definition
dpd = 30

# Path to the folder containing CSV files
folder_path = silver_loan_daily_directory

# Read all CSV files into a single DataFrame
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)

# filter only completed loans
df = df.filter(col("loan_start_date") < datetime.strptime("2024-01-01", "%Y-%m-%d"))

# create dpd flag if more than dpd
df = df.withColumn("dpd_flag", F.when(col("dpd") >= dpd, 1).otherwise(0))

# actual bads 
actual_bads_df = df.filter(col("installment_num") == 10)

# prepare for analysis
# df = df.filter(col("installment_num") < 10)

# visualise bad rate
pdf = df.toPandas()

# Group by col_A and count occurrences in col_B
grouped = pdf.groupby('mob')['dpd_flag'].mean()

# Sort the index (x-axis) of the grouped DataFrame
grouped = grouped.sort_index()

# Plotting
grouped.plot(kind='line', marker='o')

plt.title('DPD: '+ str(dpd))
plt.xlabel('mob')
plt.ylabel('bad rate')
plt.grid(True)
plt.show()


# %%
df.show()

# %% [markdown]
# ### Click stream

# %%
# create silver datalake for feature clickstream
silver_clickstream_directory = "datamart/silver/feature_clickstream/"

if not os.path.exists(silver_clickstream_directory):
    os.makedirs(silver_clickstream_directory)

# %%
    # run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table_feature_clickstream.process_silver_table(date_str, bronze_clickstream_directory, silver_clickstream_directory, silver_loan_daily_directory,spark)

# %% [markdown]
# Noted that subsequent to July 2024, alot of customer information does not have clickstream data coming in. 

# %%
utils.data_processing_silver_table_feature_clickstream.process_silver_table(date_str, bronze_clickstream_directory, silver_clickstream_directory, silver_loan_daily_directory,spark).toPandas()

# %% [markdown]
# ### Features Attributes

# %%
# create silver datalake for feature attributes
silver_attributes_directory = "datamart/silver/features_attributes/"

if not os.path.exists(silver_attributes_directory):
    os.makedirs(silver_attributes_directory)

# %%
# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table_features_attributes.process_silver_table_features_attributes(date_str, bronze_attributes_directory, silver_attributes_directory, spark)

# %%
utils.data_processing_silver_table_features_attributes.process_silver_table_features_attributes(date_str, bronze_attributes_directory, silver_attributes_directory, spark).toPandas()

# %% [markdown]
# ### Features Financials

# %%
# create silver datalake for feature financials
silver_financials_directory = "datamart/silver/features_financials/"

if not os.path.exists(silver_financials_directory):
    os.makedirs(silver_financials_directory)

# %%
# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table_features_financials.process_silver_table(date_str, bronze_financials_directory, silver_financials_directory, spark)

# %%
utils.data_processing_silver_table_features_financials.process_silver_table(date_str, bronze_financials_directory, silver_financials_directory, spark).toPandas()

# %% [markdown]
# ### Aggregated features silver

# %%
# create silver datalake for feature financials
silver_aggregated_directory = "datamart/silver/features_aggregated/"

if not os.path.exists(silver_aggregated_directory):
    os.makedirs(silver_aggregated_directory)

# %%
# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table_aggregated.process_silver_aggregation(date_str, silver_clickstream_directory, silver_attributes_directory, silver_financials_directory, silver_aggregated_directory, spark)

# %%
utils.data_processing_silver_table_aggregated.process_silver_aggregation("2023-01-01", silver_clickstream_directory, silver_attributes_directory, silver_financials_directory, silver_aggregated_directory, spark).toPandas()

# %% [markdown]
# ## Build gold table for labels

# %%
# create gold datalake
gold_label_store_directory = "datamart/gold/label_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# %%
# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table_lms.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)


# %%
utils.data_processing_gold_table_lms.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6).dtypes


# %% [markdown]
# ## inspect label store

# %%
folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

df.show()

# %%
df.printSchema()

# %% [markdown]
# ### Gold table for features

# %%
# create gold datalake
gold_feature_store_directory = "datamart/gold/feature_store/"

if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

# %%
# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table_features_aggregated.process_gold_feature_store(date_str, silver_aggregated_directory, gold_feature_store_directory, spark)

# %%
folder_path = gold_feature_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

df.show()

# %%



