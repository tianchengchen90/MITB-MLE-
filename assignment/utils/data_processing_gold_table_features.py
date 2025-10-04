import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, monotonically_increasing_id
from pyspark.sql.types import DateType, StringType
from datetime import datetime

# --- Configuration Placeholder (In a real system, this would be passed via arguments/config) ---
SILVER_CLICKSTREAM_DIR = "/mnt/silver/features/clickstream/"
SILVER_ATTRIBUTES_DIR = "/mnt/silver/features/attributes/"
SILVER_FINANCIALS_DIR = "/mnt/silver/features/financials/"
GOLD_FEATURE_STORE_DIR = "/mnt/gold/feature_store/"
# --------------------------------------------------------------------------------------------


def create_gold_feature_store(snapshot_date_str: str, spark: SparkSession):
    """
    Reads the three Silver feature tables, performs a single inner join on 
    customer_id and snapshot_date, and writes the final Gold feature table.
    """
    print(f"Starting Gold Layer aggregation for snapshot date: {snapshot_date_str}")
    
    # Use .date() for compatibility with DateType
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()

    date_formatted = snapshot_date_str.replace('-', '_')
    
    # 1. Define file paths for the Silver partitions
    clickstream_path = SILVER_CLICKSTREAM_DIR + f"silver_feature_clickstream_{date_formatted}.parquet"
    attributes_path = SILVER_ATTRIBUTES_DIR + f"silver_attributes_{date_formatted}.parquet"
    financials_path = SILVER_FINANCIALS_DIR + f"silver_features_financials_{date_formatted}.parquet"
    
    try:
        # 2. Read Silver Tables
        df_clickstream = spark.read.parquet(clickstream_path).alias("cs")
        df_attributes = spark.read.parquet(attributes_path).alias("at")
        df_financials = spark.read.parquet(financials_path).alias("fn")
        
        print(f"Loaded clickstream rows: {df_clickstream.count()}")
        print(f"Loaded attributes rows: {df_attributes.count()}")
        print(f"Loaded financials rows: {df_financials.count()}")
        
    except Exception as e:
        print(f"Error loading a Silver feature table for date {snapshot_date_str}: {e}")
        raise # Stop if core feature data is missing

    # 3. Define the Join Keys
    # All Silver tables were prepared with 'customer_id' and 'snapshot_date'
    # The 'loan_id' from clickstream will also be a key, but only the clickstream table has it.
    join_keys = ["customer_id", "snapshot_date"]

    # 4. Perform the Joins (Two consecutive INNER JOINs)
    
    # Join 1: Clickstream (cs) + Attributes (at)
    # We use inner join to ensure we only keep customers present in BOTH feature sets on the snapshot day.
    df_joined_1 = df_clickstream.join(
        df_attributes,
        on=join_keys,
        how='inner'
    )
    
    # Join 2: Result of Join 1 + Financials (fn)
    df_gold = df_joined_1.join(
        df_financials,
        on=join_keys,
        how='inner'
    )
    
    print(f"Final combined Gold Feature Store rows: {df_gold.count()}")

    # 5. Select Final Columns and Drop Redundant/Intermediate Columns
    
    # Exclude redundant join keys from aliased tables (only keep the ones from df_clickstream's alias)
    # and drop any intermediate/raw columns like 'type_of_loan', 'credit_history_age', etc.
    
    # Get all columns from the joined DataFrame
    all_cols = df_gold.columns
    
    # Columns to be dropped from the final Gold feature store for cleanliness/model readiness
    # Drop original raw columns that were transformed/encoded
    cols_to_drop = [
        "name", "ssn", "credit_history_years", "credit_history_months", 
        "loan_type_count_sum", "loan_array", "loan_type_clean", 
        "type_of_loan", "credit_mix", "credit_history_age", "payment_behaviour"
    ]
    
    # Also drop the loan_id from the gold feature store (temporarily). 
    # It will be joined later with the Label Store for PIT-correctness.
    # We keep it here as it was created in the clickstream script.
    
    df_final_features = df_gold.drop(*cols_to_drop)

    # 6. Optional: Add a globally unique key for the final Gold record
    df_final_features = df_final_features.withColumn(
        "feature_vector_id", 
        F.sha2(F.concat(col("customer_id"), col("snapshot_date").cast(StringType())), 256)
    )
    
    # 7. Write the Gold Feature Store Table
    final_partition_name = f"gold_feature_store_{date_formatted}.parquet"
    final_filepath = GOLD_FEATURE_STORE_DIR + final_partition_name
    
    # Partition by the snapshot_date for time-series querying efficiency
    df_final_features.write.mode("overwrite").partitionBy("snapshot_date").parquet(final_filepath)
    
    print(f"Successfully saved Gold Feature Store to: {final_filepath}")
    
    return df_final_features