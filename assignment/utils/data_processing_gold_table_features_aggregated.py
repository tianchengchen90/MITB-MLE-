import os
import pyspark
import pyspark.sql.functions as F
import argparse
from datetime import datetime

# Explicitly import PySpark functions used
from pyspark.sql.functions import col
from pyspark.sql import SparkSession

def analyze_null_counts(df, df_name):
    """
    Calculates and prints the count and percentage of nulls for columns containing nulls.
    (Included for data quality checks on the Gold DataFrame)
    
    Args:
        df (DataFrame): The PySpark DataFrame to analyze.
        df_name (str): A descriptive name for the DataFrame (e.g., 'df_gold').
    """
    total_rows = df.count()
    if total_rows == 0:
        print(f"\n--- Null Analysis for {df_name} ---")
        print("DataFrame is empty. Cannot perform null analysis.")
        print("-----------------------------------")
        return
        
    null_counts = [
        F.sum(F.when(col(c).isNull(), 1).otherwise(0)).alias(c) 
        for c in df.columns
    ]
    
    try:
        null_row = df.agg(*null_counts).collect()[0]
    except Exception as e:
        print(f"Error calculating null counts for {df_name}: {e}")
        return
    
    print(f"\n--- Null Analysis for {df_name} (Total Rows: {total_rows}) ---")
    has_nulls = False
    
    for col_name, null_count in null_row.asDict().items():
        if null_count > 0:
            null_percentage = (null_count / total_rows) * 100
            print(f"| Column: {col_name:30} | Null Count: {null_count:10} | Null %: {null_percentage:6.2f}% |")
            has_nulls = True
            
    if not has_nulls:
        print("✅ Success: No columns contain null values.")
    print("------------------------------------------------------------------\n")


def process_gold_feature_store(snapshot_date_str, silver_aggregated_directory, gold_feature_store_directory, spark):
    """
    Reads the unified Silver Aggregated table, performs final Gold layer transformations,
    and writes the prepared data to the Gold layer (Feature Store staging).
    
    Args:
        snapshot_date_str (str): Snapshot date for partitioning (YYYY-MM-DD).
        silver_aggregated_directory (str): Base directory for the input Silver aggregated tables.
        gold_feature_store_directory (str): Output directory for the Gold layer data.
        spark (SparkSession): The active Spark session.
    """
    # --- 1. CONSTRUCT INPUT FILE PATH ---
    
    # Example partition name format provided by the user
    partition_suffix = snapshot_date_str.replace('-', '_') + '.parquet'
    aggregated_partition_name = "silver_features_aggregated_" + partition_suffix
    
    # Construct the full file path for the input Silver aggregated table
    aggregated_filepath = os.path.join(silver_aggregated_directory, aggregated_partition_name)
    
    print(f"Reading aggregated Silver data from: {aggregated_filepath}")

    # --- 2. READ THE SILVER AGGREGATED TABLE ---
    
    try:
        df_aggregated = spark.read.parquet(aggregated_filepath)
        print(f"Loaded Silver aggregated data. Row count: {df_aggregated.count()}")
    except Exception as e:
        print(f"Error loading aggregated Silver table for date {snapshot_date_str}: {e}")
        # Stop execution if a required table is missing
        return

    # --- 3. GOLD LAYER TRANSFORMATIONS (Feature Store Preparation) ---
    
    df_gold = df_aggregated
    
    # 3.1 FEATURE SELECTION: Drop sensitive or redundant columns before feature store export
    COLUMNS_TO_DROP = [
        "name", 
        "ssn", 
        "credit_mix", 
        "num_of_loan", 
        "credit_history_years", 
        "credit_history_months", 
        "payment_behaviour"
    ]
    
    # Use selectExpr to safely drop columns only if they exist
    # Note: We must ensure we keep all columns *not* in COLUMNS_TO_DROP
    existing_columns = df_gold.columns
    columns_to_keep = [c for c in existing_columns if c not in COLUMNS_TO_DROP]
    
    df_gold = df_gold.select(*columns_to_keep)
    print(f"Dropped columns: {COLUMNS_TO_DROP}")

    # 3.2 Add a final feature, e.g., a simple flag based on existing features
    df_gold = df_gold.withColumn(
        "high_value_customer_flag",
        F.when(
            (F.col("transaction_value_encoded") > 2) & 
            (F.col("spending_level_encoded") == 3), 
            1
        ).otherwise(0)
    )
    
    # 3.3 Final data type checks/casting if required
    # df_gold = df_gold.withColumn("num_credit_inquiries", F.col("num_credit_inquiries").cast("int"))

    print(f"Gold layer processing complete. Final column count: {len(df_gold.columns)}")

    # --- 4. DATA QUALITY CHECK: ANALYZE NULLS ---
    # This check ensures that any new Gold features or previous imputation steps worked correctly.
    analyze_null_counts(df_gold, "df_gold_feature_store")
    
    # --- 5. SAVE THE GOLD LAYER TABLE ---
    
    final_partition_name = "gold_feature_store_" + partition_suffix
    final_filepath = os.path.join(gold_feature_store_directory, final_partition_name)
    
    print(f"Saving Gold layer data to: {final_filepath}")
    
    # Write the final DataFrame, usually partitioned by customer_id or date
    # Here we overwrite the single partition file.
    df_gold.write.mode("overwrite").parquet(final_filepath)
    
    print("Successfully saved Gold layer table.")
    
    return df_gold

# Example of how to run this script from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Gold layer for feature store from Silver Aggregation.")
    parser.add_argument("--snapshot_date", required=True, help="Snapshot date in YYYY-MM-DD format (used for partitioning).")
    parser.add_argument("--silver_aggregated_dir", required=True, help="Input directory for the Silver aggregated table.")
    parser.add_argument("--gold_feature_store_dir", required=True, help="Output directory for the Gold feature store data.")
    
    args = parser.parse_args()

    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName(f"GoldFeatureStore_{args.snapshot_date}") \
        .getOrCreate()

    # Call the main processing function
    process_gold_feature_store(
        snapshot_date_str=args.snapshot_date,
        silver_aggregated_directory=args.silver_aggregated_dir,
        gold_feature_store_directory=args.gold_feature_store_dir,
        spark=spark
    )

    spark.stop()