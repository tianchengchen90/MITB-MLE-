import os
import pyspark
import pyspark.sql.functions as F
import argparse
from datetime import datetime

# Explicitly import PySpark functions used
from pyspark.sql.functions import col, when
from pyspark.sql import SparkSession

def analyze_null_counts(df, df_name):
    """
    Calculates and prints the count and percentage of nulls for columns containing nulls.
    
    Args:
        df (DataFrame): The PySpark DataFrame to analyze.
        df_name (str): A descriptive name for the DataFrame (e.g., 'df_aggregated').
    """
    total_rows = df.count()
    if total_rows == 0:
        print(f"\n--- Null Analysis for {df_name} ---")
        print("DataFrame is empty. Cannot perform null analysis.")
        print("-----------------------------------")
        return
        
    # Calculate null counts for all columns
    # We use F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)) to count nulls efficiently
    null_counts = [
        F.sum(F.when(col(c).isNull(), 1).otherwise(0)).alias(c) 
        for c in df.columns
    ]
    
    # Execute the aggregation
    # We collect the result, which is a single row with null counts per column
    try:
        null_row = df.agg(*null_counts).collect()[0]
    except Exception as e:
        print(f"Error calculating null counts for {df_name}: {e}")
        return
    
    print(f"\n--- Null Analysis for {df_name} (Total Rows: {total_rows}) ---")
    has_nulls = False
    
    # Iterate through the results and print only columns with nulls
    for col_name, null_count in null_row.asDict().items():
        if null_count > 0:
            null_percentage = (null_count / total_rows) * 100
            # Print with f-string formatting for alignment
            print(f"| Column: {col_name:30} | Null Count: {null_count:10} | Null %: {null_percentage:6.2f}% |")
            has_nulls = True
            
    if not has_nulls:
        print("✅ Success: No columns contain null values.")
    print("------------------------------------------------------------------\n")


def process_silver_aggregation(snapshot_date_str, silver_clickstream_directory, silver_attributes_directory, silver_financials_directory, silver_aggregated_directory, spark):
    """
    Reads the silver layer tables for clickstream, attributes, and financials,
    performs an inner join on 'customer_id' and 'snapshot_date' to create a unified feature set,
    imputes nulls for encoded features using the mode (regardless of percentage),
    performs a final null check, and writes the result to a silver Parquet table.
    """
    # --- 1. CONSTRUCT FILE PATHS FOR INPUT SILVER TABLES ---
    
    # Format the snapshot date for file names (e.g., '2023_01_01')
    partition_suffix = snapshot_date_str.replace('-', '_') + '.parquet'
    
    # Define paths for each of the three silver tables
    clickstream_filepath = os.path.join(silver_clickstream_directory, 'silver_feature_clickstream_' + partition_suffix)
    attributes_filepath = os.path.join(silver_attributes_directory, 'silver_attributes_' + partition_suffix)
    financials_filepath = os.path.join(silver_financials_directory, 'silver_features_financials_' + partition_suffix)
    
    print(f"Reading clickstream data from: {clickstream_filepath}")
    print(f"Reading attributes data from: {attributes_filepath}")
    print(f"Reading financials data from: {financials_filepath}")

    # --- 2. READ THE THREE SILVER TABLES ---
    
    try:
        df_clickstream = spark.read.parquet(clickstream_filepath)
        print(f"Loaded clickstream data. Row count: {df_clickstream.count()}")
        
        df_attributes = spark.read.parquet(attributes_filepath)
        print(f"Loaded attributes data. Row count: {df_attributes.count()}")
        
        df_financials = spark.read.parquet(financials_filepath)
        print(f"Loaded financials data. Row count: {df_financials.count()}")
        
    except Exception as e:
        print(f"Error loading one or more silver tables for date {snapshot_date_str}: {e}")
        # Stop execution if a required table is missing
        return

    # --- 3. PERFORM INNER JOINS TO AGGREGATE FEATURES ---
    
    # Define the common keys for joining
    join_keys = ["customer_id", "snapshot_date"]
    
    # Join attributes with financials first
    df_agg_1 = df_attributes.join(df_financials, on=join_keys, how='inner')
    print(f"Row count after joining attributes and financials: {df_agg_1.count()}")
    
    # Join the result with the clickstream data
    df_aggregated = df_agg_1.join(df_clickstream, on=join_keys, how='inner')
    print(f"Row count after final join with clickstream: {df_aggregated.count()}")
    
    # Optional: Verify schema and final column count
    print("Final aggregated schema:")
    df_aggregated.printSchema()
    print(f"Total number of columns in aggregated table: {len(df_aggregated.columns)}")


    # --- 3.5. IMPUTATION OF ENCODED FIELDS (using Mode for ALL Nulls in specified columns) ---
    
    # List of categorical/ordinal features to impute using mode
    MODE_IMPUTATION_COLS = [
        "credit_mix_encoded", 
        "spending_level_encoded", 
        "transaction_value_encoded",
        "num_credit_inquiries", # Example integer feature that benefits from mode imputation
    ] 

    total_rows = df_aggregated.count()

    if total_rows > 0:
        print(f"\n--- Starting Mode Imputation for Encoded/Categorical Columns (No Threshold Applied) ---")
        for column_to_impute in MODE_IMPUTATION_COLS:
            if column_to_impute in df_aggregated.columns:
                # Check if the column has nulls and calculate the count
                null_count = df_aggregated.filter(F.col(column_to_impute).isNull()).count()
                
                # Calculate ratio for logging, but not for threshold check
                null_ratio = null_count / total_rows if total_rows > 0 else 0

                # Impute if any nulls exist in the specified column
                if null_count > 0:
                    print(f"-> Imputing '{column_to_impute}' (Null Count: {null_count}, Ratio: {null_ratio:.2f}) with mode.")
                    try:
                        # Calculate the mode (the value with the highest count)
                        mode_value = df_aggregated.groupBy(column_to_impute) \
                                                     .count() \
                                                     .orderBy(F.desc("count")) \
                                                     .limit(1) \
                                                     .collect()[0][column_to_impute]
                                                     
                        # Apply the imputation
                        df_aggregated = df_aggregated.fillna(mode_value, subset=[column_to_impute])
                        print(f"-> Successfully imputed '{column_to_impute}' with mode value: {mode_value}.")

                    except Exception as e:
                        print(f"WARNING: Could not calculate or apply mode for '{column_to_impute}': {e}")
                else:
                    print(f"-> Skipping imputation for '{column_to_impute}'. No nulls found.")
    else:
        print("WARNING: Aggregated DataFrame has 0 rows. Skipping mode imputation.")
        
    # --- 4. DATA QUALITY CHECK: ANALYZE NULLS ---
    analyze_null_counts(df_aggregated, "df_aggregated_final")
    
    # --- 5. SAVE THE AGGREGATED SILVER TABLE ---
    
    final_partition_name = "silver_features_aggregated_" + partition_suffix
    final_filepath = os.path.join(silver_aggregated_directory, final_partition_name)
    
    print(f"Saving aggregated silver data to: {final_filepath}")
    
    # Write the final DataFrame to a Parquet file, overwriting if it exists
    df_aggregated.write.mode("overwrite").parquet(final_filepath)
    
    print("Successfully saved aggregated silver table.")
    
    return df_aggregated

# Example of how to run this script from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate silver feature tables.")
    parser.add_argument("--snapshot_date", required=True, help="Snapshot date in YYYY-MM-DD format.")
    parser.add_argument("--silver_clickstream_dir", required=True, help="Input directory for silver clickstream data.")
    parser.add_argument("--silver_attributes_dir", required=True, help="Input directory for silver attributes data.")
    parser.add_argument("--silver_financials_dir", required=True, help="Input directory for silver financials data.")
    parser.add_argument("--silver_aggregated_dir", required=True, help="Output directory for the aggregated silver table.")
    
    args = parser.parse_args()

    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName(f"SilverAggregation_{args.snapshot_date}") \
        .getOrCreate()

    # Call the main processing function
    process_silver_aggregation(
        snapshot_date_str=args.snapshot_date,
        silver_clickstream_directory=args.silver_clickstream_dir,
        silver_attributes_directory=args.silver_attributes_dir,
        silver_financials_directory=args.silver_financials_dir,
        silver_aggregated_directory=args.silver_aggregated_dir,
        spark=spark
    )

    spark.stop()

