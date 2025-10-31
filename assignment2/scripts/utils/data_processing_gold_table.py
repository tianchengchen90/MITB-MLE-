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
from pyspark.sql.window import Window

# --- TOP-LEVEL HELPER FUNCTION ---

def _read_silver_snapshots_upto(snapshot_date_str, silver_base_dir, feature_name, spark):
    """
    Reads all parquet files for a given Silver feature (across all snapshots)
    on or before the snapshot_date.
    Returns one unified Spark DataFrame.
    """
    # Note: Original code had "features_clickstream" etc. This logic assumes
    # the silver subfolder name is the same as the feature_name.
    # e.g., features_financials -> silver_base_dir/financials/
    
    # Mapping from the old "features_financials" name to the silver subfolder
    folder_mapping = {
        "features_clickstream": "clickstream",
        "features_attributes": "attributes",
        "features_financials": "financials"
    }
    
    silver_subfolder = folder_mapping.get(feature_name, feature_name)
    feature_dir = os.path.join(silver_base_dir, silver_subfolder)
    
    all_partition_dirs = glob.glob(os.path.join(feature_dir, "*.parquet"))
    
    if not all_partition_dirs:
        print(f"No Silver files found for {feature_name} at {feature_dir}")
        return None

    snapshot_cutoff = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    eligible_files = []
    
    for f in all_partition_dirs:
        try:
            # Extract date string like 2023_01_01 from the filename
            date_str = "_".join(f.split("_")[-3:]).replace(".parquet", "")
            file_date = datetime.strptime(date_str, "%Y_%m_%d").date()
            if file_date <= snapshot_cutoff:
                eligible_files.append(f)
        except Exception:
            print(f"Could not parse date from: {f}")
            continue

    if not eligible_files:
        print(f"No eligible Silver files for {feature_name} on or before {snapshot_date_str}")
        return None

    # Spark reads a list of paths
    df = spark.read.parquet(*eligible_files)
    print(f"Loaded {feature_name}: {len(eligible_files)} snapshots (≤ {snapshot_date_str}), {df.count()} total rows")
    return df


# --- NEW FUNCTION 1: LABEL STORE ---

def create_label_store(snapshot_date_str, silver_loan_daily_dir, gold_label_store_directory, spark, dpd, mob):
    """
    Creates the Gold Label Store for a specific snapshot and label definition.
    """
    print(f"\n--- Starting Gold Label Store (dpd={dpd}, mob={mob}) ---")
    
    # --- Load Loan Data (Current Snapshot Only) ---
    date_str_formatted = snapshot_date_str.replace('-', '_')
    loan_daily_path = os.path.join(silver_loan_daily_dir, f"silver_loan_daily_{date_str_formatted}.parquet")
    
    if not os.path.exists(loan_daily_path):
        print(f"Missing Silver dataset: {loan_daily_path}. Aborting label store.")
        return None
    
    df_loan = spark.read.parquet(loan_daily_path)
    print(f"Loaded {loan_daily_path}: {df_loan.count()} rows")

    # --- PART 1: LABEL STORE CREATION ---

    # get customer at specified mob
    df_label = df_loan.filter(col("mob") == mob)

    # get label
    df_label = df_label.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df_label = df_label.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    
    # select columns to save
    df_label = df_label.select("customer_id", "loan_id", "label", "label_def", "snapshot_date")

    # save gold label store
    os.makedirs(gold_label_store_directory, exist_ok=True)
    label_outpath = os.path.join(gold_label_store_directory, f"gold_label_store_{date_str_formatted}.parquet")

    df_label.write.mode("overwrite").parquet(label_outpath)
    print(f"Label Store saved to: {label_outpath}")
    print("--- Completed Gold Label Store ---")

    return df_label


# --- NEW FUNCTION 2: FEATURE STORE ---

def create_feature_store(snapshot_date_str, silver_base_dir, gold_feature_store_directory, spark):
    """
    Creates the Gold Feature Store by joining all silver features
    as they were at the time of loan origination (mob=0).
    """
    print(f"\n--- Starting Gold Feature Store ---")
    date_str_formatted = snapshot_date_str.replace('-', '_')

    # --- Load Loan Data (Current Snapshot Only) ---
    loan_daily_path = os.path.join(silver_base_dir, "loan_daily", f"silver_loan_daily_{date_str_formatted}.parquet")
    if not os.path.exists(loan_daily_path):
        print(f"Missing Silver dataset: {loan_daily_path}. Aborting feature store.")
        return None
    
    df_loan = spark.read.parquet(loan_daily_path)
    print(f"Loaded {loan_daily_path}: {df_loan.count()} rows")

    # --- Load Feature Data (All Historical Snapshots) ---
    df_click = _read_silver_snapshots_upto(snapshot_date_str, silver_base_dir, "features_clickstream", spark)
    df_attr  = _read_silver_snapshots_upto(snapshot_date_str, silver_base_dir, "features_attributes", spark)
    df_fin   = _read_silver_snapshots_upto(snapshot_date_str, silver_base_dir, "features_financials", spark) 
    
    if df_click is None and df_attr is None and df_fin is None:
        print("No historical feature data found. Aborting feature store.")
        return None

    # --- PART 2: FEATURE STORE CREATION ---

    # Base table: all loans at mob=0 (i.e., when they were opened)
    df_base = df_loan.filter(col("mob") == 0).select("loan_id", "customer_id", "loan_start_date", "snapshot_date")
    print(f"   Month 0 Loans (Base): {df_base.count()} rows")

    # Helper for temporal join (latest snapshot_date ≤ loan_start_date)
    def temporal_join(df_feat, base_df, join_name):
        """
        Temporal join that matches each loan (base_df) with the latest snapshot
        of features (df_feat) where snapshot_date <= loan_start_date.
        """
        if df_feat is None:
            print(f"Skipping {join_name}: no data found")
            return base_df
    
        # Ensure both have proper date types
        df_feat = (
            df_feat
            .withColumnRenamed("snapshot_date", f"{join_name}_snapshot_date")
            .withColumn(f"{join_name}_snapshot_date", F.to_date(f"{join_name}_snapshot_date"))
        )
        base_df = base_df.withColumn("loan_start_date", F.to_date("loan_start_date"))
    
        # Join first by Customer_ID
        joined = base_df.join(df_feat, on="customer_id", how="left")
    
        # Filter only records before or on loan_start_date
        joined = joined.filter(F.col(f"{join_name}_snapshot_date") <= F.col("loan_start_date"))
    
        # Rank snapshots by recency per loan
        window_spec = Window.partitionBy("loan_id").orderBy(F.col(f"{join_name}_snapshot_date").desc())
        ranked = joined.withColumn("rank", F.row_number().over(window_spec))
    
        # Keep only the most recent valid snapshot per loan
        latest = ranked.filter(F.col("rank") == 1).drop("rank")
    
        print(f"Joined {join_name}: picked latest snapshot ≤ loan_start_date")
        
        # Get list of new feature columns
        feat_cols = [c for c in latest.columns if c not in base_df.columns] + ["loan_id"]

        final_df = base_df.join(
            latest.select(feat_cols),
            on="loan_id",
            how="left"
        )
        
        return final_df

    # Join all feature sets
    df_feature = df_base
    
    df_feature = temporal_join(df_attr, df_feature, "attributes")
    print(f"   After joining attributes: {df_feature.count()} rows")
    
    df_feature = temporal_join(df_fin, df_feature, "financials")
    print(f"   After joining financials: {df_feature.count()} rows")
    
    df_feature = temporal_join(df_click, df_feature, "clickstream")
    print(f"   After joining clickstream: {df_feature.count()} rows")

    # Drop helper and PII columns
    cols_to_drop = [
        "loan_start_date", "Name", "ssn", # Drop PII
        "attributes_snapshot_date", "financials_snapshot_date", "clickstream_snapshot_date",
    ]
    
    # Drop columns only if they exist
    for c in cols_to_drop:
        if c in df_feature.columns:
            df_feature = df_feature.drop(c)

    # Save feature store
    os.makedirs(gold_feature_store_directory, exist_ok=True)
    feature_outpath = os.path.join(gold_feature_store_directory, f"gold_feature_store_{date_str_formatted}.parquet")

    df_feature.write.mode("overwrite").parquet(feature_outpath)
    print(f"Feature Store saved to: {feature_outpath}")
    print("--- Completed Gold Feature Store ---")

    return df_feature