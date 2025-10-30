import argparse
import os
import glob
import pickle
import pandas as pd
import numpy as np
import pprint
from datetime import datetime
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

# to call this script:
# python model_monitor.py --snapshotdate "2024-06-01"

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

def main(snapshotdate,modelname):
    print('\n\n---starting model monitoring job---\n\n')

    # --- Configuration ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_bank_directory"] = "model_bank/"
    config["predictions_directory"] = "datamart/gold/model_predictions/"
    config["label_store_directory"] = "datamart/gold/label_store/"
    config["drift_flag_path"] = "datamart/gold/model_drift_detected.flag"
    config["drift_threshold"] = -0.05  # Drop in AUC beyond 5% → drift
    config["model_name"] = modelname
    config["model_bank_directory"] = "model_bank/"
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]

    pprint.pprint(config)

    
    # --- Load latest model artefact ---
    model_files = glob.glob(os.path.join("model_bank/auto_ml", "*.pkl"))
    if not model_files:
        print("No model found in model_bank — skipping monitoring.")
        
    else: 
        latest_model_file = max(model_files, key=os.path.getctime)
        print(f"Found latest model: {latest_model_file}")
        
        with open(latest_model_file, "rb") as f:
            lastest_model_artefact = pickle.load(f)
        
        lastest_model_version = lastest_model_artefact["model_version"]
        lastest_training_auc = lastest_model_artefact["results"]["auc_oot"]
        print(f"Lastest Model version: {lastest_model_version}, training OOT AUC: {lastest_training_auc:.4f}")
    
    with open(config['model_artefact_filepath'], "rb") as f:
        prd_model_artefact = pickle.load(f)
    
    prd_model_version = prd_model_artefact["model_version"]
    prd_training_auc = prd_model_artefact["results"]["auc_oot"]
    print(f"Production Model version: {prd_model_version}, training OOT AUC: {prd_training_auc:.4f}\n")

    
    # --- Load latest predictions ---
    pred_folder = os.path.join(config["predictions_directory"], prd_model_version)
    prediction_files = glob.glob(os.path.join(pred_folder, "*.parquet"))
    
    if not prediction_files:
        print(f"No prediction files found under {pred_folder}.")
        return
    
    preds_df = pd.concat([pd.read_parquet(f) for f in prediction_files], ignore_index=True)
    preds_df['feature_snapshot_date'] = pd.to_datetime(preds_df['feature_snapshot_date'])
    
    preds_df = preds_df[preds_df["feature_snapshot_date"] <= config["snapshot_date"]]
    
    print(f"Predictions loaded for snapshot <= {config['snapshot_date_str']}: {len(preds_df)} rows")

    
    # --- Load matching label data ---
    label_files = glob.glob(os.path.join(config["label_store_directory"], "*.parquet"))
    if not label_files:
        print("No label store files found.")
        return

    label_df = pd.concat([pd.read_parquet(f) for f in label_files], ignore_index=True)
    label_df['snapshot_date'] = pd.to_datetime(label_df['snapshot_date'])
       print(f"Labels loaded: {len(label_df)} rows")

    
    # --- Merge predictions with labels ---
    merged_df = pd.merge(
        preds_df,
        label_df[["customer_id", "label"]],
        on="customer_id",
        how="inner"
    )

    if merged_df.empty:
        print("No matching rows between predictions and labels. Skipping monitoring.")
        return

    print(f"Merged dataframe: {merged_df.shape[0]} rows")

    
    # --- Compute latest performance ---
    lastest_merged_df = merged_df[merged_df["feature_snapshot_date"] == config["snapshot_date"]]
    if len(lastest_merged_df) == 0:
        print(f"No label store files found for for snapshot = {config['snapshot_date_str']} .")
    
    else:
        latest_auc = roc_auc_score(lastest_merged_df["label"], lastest_merged_df["model_predictions"])
        print(f"Latest snapshot AUC: {latest_auc:.4f}")
        print(f"Training OOT AUC: {prd_training_auc:.4f}")
        
        # Compute AUC difference vs model's oot AUC
        auc_diff = latest_auc - prd_training_auc
        print(f"AUC difference from training OOT: {auc_diff:+.4f}")


        # --- Decide if drift detected ---
        flag_path = config["drift_flag_path"]
    
        if auc_diff < config["drift_threshold"]:
            print(f"Drift detected (AUC drop {auc_diff:+.4f}). Creating drift flag.")
            open(flag_path, "w").close()
        else:
            print("No drift detected.")
            if os.path.exists(flag_path):
                os.remove(flag_path)
                print("Old drift flag removed (if existed).")

    # --- Historical monitoring and visualization ---
    # Group and aggregate monthly performance
    monthly_perf = (
        merged_df.groupby("feature_snapshot_date", group_keys=False)
        .apply(lambda x: pd.Series({
            "AUC": roc_auc_score(x["label"], x["model_predictions"]) if x["label"].notna().any() else np.nan,
            "DefaultRate": x["label"].mean(),
            "PredMean": x["model_predictions"].mean(),
            "PredStd": x["model_predictions"].std(),
            "Count": len(x)
        }))
        .reset_index()
    )

    # Save monthly metrics for dashboard or audits
    monitor_summary_dir = "datamart/gold/model_monitor_summary/"
    os.makedirs(monitor_summary_dir, exist_ok=True)
    monthly_perf.to_parquet(
        os.path.join(monitor_summary_dir, f"model_perf_summary_{snapshotdate.replace('-', '_')}.parquet"),
        index=False
    )

    # --- Visualization ---
    
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_perf["feature_snapshot_date"], monthly_perf["AUC"], marker='o', label='AUC')
    plt.plot(monthly_perf["feature_snapshot_date"], monthly_perf["DefaultRate"], linestyle='--', label='Default Rate')
    plt.title("Model AUC vs Default Rate Over Time")
    plt.xlabel("Snapshot Date")
    plt.legend(); plt.grid(); plt.tight_layout()
    
    reports_dir = "reports/"
    os.makedirs(reports_dir, exist_ok=True)
    plt.savefig(os.path.join(reports_dir, f"model_perf_trend_{snapshotdate.replace('-', '_')}.png"))
    print(f"Saved visualization to {reports_dir}")


    print('\n\n---model monitoring completed---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model monitoring job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)
