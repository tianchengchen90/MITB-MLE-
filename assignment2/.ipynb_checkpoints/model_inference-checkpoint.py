import argparse
import os
import glob
import pandas as pd
import pickle
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# to call this script: python model_train.py --snapshotdate "2024-09-01"

def main(snapshotdate, modelname):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    
    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = modelname
    config["model_bank_directory"] = "model_bank/"
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
    
    pprint.pprint(config)
    

    # --- load model artefact from model bank ---
    # Load the model from the pickle file
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    
    print("Model loaded successfully! " + config["model_artefact_filepath"])


    # --- load feature store ---
    folder_path = "datamart/gold/feature_store/"
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    features_store_sdf = spark.read.option("header", "true").parquet(*files_list)
    features_store_sdf = features_store_sdf.withColumnRenamed(
    "snapshot_date","feature_snapshot_date"
    )

    print("row_count:",features_store_sdf.count())
    
    
    # extract feature store
    features_sdf = features_store_sdf.filter((F.col("feature_snapshot_date") == config["snapshot_date"]))
    print("extracted features_sdf", features_sdf.count(), config["snapshot_date"])


    # --- skip if no data for this snapshot ---
    if features_sdf.count() == 0:
        print(f"No feature data found for snapshot {config['snapshot_date_str']}. Skipping inference.")
        spark.stop()
        print('---skipped job---\n\n')
        return
    
    features_pdf = features_sdf.toPandas()


    # --- preprocess data for modeling ---
    # prepare X_inference
    exclude_cols = [
    "customer_id", "feature_snapshot_date"
    ]
    feature_cols = [c for c in features_pdf.columns if c not in exclude_cols]
    X_inference = features_pdf[feature_cols].copy()

    # Identify categorical columns
    cat_cols = X_inference.select_dtypes(include=['object']).columns.tolist()
    
    # Handle categorical encoding
    for col in cat_cols:
        # Try to reuse training mappings if stored; otherwise, auto-map
        try:
            mapping = model_artefact["preprocessing_transformers"].get(f"{col}_mapping", None)
            if mapping is not None:
                X_inference[col] = X_inference[col].map(mapping)
            else:
                # fallback: derive from inference data
                X_inference[col] = X_inference[col].astype('category').cat.codes
        except Exception as e:
            print(f"Warning: Could not map column {col} ({e}), fallback to category codes.")
            X_inference[col] = X_inference[col].astype('category').cat.codes
    
    # Replace NaN / inf values
    X_inference.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_inference.fillna(X_inference.mean(), inplace=True)
    
    # apply transformer - standard scaler
    X_inference = X_inference.astype(float)
    transformer_stdscaler = model_artefact["preprocessing_transformers"]["stdscaler"]
    X_inference = transformer_stdscaler.transform(X_inference)
    
    print('X_inference', X_inference.shape[0])


    # --- model prediction inference ---
    # load model
    model = model_artefact["model"]
    
    # predict model
    y_inference = model.predict_proba(X_inference)[:, 1]
    
    # prepare output
    y_inference_pdf = features_pdf[["customer_id","feature_snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference

    print("Sample predictions:")
    print(y_inference_pdf.head())
    
    print("Mean predicted probability:", round(y_inference.mean(), 4))
    print("Total records:", len(y_inference_pdf))

    # --- save model inference to datamart gold table ---
    # create bronze datalake
    gold_directory = f"datamart/gold/model_predictions/{config['model_name'][:-4]}/"
    print(gold_directory)
    
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)
    
    # save gold table - IRL connect to database to write
    partition_name = config["model_name"][:-4] + "_predictions_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)

    
    # --- end spark session --- 
    spark.stop()
    
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)