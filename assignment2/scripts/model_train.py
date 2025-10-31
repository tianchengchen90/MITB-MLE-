import os
import glob
import argparse
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV


# to call this script: python model_train.py --snapshotdate "2024-09-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    
    # --- set up config ---
    model_train_date_str = snapshotdate
    train_test_period_months = 12
    oot_period_months = 2
    train_test_ratio = 0.8
    
    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["train_test_period_months"] = train_test_period_months
    config["oot_period_months"] =  oot_period_months
    config["model_train_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d")
    config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
    config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = oot_period_months)
    config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
    config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = train_test_period_months)
    config["train_test_ratio"] = train_test_ratio 
    
    pprint.pprint(config)
    

    # --- get label store ---
    # connect to label store
    folder_path = "datamart/gold/label_store/"
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    label_store_sdf = spark.read.option("header", "true").parquet(*files_list)
    print("row_count:",label_store_sdf.count())
    # label_store_sdf.show()

    # extract label store
    # labels_sdf = label_store_sdf.filter((F.col("snapshot_date") >= config["train_test_start_date"]) & (F.col("snapshot_date") <= config["oot_end_date"]))
    # print("extracted labels_sdf", labels_sdf.count(), config["train_test_start_date"], config["oot_end_date"])
    labels_sdf = label_store_sdf
    

    # --- get features ---
    folder_path = "datamart/gold/feature_store/"
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    features_store_sdf = spark.read.option("header", "true").parquet(*files_list)
    features_store_sdf = features_store_sdf.drop(
        "attributes_snapshot_date", "financials_snapshot_date", "clickstream_snapshot_date"
    )
    features_store_sdf = features_store_sdf.withColumnRenamed(
        "snapshot_date","feature_snapshot_date"
    )
    
    print("row_count:",features_store_sdf.count())
    # features_store_sdf.show()

    # extract label store
    features_sdf = features_store_sdf.filter((F.col("feature_snapshot_date") >= config["train_test_start_date"]) & (F.col("feature_snapshot_date") <= config["oot_end_date"]))
    
    print("extracted features_sdf", features_sdf.count(), config["train_test_start_date"], config["oot_end_date"])
    
    if features_sdf.count() < 1000:
        print(f"Insufficient features for model training {features_sdf.count()} — skipping model training.")
        spark.stop()
        return

    
    # --- preprocess data for modeling ---
    # prepare data for modeling
    data_pdf = labels_sdf.join(features_sdf, on=["customer_id"], how="inner").toPandas()

    # split data into train - test - oot
    oot_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["oot_start_date"].date()) & (data_pdf['snapshot_date'] <= config["oot_end_date"].date())]
    train_test_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["train_test_start_date"].date()) & (data_pdf['snapshot_date'] <= config["train_test_end_date"].date())]

    if len(train_test_pdf) < 1000:
        print(f"Insufficient train_test data for model training {len(train_test_pdf)} — skipping model training.")
        spark.stop()
        return
    
    exclude_cols = [
    "customer_id", "snapshot_date", "feature_snapshot_date", "label", "loan_id", "label_def"
]
    feature_cols = [c for c in data_pdf.columns if c not in exclude_cols]
    
    print(f"Using {len(feature_cols)} feature columns")
    
    X_oot = oot_pdf[feature_cols]
    y_oot = oot_pdf["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        train_test_pdf[feature_cols], train_test_pdf["label"], 
        test_size= 1 - config["train_test_ratio"],
        random_state=88,     # Ensures reproducibility
        shuffle=True,        # Shuffle the data before splitting
        stratify=train_test_pdf["label"]           # Stratify based on the label column
    )
    
    
    print('X_train', X_train.shape[0])
    print('X_test', X_test.shape[0])
    print('X_oot', X_oot.shape[0])
    print('y_train', y_train.shape[0], round(y_train.mean(),2))
    print('y_test', y_test.shape[0], round(y_test.mean(),2))
    print('y_oot', y_oot.shape[0], round(y_oot.mean(),2))


    # --- preprocess data ---
    # Copy training, test, oot data
    X_train_prep = X_train.copy()
    X_test_prep = X_test.copy()
    X_oot_prep = X_oot.copy()
    
    # Encode categorical columns
    cat_cols = X_train_prep.select_dtypes(include=['object']).columns.tolist()
    print("Categorical columns:", cat_cols)
    
    for col in cat_cols:
        # Use category codes for simplicity
        all_values = pd.concat([X_train_prep[col], X_test_prep[col], X_oot_prep[col]], axis=0)
        mapping = {cat: i for i, cat in enumerate(all_values.astype('category').cat.categories)}
        X_train_prep[col] = X_train_prep[col].map(mapping)
        X_test_prep[col] = X_test_prep[col].map(mapping)
        X_oot_prep[col] = X_oot_prep[col].map(mapping)
    
    # Replace NaN / inf
    for df in [X_train_prep, X_test_prep, X_oot_prep]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.mean(), inplace=True)
    
    # Fit scaler on training data ONLY
    transformer_stdscaler = StandardScaler()
    transformer_stdscaler.fit(X_train_prep)
    
    X_train_processed = transformer_stdscaler.transform(X_train_prep)
    X_test_processed = transformer_stdscaler.transform(X_test_prep)
    X_oot_processed = transformer_stdscaler.transform(X_oot_prep)
    
    # Quick validation
    print('X_train_processed', X_train_processed.shape[0])
    print('X_test_processed', X_test_processed.shape[0])
    print('X_oot_processed', X_oot_processed.shape[0])


    # --- model train model ---
    # Define the XGBoost classifier
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=88)
    
    # Define the hyperparameter space to search
    param_dist = {
        'n_estimators': [25, 50],
        'max_depth': [2, 3],  # lower max_depth to simplify the model
        'learning_rate': [0.01, 0.1],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }
    
    # Create a scorer based on AUC score
    auc_scorer = make_scorer(roc_auc_score)
    
    # Set up the random search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        scoring=auc_scorer,
        n_iter=100,  # Number of iterations for random search
        cv=3,       # Number of folds in cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1   # Use all available cores
    )
    
    # Perform the random search
    random_search.fit(X_train_processed, y_train)
    
    # Output the best parameters and best score
    print("Best parameters found: ", random_search.best_params_)
    print("Best AUC score: ", random_search.best_score_)
    
    # Evaluate the model on the train set
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_train_processed)[:, 1]
    train_auc_score = roc_auc_score(y_train, y_pred_proba)
    print("Train AUC score: ", train_auc_score)
    
    # Evaluate the model on the test set
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]
    test_auc_score = roc_auc_score(y_test, y_pred_proba)
    print("Test AUC score: ", test_auc_score)
    
    # Evaluate the model on the oot set
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_oot_processed)[:, 1]
    oot_auc_score = roc_auc_score(y_oot, y_pred_proba)
    print("OOT AUC score: ", oot_auc_score)
    
    print("TRAIN GINI score: ", round(2*train_auc_score-1,3))
    print("Test GINI score: ", round(2*test_auc_score-1,3))
    print("OOT GINI score: ", round(2*oot_auc_score-1,3))


    # --- save model inference to datamart gold table ---
    model_artefact = {}

    model_artefact['model'] = best_model
    model_artefact['model_version'] = "credit_model_"+config["model_train_date_str"].replace('-','_')
    model_artefact['preprocessing_transformers'] = {}
    model_artefact['preprocessing_transformers']['stdscaler'] = transformer_stdscaler
    model_artefact['data_dates'] = config
    model_artefact['data_stats'] = {}
    model_artefact['data_stats']['X_train'] = X_train.shape[0]
    model_artefact['data_stats']['X_test'] = X_test.shape[0]
    model_artefact['data_stats']['X_oot'] = X_oot.shape[0]
    model_artefact['data_stats']['y_train'] = round(y_train.mean(),2)
    model_artefact['data_stats']['y_test'] = round(y_test.mean(),2)
    model_artefact['data_stats']['y_oot'] = round(y_oot.mean(),2)
    model_artefact['results'] = {}
    model_artefact['results']['auc_train'] = train_auc_score
    model_artefact['results']['auc_test'] = test_auc_score
    model_artefact['results']['auc_oot'] = oot_auc_score
    model_artefact['results']['gini_train'] = round(2*train_auc_score-1,3)
    model_artefact['results']['gini_test'] = round(2*test_auc_score-1,3)
    model_artefact['results']['gini_oot'] = round(2*oot_auc_score-1,3)
    model_artefact['hp_params'] = random_search.best_params_
    
    pprint.pprint(model_artefact)


    # --- save artefact to model bank ---
    # create model_bank dir
    model_bank_directory = "model_bank/auto_ml"
    
    if not os.path.exists(model_bank_directory):
        os.makedirs(model_bank_directory)
    
    # Full path to the file
    file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')
    
    # Write the model to a pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(model_artefact, file)
    
    print(f"Model saved to {file_path}")

    
    # --- end spark session --- 
    spark.stop()
    
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    # parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)
