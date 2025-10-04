import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql.types import IntegerType, DateType
from datetime import date

def create_gold_ml_table(silver_df, labels_df, oot_start_date, spark):
    """
    Prepares a silver DataFrame for ML by creating train, validation, and Out-of-Time (OOT) sets.
    It performs a time-based split for the OOT set and a random split for train/validation.

    Args:
        silver_df (DataFrame): The cleaned silver DataFrame from the previous step.
        labels_df (DataFrame): A DataFrame containing customer_id and the target label.
        oot_start_date (datetime.date): The cutoff date. Data from this date onwards will be in the OOT set.
        spark (SparkSession): The active SparkSession.

    Returns:
        (DataFrame, DataFrame, DataFrame): A tuple containing the model-ready training,
                                           validation, and OOT DataFrames.
    """
    print("--- Starting Gold Layer Preparation ---")

    # 1. Join Feature Store with Label Store
    df = silver_df.join(labels_df, on="customer_id", how="inner")
    print(f"Joined features with labels. Row count: {df.count()}")

    # 2. Feature Selection
    cols_to_drop = [
        "credit_mix",
        "payment_behaviour",
        "credit_history_age",
        "payment_of_min_amount",
        "credit_history_years",
        "credit_history_months",
        "loan_type_count_sum"
    ]
    df_selected = df.drop(*cols_to_drop)

    label_col = "is_default"
    df_selected = df_selected.withColumn(label_col, F.col(label_col).cast(IntegerType()))

    # snapshot_date is used for splitting, not as a feature
    feature_cols = [c for c in df_selected.columns if c not in [label_col, "customer_id", "snapshot_date"]]
    print("\nFinal features selected for the model:")
    print(feature_cols)
    
    # 3. Final Check for Nulls and Imputation
    df_imputed = df_selected.fillna(0, subset=feature_cols)

    # =========================================================================
    # ## ✨ MODIFICATION: Split into Development (Train/Validation) and OOT   ##
    # ## This is a time-based split to simulate real-world model deployment. ##
    # =========================================================================
    
    # 4. Split data into development and OOT sets based on date
    dev_df_raw = df_imputed.filter(F.col("snapshot_date") < oot_start_date)
    oot_df_raw = df_imputed.filter(F.col("snapshot_date") >= oot_start_date)

    # 5. Split development data into training and validation sets (random split)
    train_df_raw, validation_df_raw = dev_df_raw.randomSplit([0.8, 0.2], seed=42)
    
    print(f"\nTraining set count (before scaling): {train_df_raw.count()}")
    print(f"Validation set count (before scaling): {validation_df_raw.count()}")
    print(f"Out-of-Time (OOT) set count (before scaling): {oot_df_raw.count()}")

    # 6. Define the ML Preprocessing Pipeline
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="unscaled_features",
        handleInvalid="skip"
    )

    scaler = StandardScaler(
        inputCol="unscaled_features",
        outputCol="features",
        withStd=True,
        withMean=False
    )
    pipeline = Pipeline(stages=[assembler, scaler])

    # 7. Fit the pipeline ONLY on the training data
    print("\nFitting the preprocessing pipeline on the training data...")
    pipeline_model = pipeline.fit(train_df_raw)

    # 8. Transform all three data sets using the fitted pipeline
    print("Transforming train, validation, and OOT sets...")
    train_df_transformed = pipeline_model.transform(train_df_raw)
    validation_df_transformed = pipeline_model.transform(validation_df_raw)
    oot_df_transformed = pipeline_model.transform(oot_df_raw)

    # 9. Final Selection for Model Fitting
    def finalize_df(df_in):
        return df_in.select(F.col("customer_id"), F.col(label_col).alias("label"), F.col("features"))

    train_df_final = finalize_df(train_df_transformed)
    validation_df_final = finalize_df(validation_df_transformed)
    oot_df_final = finalize_df(oot_df_transformed)

    print("\n--- Gold Layer Preparation Complete ---")
    print("Final model-ready schema:")
    train_df_final.printSchema()

    return train_df_final, validation_df_final, oot_df_final

# Example Usage (if you were running this script directly)
if __name__ == '__main__':
    from pyspark.sql import SparkSession
    from datetime import timedelta

    spark = SparkSession.builder.appName("GoldLayerPrep").getOrCreate()

    # Create a dummy silver_df with snapshot_date for OOT splitting
    base_date = date(2023, 1, 1)
    silver_df_schema = [
        "customer_id", "snapshot_date", "annual_income", "monthly_inhand_salary", "num_bank_accounts",
        "num_credit_card", "interest_rate", "num_of_loan", "delay_from_due_date",
        "num_of_delayed_payment", "changed_credit_limit", "num_credit_inquiries", "credit_mix",
        "outstanding_debt", "credit_utilization_ratio", "credit_history_age", "payment_of_min_amount",
        "total_emi_per_month", "amount_invested_monthly", "payment_behaviour", "monthly_balance",
        "credit_mix_encoded", "credit_history_years", "credit_history_months", "credit_history_months_total",
        "spending_level_encoded", "transaction_value_encoded", "loan_type_count_sum",
        "auto_loan", "credit_builder_loan", "debt_consolidation_loan", "home_equity_loan",
        "mortgage_loan", "not_specified", "payday_loan", "personal_loan", "student_loan"
    ]
    # Generate 100 rows of data over 10 months
    silver_data = []
    for i in range(100):
        row_date = base_date + timedelta(days=i*3) # Spread data over time
        row = (f"cust{i}", row_date, 50000.0 + i*10, 4000.0, 2, 3, 12, 2, 10, 1, 5000.0, 1, "Good", 15000.0, 0.4, "10 Years 5 Months", "Yes", 500.0, 200.0, "High_spent_Large_value", 3000.0, 3, 10, 5, 125, 2, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0)
        silver_data.append(row)
        
    silver_df = spark.createDataFrame(silver_data, silver_df_schema)
    silver_df = silver_df.withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))
    
    # Dummy labels table
    labels_data = [(f"cust{i}", i % 2) for i in range(100)]
    labels_df = spark.createDataFrame(labels_data, ["customer_id", "is_default"])

    # Define the date to split OOT data. E.g., use the last 2 months for OOT
    oot_cutoff_date = date(2023, 8, 1)

    train_table, validation_table, oot_table = create_gold_ml_table(silver_df, labels_df, oot_cutoff_date, spark)
    
    print(f"\nTotal rows in final Training table: {train_table.count()}")
    train_table.show(5, truncate=False)

    print(f"\nTotal rows in final Validation table: {validation_table.count()}")
    validation_table.show(5, truncate=False)
    
    print(f"\nTotal rows in final OOT table: {oot_table.count()}")
    oot_table.show(5, truncate=False)

    spark.stop()
