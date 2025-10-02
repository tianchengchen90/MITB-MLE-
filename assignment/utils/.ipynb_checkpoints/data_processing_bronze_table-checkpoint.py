import os
from datetime import datetime
import pyspark
from pyspark.sql.functions import col

def process_bronze_table(spark: pyspark.sql.SparkSession, 
                         snapshot_date_str: str, 
                         source_csv_path: str, 
                         bronze_directory: str, 
                         bronze_table_prefix: str) -> pyspark.sql.DataFrame:
    """
    Reads data from a source CSV, filters it by the snapshot date, 
    and saves it as a new CSV partition in the specified bronze directory.

    This function generalizes the ingestion process for various bronze tables.

    Args:
        spark (pyspark.sql.SparkSession): The active Spark session.
        snapshot_date_str (str): The snapshot date in 'YYYY-MM-DD' format.
        source_csv_path (str): The file path for the source CSV data.
        bronze_directory (str): The directory to save the bronze table partition.
        bronze_table_prefix (str): The prefix for the output CSV file name 
                                  (e.g., 'bronze_feature_clickstream_').

    Returns:
        pyspark.sql.DataFrame: The filtered Spark DataFrame for the given snapshot date.
    """
    # Prepare arguments
    try:
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    except ValueError:
        print(f"Error: Incorrect date format for '{snapshot_date_str}'. Please use YYYY-MM-DD.")
        return None

    print(f"Processing {bronze_table_prefix} for snapshot date: {snapshot_date_str}")

    # Connect to the source backend and load data
    # In a real-world scenario, this would connect to a source system.
    try:
        df = spark.read.csv(source_csv_path, header=True, inferSchema=True)
        filtered_df = df.filter(col('snapshot_date') == snapshot_date)
    except Exception as e:
        print(f"Error reading or filtering data from {source_csv_path}: {e}")
        return None
    
    row_count = filtered_df.count()
    print(f"Row count for {snapshot_date_str}: {row_count}")

    if row_count == 0:
        print(f"Warning: No data found for snapshot date {snapshot_date_str} in {source_csv_path}.")
    
    # Save the bronze table to the datamart
    # In a real-world scenario, this would write to a database or data lake.
    partition_name = f"{bronze_table_prefix}{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_directory, partition_name)
    
    try:
        # Ensure the target directory exists
        os.makedirs(bronze_directory, exist_ok=True)
        
        # Convert to Pandas to write a single CSV file.
        # For larger datasets, consider using df.write.csv() with repartition(1).
        filtered_df.toPandas().to_csv(filepath, index=False)
        print(f'Successfully saved to: {filepath}')
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")
        return None

    return filtered_df

# Example of how to use the new utility function
if __name__ == '__main__':
    # This is a placeholder for your actual SparkSession initialization
    spark_session = pyspark.sql.SparkSession.builder.appName("BronzeProcessing").getOrCreate()
    
    snapshot_date = "2023-01-01"

    # --- Configuration for each table ---
    # In a real application, you might load this configuration from a file (e.g., YAML, JSON)
    tables_to_process = [
        {
            "source_path": "data/feature_clickstream.csv",
            "bronze_dir": "bronze/feature_clickstream/",
            "prefix": "bronze_feature_clickstream_"
        },
        {
            "source_path": "data/features_attributes.csv",
            "bronze_dir": "bronze/features_attributes/",
            "prefix": "bronze_features_attributes_"
        },
        {
            "source_path": "data/features_financials.csv",
            "bronze_dir": "bronze/features_financials/",
            "prefix": "bronze_features_financials_"
        },
        {
            "source_path": "data/lms_loan_daily.csv",
            "bronze_dir": "bronze/lms_loan_daily/",
            "prefix": "bronze_loan_daily_"
        }
    ]

    # --- Loop through and process each table ---
    for table_config in tables_to_process:
        process_bronze_table(
            spark=spark_session,
            snapshot_date_str=snapshot_date,
            source_csv_path=table_config["source_path"],
            bronze_directory=table_config["bronze_dir"],
            bronze_table_prefix=table_config["prefix"]
        )
        print("-" * 40)
        
    spark_session.stop()
