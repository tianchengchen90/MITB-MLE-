from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.python import ShortCircuitOperator
from datetime import datetime, timedelta
import os
import glob

def check_sufficient_data_for_training(**context):
    """
    Check if we have sufficient data for model training.

    DYNAMIC WINDOWS (relative mode):
      The model config uses relative windows that calculate backwards from snapshot_date.
      With 12-month training + 2-month validation + 2-month test + 1-month OOT,
      and the fact that labels is bult at MOB =6, so
      12 + 2 + 2 + 1 + 6
      we need 23 months of data. Starting from 2023-01-01, the earliest we can train
      is when snapshot_date reaches 2024-12-01.

    FIXED WINDOWS (absolute mode):
      Uses hardcoded dates. Still requires data through 2024-12-01 for OOT period.

    RETRAINING:
      After initial training, this allows monthly retraining with rolling windows.
      To control retraining frequency, adjust the DAG schedule or add custom logic here.

    Returns True only if execution_date >= 2024-12-01.
    """
    execution_date = context["ds"]  # Format: YYYY-MM-DD
    min_date_for_training = "2024-12-01"

    should_train = execution_date >= min_date_for_training

    if should_train:
        print(
            f"✅ Sufficient data available (execution_date={execution_date}). Proceeding with model training."
        )
        print(
            "   Training will use data from calculated temporal windows based on this snapshot_date."
        )
    else:
        print(
            f"⏭️  Skipping model training (execution_date={execution_date} < {min_date_for_training}). Insufficient data."
        )
        print(
            "   Need at least 23 months of data (12 train + 2 val + 2 test + 1 OOT + 6 due to MOB=6)."
        )

    return should_train

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'credit_risk_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for credit risk prediction run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:

    # data pipeline

    # --- label store ---

    dep_check_source_label_data = DummyOperator(task_id="dep_check_source_label_data")

    bronze_label_store = BashOperator(
        task_id='run_bronze_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_label_store = BashOperator(
        task_id="silver_label_store",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 silver_label_store.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    gold_label_store = BashOperator(
        task_id="gold_label_store",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 gold_label_store.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    label_store_completed = DummyOperator(task_id="label_store_completed")

    # Define task dependencies to run scripts sequentially
    dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed

    # --- feature store ---
    dep_check_source_data_bronze_1 = DummyOperator(task_id="dep_check_source_data_bronze_1")
    dep_check_source_data_bronze_2 = DummyOperator(task_id="dep_check_source_data_bronze_2")
    dep_check_source_data_bronze_3 = DummyOperator(task_id="dep_check_source_data_bronze_3")

    bronze_table_1 = BashOperator(
        task_id="bronze_table_1",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 bronze_table_1.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    bronze_table_2 = BashOperator(
        task_id="bronze_table_2",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 bronze_table_2.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    bronze_table_3 = BashOperator(
        task_id="bronze_table_3",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 bronze_table_3.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_table_1 = BashOperator(
        task_id="silver_table_1",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 silver_table_1.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_table_2 = BashOperator(
        task_id="silver_table_2",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 silver_table_2.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    gold_feature_store = BashOperator(
        task_id="gold_feature_store",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 gold_feature_store.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    feature_store_completed = DummyOperator(task_id="feature_store_completed")
    
    # Define task dependencies to run scripts sequentially
    dep_check_source_data_bronze_1 >> bronze_table_1 >> silver_table_1 >> gold_feature_store
    dep_check_source_data_bronze_2 >> bronze_table_2 >> silver_table_1 >> gold_feature_store
    dep_check_source_data_bronze_3 >> bronze_table_3 >> silver_table_2 >> gold_feature_store
    gold_feature_store >> feature_store_completed


    # --- model inference ---
    model_inference_start = DummyOperator(task_id="model_inference_start")

    model_inference = BashOperator(
        task_id='model_inference',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_inference.py --snapshotdate "{{ ds }}" --modelname "credit_model_2024_09_01.pkl"'
        ),
    )
    model_inference_completed = DummyOperator(task_id="model_inference_completed")

    feature_store_completed >> model_inference_start
    label_store_completed >> model_inference_start >> model_inference >> model_inference_completed

    # --- model monitoring ---
    model_monitor_start = DummyOperator(task_id="model_monitor_start")

    model_monitor = BashOperator(
        task_id='model_monitor',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_monitor.py --snapshotdate "{{ ds }}" --modelname "credit_model_2024_09_01.pkl"'
        ),
    )

    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")

    # Define task dependencies to run scripts sequentially
    model_inference_completed >> model_monitor_start >> model_monitor >> model_monitor_completed
    
    # --- model auto training ---

    # Check if we have enough data before running model training
    check_training_data = ShortCircuitOperator(
        task_id="check_training_data",
        python_callable=check_sufficient_data_for_training,
        provide_context=True,
    )

    model_automl_start = DummyOperator(task_id="model_automl_start")
    
    model_1_automl = BashOperator(
        task_id='model_1_automl',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_train.py --snapshotdate "{{ ds }}"'
        ),
    )

    model_automl_completed = DummyOperator(task_id="model_automl_completed")
    
    # Define task dependencies to run scripts sequentially
    feature_store_completed >> check_training_data
    label_store_completed >> check_training_data
    check_training_data >> model_automl_start >> model_1_automl >> model_automl_completed