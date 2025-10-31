from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import ShortCircuitOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_models_exist_for_inference(**context):
    """
    Check if trained models exist before running inference.

    Models are stored in:
    - /opt/airflow/scripts/model_store/model_1/model.pkl
    - /opt/airflow/scripts/model_store/model_2/model.pkl

    Returns True only if both models exist.
    """
    import os

    model_1_path = "/opt/airflow/scripts/model_store/model_1/model.pkl"
    model_2_path = "/opt/airflow/scripts/model_store/model_2/model.pkl"

    model_1_exists = os.path.exists(model_1_path)
    model_2_exists = os.path.exists(model_2_path)

    if model_1_exists and model_2_exists:
        print("✅ Both models exist. Proceeding with inference.")
        print(f"   Model 1: {model_1_path}")
        print(f"   Model 2: {model_2_path}")
        return True
    else:
        print("⏭️  Skipping inference - models not yet trained.")
        if not model_1_exists:
            print(f"   Missing: {model_1_path}")
        if not model_2_exists:
            print(f"   Missing: {model_2_path}")
        print(
            "   Models will be available after training completes (execution_date >= 2024-06-01)."
        )
        return False


def check_inference_completed_for_monitoring(**context):
    """
    Check if inference completed successfully before running monitoring.

    TEMPORAL REQUIREMENT:
    - Monitoring joins predictions (from mob=0) with labels (from mob=6)
    - Labels on snapshot_date are for loans at mob=6
    - These loans were at mob=0 exactly 6 months earlier
    - So monitoring requires predictions from 6 months before snapshot_date

    Returns True only if:
    1. Models exist (so inference runs)
    2. Predictions from 6 months ago exist (for temporal matching with today's labels)
    """
    import os
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    # Check if models exist
    model_1_path = "/opt/airflow/scripts/model_store/model_1/model.pkl"
    model_2_path = "/opt/airflow/scripts/model_store/model_2/model.pkl"
    models_exist = os.path.exists(model_1_path) and os.path.exists(model_2_path)

    if not models_exist:
        print("⏭️  Skipping monitoring - models don't exist yet.")
        return False

    # Check if predictions from 6 months ago exist
    execution_date_str = context["ds"]  # Format: YYYY-MM-DD
    execution_date = datetime.strptime(execution_date_str, "%Y-%m-%d")
    prediction_date = execution_date - relativedelta(months=6)
    prediction_date_str = prediction_date.strftime("%Y_%m_%d")

    predictions_dir = "/opt/airflow/scripts/datamart/gold/predictions/"
    model_1_predictions = (
        f"{predictions_dir}model_1_predictions_{prediction_date_str}.parquet"
    )
    model_2_predictions = (
        f"{predictions_dir}model_2_predictions_{prediction_date_str}.parquet"
    )

    predictions_exist = os.path.exists(model_1_predictions) and os.path.exists(
        model_2_predictions
    )

    if predictions_exist:
        print(
            "✅ Models and predictions from 6 months ago exist. Proceeding with monitoring."
        )
        print(f"   Prediction date: {prediction_date.strftime('%Y-%m-%d')}")
        print(f"   Label date: {execution_date_str}")
        return True
    else:
        print("⏭️  Skipping monitoring - predictions from 6 months ago don't exist.")
        print(f"   Need predictions from: {prediction_date.strftime('%Y-%m-%d')}")
        print("   This is expected for the first 6 months after inference starts.")
        if not os.path.exists(model_1_predictions):
            print(f"   Missing: {model_1_predictions}")
        if not os.path.exists(model_2_predictions):
            print(f"   Missing: {model_2_predictions}")
        return False


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


with DAG(
    "dag",
    default_args=default_args,
    description="data pipeline run once a month",
    schedule_interval="0 0 1 * *",  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:
    # data pipeline

    # --- label store ---

    dep_check_source_label_data = DummyOperator(task_id="dep_check_source_label_data")

    bronze_label_store = BashOperator(
        task_id="run_bronze_label_store",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 bronze_label_store.py "
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
    (
        dep_check_source_label_data
        >> bronze_label_store
        >> silver_label_store
        >> gold_label_store
        >> label_store_completed
    )

    # --- feature store ---
    dep_check_source_data_bronze_1 = DummyOperator(
        task_id="dep_check_source_data_bronze_1"
    )

    dep_check_source_data_bronze_2 = DummyOperator(
        task_id="dep_check_source_data_bronze_2"
    )

    dep_check_source_data_bronze_3 = DummyOperator(
        task_id="dep_check_source_data_bronze_3"
    )

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
    (
        dep_check_source_data_bronze_1
        >> bronze_table_1
        >> silver_table_1
        >> gold_feature_store
    )
    (
        dep_check_source_data_bronze_2
        >> bronze_table_2
        >> silver_table_1
        >> gold_feature_store
    )
    (
        dep_check_source_data_bronze_3
        >> bronze_table_3
        >> silver_table_2
        >> gold_feature_store
    )
    gold_feature_store >> feature_store_completed

    # --- model inference ---

    # Check if models exist before attempting inference
    # NOTE: Runs in parallel with training. On first run (2024-12-01), inference skips
    # because models don't exist yet. seed_inference_backfill handles this by creating
    # predictions after training completes. Subsequent runs use existing models immediately.
    check_models_for_inference = ShortCircuitOperator(
        task_id="check_models_for_inference",
        python_callable=check_models_exist_for_inference,
        provide_context=True,
    )

    model_inference_start = DummyOperator(task_id="model_inference_start")

    model_1_inference = BashOperator(
        task_id="model_1_inference",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 model_1_inference.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    model_2_inference = BashOperator(
        task_id="model_2_inference",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 model_2_inference.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    model_inference_completed = DummyOperator(task_id="model_inference_completed")

    # Define task dependencies to run scripts sequentially
    # Only run inference if models exist
    feature_store_completed >> check_models_for_inference
    check_models_for_inference >> model_inference_start
    model_inference_start >> model_1_inference >> model_inference_completed
    model_inference_start >> model_2_inference >> model_inference_completed

    # --- model monitoring ---

    # Check if inference completed before attempting monitoring
    check_inference_for_monitoring = ShortCircuitOperator(
        task_id="check_inference_for_monitoring",
        python_callable=check_inference_completed_for_monitoring,
        provide_context=True,
    )

    model_monitor_start = DummyOperator(task_id="model_monitor_start")

    model_1_monitor = BashOperator(
        task_id="model_1_monitor",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 model_1_monitor.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    model_2_monitor = BashOperator(
        task_id="model_2_monitor",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 model_2_monitor.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")

    # Define task dependencies to run scripts sequentially
    # Only run monitoring if inference completed
    model_inference_completed >> check_inference_for_monitoring
    check_inference_for_monitoring >> model_monitor_start
    model_monitor_start >> model_1_monitor >> model_monitor_completed
    model_monitor_start >> model_2_monitor >> model_monitor_completed

    # --- visualization ---

    # Generate performance visualizations after monitoring completes
    visualize_monitoring = BashOperator(
        task_id="visualize_monitoring",
        bash_command=("cd /opt/airflow/scripts && python3 visualize_monitoring.py"),
        trigger_rule="all_success",  # Only run if monitoring succeeds
    )

    # Visualization runs after monitoring (non-blocking)
    model_monitor_completed >> visualize_monitoring

    # --- action evaluation ---

    # Evaluate monitoring metrics against thresholds to determine required action
    evaluate_monitoring_actions = BashOperator(
        task_id="evaluate_monitoring_actions",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 evaluate_monitoring_action.py --model-id model_1 && "
            "python3 evaluate_monitoring_action.py --model-id model_2"
        ),
        trigger_rule="all_success",  # Only run if visualization succeeds
    )

    # Action evaluation runs after visualization
    visualize_monitoring >> evaluate_monitoring_actions

    # --- model auto training ---

    # Check if we have enough data before running model training
    check_training_data = ShortCircuitOperator(
        task_id="check_training_data",
        python_callable=check_sufficient_data_for_training,
        provide_context=True,
    )

    model_automl_start = DummyOperator(task_id="model_automl_start")

    model_1_automl = BashOperator(
        task_id="model_1_automl",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 model_1_automl_v2.py "
            '--snapshotdate "{{ ds }}" '
            "--config model_config.json"
        ),
    )

    model_2_automl = BashOperator(
        task_id="model_2_automl",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 model_2_automl_v2.py "
            '--snapshotdate "{{ ds }}" '
            "--config model_config.json"
        ),
    )

    model_automl_completed = DummyOperator(task_id="model_automl_completed")

    # Ensure the first six months of predictions exist once models are trained
    seed_inference_backfill = BashOperator(
        task_id="seed_inference_backfill",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 seed_inference_backfill.py "
            '--snapshotdate "{{ ds }}" '
            "--backfill-months 8"
        ),
    )

    # Define task dependencies to run scripts sequentially
    # Only run model training if we have sufficient data (>= 2024-06-01)
    feature_store_completed >> check_training_data
    label_store_completed >> check_training_data
    check_training_data >> model_automl_start
    model_automl_start >> model_1_automl >> model_automl_completed
    model_automl_start >> model_2_automl >> model_automl_completed
    model_automl_completed >> seed_inference_backfill