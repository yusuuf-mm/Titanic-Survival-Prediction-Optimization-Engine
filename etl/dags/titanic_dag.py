#!/usr/bin/env python3
"""
Titanic ETL Pipeline - Bruin DAG Definition

This DAG orchestrates the Titanic data pipeline:
  ingest → clean → transform → save

Usage:
    # Run locally
    python -m bruin run dags/titanic_dag.py
    
    # Run in cloud (AWS Lambda)
    bruin deploy dags/titanic_dag.py --provider aws
"""

from bruin import DAG, Task


# Define the DAG
titanic_etl_dag = DAG(
    name="titanic_etl_pipeline",
    description="Titanic data ETL pipeline - ingest, clean, transform, save",
    schedule_interval="0 2 * * *",  # Daily at 2 AM
    default_args={
        "owner": "data-team",
        "depends_on_past": False,
        "email_on_failure": False,
    }
)


# Define tasks
ingest_task = Task(
    id="ingest",
    name="Ingest Titanic Data",
    func="etl.tasks.ingest.run",
    description="Download raw Titanic CSV from source and save to S3",
    retries=2,
    retry_delay=60,  # 1 minute
)

clean_task = Task(
    id="clean",
    name="Clean Data",
    func="etl.tasks.clean.run",
    description="Handle missing values, remove duplicates",
    retries=2,
    retry_delay=60,
)

transform_task = Task(
    id="transform",
    name="Transform Data",
    func="etl.tasks.transform.run",
    description="Feature engineering for training",
    retries=2,
    retry_delay=60,
)

save_task = Task(
    id="save",
    name="Save Training Data",
    func="etl.tasks.save.run",
    description="Validate and save final training dataset",
    retries=2,
    retry_delay=60,
)


# Define task dependencies
# ingest → clean → transform → save
titanic_etl_dag >> ingest_task >> clean_task >> transform_task >> save_task


# Set context passing between tasks
# The output of each task is passed to the next task
ingest_task >> clean_task
clean_task >> transform_task
transform_task >> save_task


# Export the DAG
dag = titanic_etl_dag


# Alternative: Simple function-based DAG (easier to understand)
def create_titanic_dag():
    """
    Simple function-based DAG for demonstration.
    This is equivalent to the DAG defined above.
    """
    from bruin import DAG, Task
    
    dag = DAG(
        name="titanic_etl",
        description="Titanic ETL Pipeline"
    )
    
    # Create tasks
    t1 = Task(id="ingest", func="etl.tasks.ingest.run")
    t2 = Task(id="clean", func="etl.tasks.clean.run")
    t3 = Task(id="transform", func="etl.tasks.transform.run")
    t4 = Task(id="save", func="etl.tasks.save.run")
    
    # Set dependencies: t1 → t2 → t3 → t4
    t1 >> t2 >> t3 >> t4
    
    return dag


if __name__ == "__main__":
    # Print DAG info
    print("=" * 60)
    print("Titanic ETL Pipeline - Bruin DAG")
    print("=" * 60)
    print(f"DAG Name: {titanic_etl_dag.name}")
    print(f"Description: {titanic_etl_dag.description}")
    print(f"Schedule: {titanic_etl_dag.schedule_interval}")
    print()
    print("Tasks:")
    print("  1. ingest   - Download raw Titanic data")
    print("  2. clean    - Handle missing values")
    print("  3. transform - Feature engineering")
    print("  4. save     - Finalize training data")
    print()
    print("Flow: ingest → clean → transform → save")
    print("=" * 60)
    print()
    print("To run locally:")
    print("  python -m bruin run etl/dags/titanic_dag.py")
    print()
    print("To deploy to AWS:")
    print("  bruin deploy etl/dags/titanic_dag.py --provider aws")
