#!/usr/bin/env python3
"""
Clean task - Handles missing values, removes duplicates from Titanic dataset.
"""

import os
import yaml
import boto3
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config" / "s3_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_s3_client(config):
    """Initialize S3 client."""
    return boto3.client('s3', region_name=config['aws']['region'])


def get_bucket_name(config, bucket_type):
    """Get S3 bucket name with environment substitution."""
    bucket = config['buckets'][bucket_type]
    env = os.getenv('ENV', 'dev')
    return bucket.format(env=env)


def load_data(config, source_path=None):
    """Load data from local or S3."""
    local_config = config['local']
    filename = config['pipeline']['raw_file']
    
    # Try local first
    if local_config['enabled']:
        local_path = Path(local_config['raw_dir']) / filename
        if local_path.exists():
            logger.info(f"Loading from local: {local_path}")
            return pd.read_csv(local_path)
    
    # Fall back to S3
    if not source_path:
        bucket = get_bucket_name(config, 'raw')
        source_path = f"s3://{bucket}/raw/{filename}"
    
    # Parse S3 path
    if source_path.startswith('s3://'):
        parts = source_path[5:].split('/', 1)
        bucket, key = parts[0], parts[1]
        
        s3 = get_s3_client(config)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'])
    
    return pd.read_csv(source_path)


def clean_data(df):
    """
    Clean the Titanic dataset.
    
    Cleaning steps:
    - Remove duplicate rows
    - Handle missing Age (fill with median)
    - Handle missing Fare (fill with median)
    - Handle missing Embarked (fill with mode)
    - Handle missing Cabin (mark as unknown)
    """
    logger.info(f"Cleaning data: {len(df)} rows")
    original_count = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    duplicates_removed = original_count - len(df)
    logger.info(f"Removed {duplicates_removed} duplicate rows")
    
    # Fill missing Age with median
    age_median = df['Age'].median()
    df['Age'].fillna(age_median, inplace=True)
    logger.info(f"Filled missing Age with median: {age_median}")
    
    # Fill missing Fare with median
    fare_median = df['Fare'].median()
    df['Fare'].fillna(fare_median, inplace=True)
    logger.info(f"Filled missing Fare with median: {fare_median}")
    
    # Fill missing Embarked with mode
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'].fillna(embarked_mode, inplace=True)
    logger.info(f"Filled missing Embarked with mode: {embarked_mode}")
    
    # Mark missing Cabin as 'Unknown'
    df['Cabin'].fillna('Unknown', inplace=True)
    logger.info("Marked missing Cabin as 'Unknown'")
    
    # Log cleaning summary
    logger.info(f"Cleaning complete: {len(df)} rows (removed {duplicates_removed} duplicates)")
    
    return df


def save_data(df, config, output_filename):
    """Save cleaned data to local or S3."""
    local_config = config['local']
    
    # Save locally
    if local_config['enabled']:
        processed_dir = Path(local_config['processed_dir'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = processed_dir / output_filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to local: {output_path}")
        return str(output_path)
    
    # Save to S3
    s3_client = get_s3_client(config)
    bucket = get_bucket_name(config, 'processed')
    
    csv_data = df.to_csv(index=False)
    s3_client.put_object(
        Bucket=bucket,
        Key=f"clean/{output_filename}",
        Body=csv_data,
        ContentType='text/csv'
    )
    
    s3_path = f"s3://{bucket}/clean/{output_filename}"
    logger.info(f"Saved cleaned data to S3: {s3_path}")
    return s3_path


def run(**context):
    """
    Main clean task - loads raw data, cleans it, and saves to processed location.
    
    Args:
        **context: Bruin context - can receive input from previous task (ingest)
    
    Returns:
        dict: Result with cleaned data path and record count
    """
    logger.info("="*50)
    logger.info("TASK: CLEAN - Starting data cleaning")
    logger.info("="*50)
    
    config = load_config()
    
    try:
        # Get source path from context (previous task) or use default
        source_path = context.get('ingest', {}).get('s3_path') or context.get('output_path')
        
        # Load data
        df = load_data(config, source_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Clean data
        df = clean_data(df)
        
        # Save cleaned data
        output_filename = config['pipeline']['cleaned_file']
        output_path = save_data(df, config, output_filename)
        
        result = {
            'output_path': output_path,
            'records': len(df),
            'cleaned_file': output_filename
        }
        
        logger.info(f"Clean completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Clean failed: {e}")
        raise


if __name__ == "__main__":
    result = run()
    print(f"✓ Clean complete: {result}")
