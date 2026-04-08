#!/usr/bin/env python3
"""
Transform task - Feature engineering for Titanic dataset.
Creates training-ready features from cleaned data.
"""

import os
import yaml
import boto3
import pandas as pd
import numpy as np
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
    """Load cleaned data from local or S3."""
    local_config = config['local']
    filename = config['pipeline']['cleaned_file']
    
    # Try local first
    if local_config['enabled']:
        local_path = Path(local_config['processed_dir']) / filename
        if local_path.exists():
            logger.info(f"Loading from local: {local_path}")
            return pd.read_csv(local_path)
    
    # Fall back to S3
    if not source_path:
        bucket = get_bucket_name(config, 'processed')
        source_path = f"s3://{bucket}/clean/{filename}"
    
    if source_path.startswith('s3://'):
        parts = source_path[5:].split('/', 1)
        bucket, key = parts[0], parts[1]
        
        s3 = get_s3_client(config)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'])
    
    return pd.read_csv(source_path)


def transform_data(df):
    """
    Transform Titanic dataset with feature engineering.
    
    Features created:
    - family_size: Total family members (sibsp + parch + 1)
    - is_alone: Whether passenger is traveling alone
    - age_group: Binned age categories
    - fare_per_person: Fare divided by family size
    - title: Extracted title from Name
    - has_cabin: Whether cabin info is available
    """
    logger.info(f"Transforming data: {len(df)} rows")
    
    # Family size features
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    logger.info("Created family_size and is_alone features")
    
    # Age grouping
    age_bins = [0, 12, 18, 35, 60, 100]
    age_labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
    df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    logger.info("Created age_group feature")
    
    # Fare per person
    df['fare_per_person'] = df['Fare'] / df['family_size']
    df['fare_per_person'].fillna(df['Fare'], inplace=True)
    logger.info("Created fare_per_person feature")
    
    # Extract title from Name
    def extract_title(name):
        if pd.isna(name):
            return 'Unknown'
        if 'Mr.' in name:
            return 'Mr'
        elif 'Mrs.' in name:
            return 'Mrs'
        elif 'Miss.' in name:
            return 'Miss'
        elif 'Master.' in name:
            return 'Master'
        elif 'Dr.' in name:
            return 'Dr'
        elif 'Rev.' in name:
            return 'Rev'
        else:
            return 'Other'
    
    df['title'] = df['Name'].apply(extract_title)
    logger.info("Created title feature from Name")
    
    # Cabin available flag
    df['has_cabin'] = (df['Cabin'] != 'Unknown').astype(int)
    logger.info("Created has_cabin feature")
    
    # Encode categorical variables
    # Sex: male=1, female=0
    df['sex_encoded'] = (df['Sex'] == 'male').astype(int)
    
    # Embarked: one-hot encode
    df = pd.get_dummies(df, columns=['Embarked'], prefix='embarked')
    
    # Title: one-hot encode
    df = pd.get_dummies(df, columns=['title'], prefix='title')
    
    # Age group: one-hot encode
    df = pd.get_dummies(df, columns=['age_group'], prefix='age_group')
    
    logger.info(f"Transform complete: {len(df.columns)} columns")
    
    return df


def save_data(df, config, output_filename):
    """Save transformed data to local or S3."""
    local_config = config['local']
    
    if local_config['enabled']:
        processed_dir = Path(local_config['processed_dir'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = processed_dir / output_filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved transformed data to local: {output_path}")
        return str(output_path)
    
    s3_client = get_s3_client(config)
    bucket = get_bucket_name(config, 'processed')
    
    csv_data = df.to_csv(index=False)
    s3_client.put_object(
        Bucket=bucket,
        Key=f"processed/{output_filename}",
        Body=csv_data,
        ContentType='text/csv'
    )
    
    s3_path = f"s3://{bucket}/processed/{output_filename}"
    logger.info(f"Saved transformed data to S3: {s3_path}")
    return s3_path


def run(**context):
    """
    Main transform task - loads cleaned data, applies feature engineering,
    and saves training-ready dataset.
    
    Args:
        **context: Bruin context - receives input from clean task
    
    Returns:
        dict: Result with transformed data path and record count
    """
    logger.info("="*50)
    logger.info("TASK: TRANSFORM - Starting feature engineering")
    logger.info("="*50)
    
    config = load_config()
    
    try:
        # Get source path from context
        source_path = context.get('clean', {}).get('output_path')
        
        # Load cleaned data
        df = load_data(config, source_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Transform data
        df = transform_data(df)
        
        # Save transformed data
        output_filename = config['pipeline']['processed_file']
        output_path = save_data(df, config, output_filename)
        
        result = {
            'output_path': output_path,
            'records': len(df),
            'features': len(df.columns),
            'processed_file': output_filename
        }
        
        logger.info(f"Transform completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Transform failed: {e}")
        raise


if __name__ == "__main__":
    result = run()
    print(f"✓ Transform complete: {result}")
