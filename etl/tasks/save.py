#!/usr/bin/env python3
"""
Save task - Finalizes and saves training-ready dataset to S3.
Validates data quality and creates final output for model training.
"""

import os
import yaml
import boto3
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

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
    """Load transformed data from local or S3."""
    local_config = config['local']
    filename = config['pipeline']['processed_file']
    
    if local_config['enabled']:
        local_path = Path(local_config['processed_dir']) / filename
        if local_path.exists():
            logger.info(f"Loading from local: {local_path}")
            return pd.read_csv(local_path)
    
    if not source_path:
        bucket = get_bucket_name(config, 'processed')
        source_path = f"s3://{bucket}/processed/{filename}"
    
    if source_path.startswith('s3://'):
        parts = source_path[5:].split('/', 1)
        bucket, key = parts[0], parts[1]
        
        s3 = get_s3_client(config)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'])
    
    return pd.read_csv(source_path)


def validate_data(df):
    """
    Validate data quality before final save.
    
    Checks:
    - No missing target variable (Survived)
    - Minimum record count
    - Required features present
    """
    logger.info("Validating data quality...")
    
    errors = []
    warnings = []
    
    # Check target variable
    if 'Survived' not in df.columns:
        errors.append("Missing target variable: Survived")
    elif df['Survived'].isna().any():
        warnings.append("Target variable has missing values")
    
    # Check minimum records
    min_records = 100
    if len(df) < min_records:
        warnings.append(f"Low record count: {len(df)} (minimum: {min_records})")
    
    # Check for required features
    required_features = ['Pclass', 'Sex', 'Age', 'Fare']
    for feature in required_features:
        if feature not in df.columns:
            errors.append(f"Missing required feature: {feature}")
    
    # Report validation results
    if errors:
        for error in errors:
            logger.error(f"Validation error: {error}")
        raise ValueError(f"Data validation failed: {errors}")
    
    if warnings:
        for warning in warnings:
            logger.warning(f"Validation warning: {warning}")
    
    logger.info(f"Validation passed: {len(df)} records, {len(df.columns)} features")
    
    return True


def create_metadata(df, config):
    """Create metadata about the dataset."""
    metadata = {
        'created_at': datetime.utcnow().isoformat(),
        'record_count': len(df),
        'feature_count': len(df.columns),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'survival_rate': float(df['Survived'].mean()) if 'Survived' in df.columns else None
    }
    
    logger.info(f"Created metadata: survival_rate={metadata['survival_rate']}")
    
    return metadata


def save_final(df, config, metadata):
    """Save final training dataset and metadata to S3."""
    local_config = config['local']
    env = os.getenv('ENV', 'dev')
    
    # Create versioned filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    versioned_filename = f"titanic_training_v{timestamp}.csv"
    
    if local_config['enabled']:
        processed_dir = Path(local_config['processed_dir'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training data
        data_path = processed_dir / versioned_filename
        df.to_csv(data_path, index=False)
        logger.info(f"Saved training data to: {data_path}")
        
        # Save metadata
        meta_path = processed_dir / f"metadata_v{timestamp}.json"
        import json
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
        
        return {
            'data_path': str(data_path),
            'metadata_path': str(meta_path),
            'version': timestamp
        }
    
    # Save to S3
    s3_client = get_s3_client(config)
    bucket = get_bucket_name(config, 'processed')
    
    # Upload training data
    csv_data = df.to_csv(index=False)
    s3_client.put_object(
        Bucket=bucket,
        Key=f"training/{versioned_filename}",
        Body=csv_data,
        ContentType='text/csv'
    )
    data_s3_path = f"s3://{bucket}/training/{versioned_filename}"
    
    # Upload metadata
    import json
    meta_json = json.dumps(metadata, indent=2)
    s3_client.put_object(
        Bucket=bucket,
        Key=f"training/metadata_v{timestamp}.json",
        Body=meta_json,
        ContentType='application/json'
    )
    meta_s3_path = f"s3://{bucket}/training/metadata_v{timestamp}.json"
    
    logger.info(f"Saved training data to: {data_s3_path}")
    logger.info(f"Saved metadata to: {meta_s3_path}")
    
    return {
        'data_s3_path': data_s3_path,
        'metadata_s3_path': meta_s3_path,
        'version': timestamp
    }


def run(**context):
    """
    Main save task - validates data and creates final training-ready dataset.
    
    Args:
        **context: Bruin context - receives input from transform task
    
    Returns:
        dict: Result with final data path and metadata
    """
    logger.info("="*50)
    logger.info("TASK: SAVE - Finalizing training data")
    logger.info("="*50)
    
    config = load_config()
    
    try:
        # Get source path from context
        source_path = context.get('transform', {}).get('output_path')
        
        # Load transformed data
        df = load_data(config, source_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Validate data
        validate_data(df)
        
        # Create metadata
        metadata = create_metadata(df, config)
        
        # Save final dataset
        result = save_final(df, config, metadata)
        
        result['records'] = len(df)
        result['features'] = len(df.columns)
        
        logger.info(f"Save completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Save failed: {e}")
        raise


if __name__ == "__main__":
    result = run()
    print(f"✓ Save complete: {result}")
