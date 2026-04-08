#!/usr/bin/env python3
"""
Ingest task - Downloads raw Titanic CSV from source to S3 raw bucket.
"""

import os
import yaml
import boto3
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config" / "s3_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_s3_client(config):
    """Initialize S3 client using config settings."""
    return boto3.client(
        's3',
        region_name=config['aws']['region'],
    )


def get_bucket_name(config, bucket_type):
    """Get S3 bucket name with environment substitution."""
    bucket = config['buckets'][bucket_type]
    env = os.getenv('ENV', 'dev')
    return bucket.format(env=env)


def download_from_source(config):
    """Download Titanic dataset from source URL."""
    source_url = config['pipeline']['source_url']
    logger.info(f"Downloading data from {source_url}")
    
    df = pd.read_csv(source_url)
    logger.info(f"Downloaded {len(df)} records")
    
    return df


def save_locally(df, config, output_filename):
    """Save DataFrame to local directory."""
    local_config = config['local']
    raw_dir = Path(local_config['raw_dir'])
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = raw_dir / output_filename
    df.to_csv(output_path, index=False, encoding=config['pipeline']['encoding'])
    logger.info(f"Saved to local: {output_path}")
    
    return str(output_path)


def upload_to_s3(df, config, output_filename):
    """Upload DataFrame to S3 raw bucket."""
    s3_client = get_s3_client(config)
    bucket_name = get_bucket_name(config, 'raw')
    
    # Upload as CSV
    csv_data = df.to_csv(index=False, encoding=config['pipeline']['encoding'])
    s3_key = f"raw/{output_filename}"
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=csv_data,
        ContentType='text/csv'
    )
    
    logger.info(f"Uploaded to s3://{bucket_name}/{s3_key}")
    return f"s3://{bucket_name}/{s3_key}"


def run(**context):
    """
    Main ingest task - downloads data from source and saves to S3.
    
    Args:
        **context: Bruin context (optional, for cloud execution)
    
    Returns:
        dict: Result with file path and record count
    """
    logger.info("="*50)
    logger.info("TASK: INGEST - Starting data ingestion")
    logger.info("="*50)
    
    config = load_config()
    local_enabled = config['local']['enabled']
    
    try:
        # Download from source
        df = download_from_source(config)
        
        # Get output filename with timestamp
        output_filename = config['pipeline']['raw_file']
        
        # Save locally
        if local_enabled:
            local_path = save_locally(df, config, output_filename)
            result = {'local_path': local_path, 'records': len(df)}
        else:
            result = {'records': len(df)}
        
        # Upload to S3 (if not in local-only mode)
        if not local_enabled or os.getenv('UPLOAD_S3', 'false').lower() == 'true':
            s3_path = upload_to_s3(df, config, output_filename)
            result['s3_path'] = s3_path
        
        logger.info(f"Ingest completed: {result['records']} records")
        return result
        
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise


if __name__ == "__main__":
    result = run()
    print(f"✓ Ingest complete: {result}")
