#!/usr/bin/env python3
"""
Unit tests for Titanic ETL Pipeline
These tests verify the code structure without requiring full dependencies.
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestETLConfig:
    """Test ETL configuration files"""
    
    def test_s3_config_exists(self):
        """Test that S3 config file exists"""
        config_path = PROJECT_ROOT / "etl" / "config" / "s3_config.yaml"
        assert config_path.exists(), "s3_config.yaml should exist"
    
    def test_env_example_exists(self):
        """Test that .env.example file exists"""
        env_path = PROJECT_ROOT / "etl" / "config" / ".env.example"
        assert env_path.exists(), ".env.example should exist"


class TestETLTaskFiles:
    """Test ETL task files exist and have correct structure"""
    
    def test_ingest_task_exists(self):
        """Test ingest.py exists"""
        path = PROJECT_ROOT / "etl" / "tasks" / "ingest.py"
        assert path.exists(), "ingest.py should exist"
    
    def test_clean_task_exists(self):
        """Test clean.py exists"""
        path = PROJECT_ROOT / "etl" / "tasks" / "clean.py"
        assert path.exists(), "clean.py should exist"
    
    def test_transform_task_exists(self):
        """Test transform.py exists"""
        path = PROJECT_ROOT / "etl" / "tasks" / "transform.py"
        assert path.exists(), "transform.py should exist"
    
    def test_save_task_exists(self):
        """Test save.py exists"""
        path = PROJECT_ROOT / "etl" / "tasks" / "save.py"
        assert path.exists(), "save.py should exist"


class TestETLDAG:
    """Test ETL DAG definition"""
    
    def test_dag_file_exists(self):
        """Test titanic_dag.py exists"""
        path = PROJECT_ROOT / "etl" / "dags" / "titanic_dag.py"
        assert path.exists(), "titanic_dag.py should exist"
    
    def test_dag_has_bruin_import(self):
        """Test DAG file has Bruin import"""
        path = PROJECT_ROOT / "etl" / "dags" / "titanic_dag.py"
        content = path.read_text()
        assert "from bruin import" in content or "import bruin" in content


class TestDataFiles:
    """Test data files exist"""
    
    def test_titanic_csv_exists(self):
        """Test titanic.csv data file exists"""
        path = PROJECT_ROOT / "data" / "titanic.csv"
        assert path.exists(), "titanic.csv should exist"
    
    def test_titanic_csv_not_empty(self):
        """Test titanic.csv is not empty"""
        path = PROJECT_ROOT / "data" / "titanic.csv"
        assert path.stat().st_size > 0, "titanic.csv should not be empty"


class TestPythonCode:
    """Test Python code syntax and structure"""
    
    def test_ingest_syntax(self):
        """Test ingest.py has valid Python syntax"""
        path = PROJECT_ROOT / "etl" / "tasks" / "ingest.py"
        with open(path, 'r') as f:
            code = f.read()
        try:
            compile(code, str(path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in ingest.py: {e}")
    
    def test_clean_syntax(self):
        """Test clean.py has valid Python syntax"""
        path = PROJECT_ROOT / "etl" / "tasks" / "clean.py"
        with open(path, 'r') as f:
            code = f.read()
        try:
            compile(code, str(path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in clean.py: {e}")
    
    def test_transform_syntax(self):
        """Test transform.py has valid Python syntax"""
        path = PROJECT_ROOT / "etl" / "tasks" / "transform.py"
        with open(path, 'r') as f:
            code = f.read()
        try:
            compile(code, str(path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in transform.py: {e}")
    
    def test_save_syntax(self):
        """Test save.py has valid Python syntax"""
        path = PROJECT_ROOT / "etl" / "tasks" / "save.py"
        with open(path, 'r') as f:
            code = f.read()
        try:
            compile(code, str(path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in save.py: {e}")


class TestMainAppCode:
    """Test main application code syntax"""
    
    def test_train_syntax(self):
        """Test train.py has valid Python syntax"""
        path = PROJECT_ROOT / "train.py"
        with open(path, 'r') as f:
            code = f.read()
        try:
            compile(code, str(path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in train.py: {e}")
    
    def test_predict_syntax(self):
        """Test predict.py has valid Python syntax"""
        path = PROJECT_ROOT / "predict.py"
        with open(path, 'r') as f:
            code = f.read()
        try:
            compile(code, str(path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in predict.py: {e}")
    
    def test_optimization_syntax(self):
        """Test lifeboat_optimization.py has valid Python syntax"""
        path = PROJECT_ROOT / "optimization" / "lifeboat_optimization.py"
        with open(path, 'r') as f:
            code = f.read()
        try:
            compile(code, str(path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in lifeboat_optimization.py: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])