"""
Helper utilities for the customer churn prediction project.
"""

import os
import yaml
import json
import pickle
import joblib
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import pandas as pd
import numpy as np


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_env_vars() -> None:
    """Load environment variables from .env file."""
    load_dotenv()


def get_env_var(key: str, default: Optional[str] = None) -> str:
    """
    Get environment variable value.

    Args:
        key: Environment variable key
        default: Default value if key not found

    Returns:
        Environment variable value
    """
    return os.getenv(key, default)


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to save
        filepath: Path to save file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_joblib(obj: Any, filepath: str) -> None:
    """
    Save object using joblib.

    Args:
        obj: Object to save
        filepath: Path to save file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)


def load_joblib(filepath: str) -> Any:
    """
    Load object using joblib.

    Args:
        filepath: Path to joblib file

    Returns:
        Loaded object
    """
    return joblib.load(filepath)


def save_json(data: Dict, filepath: str, indent: int = 2) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save file
        indent: JSON indentation
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(filepath: str) -> Dict:
    """
    Load dictionary from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def create_directory(path: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """
    Get project root directory.

    Returns:
        Project root path
    """
    return Path(__file__).parent.parent.parent


def convert_to_numeric(series: pd.Series) -> pd.Series:
    """
    Convert series to numeric, handling errors.

    Args:
        series: Pandas series

    Returns:
        Numeric series
    """
    return pd.to_numeric(series, errors='coerce')


def calculate_business_value(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    clv: float = 1000,
    retention_cost: float = 100,
    acquisition_cost: float = 500
) -> Dict[str, float]:
    """
    Calculate business value metrics.

    Args:
        tp: True positives
        tn: True negatives
        fp: False positives
        fn: False negatives
        clv: Customer lifetime value
        retention_cost: Cost to retain a customer
        acquisition_cost: Cost to acquire a new customer

    Returns:
        Dictionary with business metrics
    """
    # Saved customers (correctly identified churners)
    saved_value = tp * (clv - retention_cost)

    # Wasted retention efforts (false alarms)
    wasted_cost = fp * retention_cost

    # Lost customers (missed churners)
    lost_value = fn * acquisition_cost

    # Net value
    net_value = saved_value - wasted_cost - lost_value

    return {
        'saved_value': saved_value,
        'wasted_cost': wasted_cost,
        'lost_value': lost_value,
        'net_value': net_value,
        'roi': (saved_value / (wasted_cost + lost_value)) if (wasted_cost + lost_value) > 0 else 0
    }
