"""
Data loading functions for the customer churn prediction project.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.helpers import load_config

logger = setup_logger(__name__)


def load_raw_data(
    data_path: Optional[str] = None,
    config_path: str = "config.yaml"
) -> pd.DataFrame:
    """
    Load raw customer churn data from CSV file.

    Args:
        data_path: Path to data file (optional, uses config if not provided)
        config_path: Path to configuration file

    Returns:
        DataFrame with raw data
    """
    try:
        if data_path is None:
            config = load_config(config_path)
            data_dir = config['data']['raw_dir']
            dataset_name = config['data']['dataset_name']
            data_path = Path(data_dir) / dataset_name

        logger.info(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)

        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        return df

    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def load_processed_data(
    data_type: str = "train",
    config_path: str = "config.yaml"
) -> pd.DataFrame:
    """
    Load processed data (train, test, or validation).

    Args:
        data_type: Type of data to load ('train', 'test', 'validation')
        config_path: Path to configuration file

    Returns:
        DataFrame with processed data
    """
    try:
        config = load_config(config_path)
        data_dir = config['data']['processed_dir']
        data_path = Path(data_dir) / f"{data_type}.csv"

        logger.info(f"Loading {data_type} data from: {data_path}")
        df = pd.read_csv(data_path)

        logger.info(f"{data_type.capitalize()} data loaded. Shape: {df.shape}")

        return df

    except FileNotFoundError:
        logger.error(f"Processed data file not found: {data_path}")
        logger.info("Please run data preparation pipeline first.")
        raise
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise


def save_processed_data(
    df: pd.DataFrame,
    data_type: str = "train",
    config_path: str = "config.yaml"
) -> None:
    """
    Save processed data to CSV file.

    Args:
        df: DataFrame to save
        data_type: Type of data ('train', 'test', 'validation')
        config_path: Path to configuration file
    """
    try:
        config = load_config(config_path)
        data_dir = Path(config['data']['processed_dir'])
        data_dir.mkdir(parents=True, exist_ok=True)

        data_path = data_dir / f"{data_type}.csv"

        logger.info(f"Saving {data_type} data to: {data_path}")
        df.to_csv(data_path, index=False)

        logger.info(f"{data_type.capitalize()} data saved successfully. Shape: {df.shape}")

    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with dataset information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }

    return info


def load_train_test_split(
    config_path: str = "config.yaml"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test datasets.

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    try:
        train_df = load_processed_data("train", config_path)
        val_df = load_processed_data("validation", config_path)
        test_df = load_processed_data("test", config_path)

        logger.info("All datasets loaded successfully")
        logger.info(f"Train: {train_df.shape}, Validation: {val_df.shape}, Test: {test_df.shape}")

        return train_df, val_df, test_df

    except Exception as e:
        logger.error(f"Error loading train/test split: {e}")
        raise
