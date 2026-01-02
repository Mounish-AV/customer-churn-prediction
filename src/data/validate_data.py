"""
Data validation functions for the customer churn prediction project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def validate_schema(
    df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
    required_columns: Optional[List[str]] = None
) -> bool:
    """
    Validate DataFrame schema.

    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names
        required_columns: List of required column names

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating data schema...")

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    # Check expected columns
    if expected_columns:
        extra_cols = set(df.columns) - set(expected_columns)
        if extra_cols:
            logger.warning(f"Extra columns found: {extra_cols}")

    logger.info("Schema validation passed")
    return True


def validate_data_types(
    df: pd.DataFrame,
    expected_types: Optional[Dict[str, str]] = None
) -> bool:
    """
    Validate data types of columns.

    Args:
        df: DataFrame to validate
        expected_types: Dictionary mapping column names to expected types

    Returns:
        True if validation passes
    """
    logger.info("Validating data types...")

    if expected_types:
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    logger.warning(
                        f"Column '{col}' has type '{actual_type}', "
                        f"expected '{expected_type}'"
                    )

    logger.info("Data type validation completed")
    return True


def check_missing_values(
    df: pd.DataFrame,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Check for missing values in DataFrame.

    Args:
        df: DataFrame to check
        threshold: Maximum allowed missing value percentage (0-1)

    Returns:
        Dictionary with missing value percentages
    """
    logger.info("Checking for missing values...")

    missing_pct = df.isnull().sum() / len(df)
    missing_dict = missing_pct[missing_pct > 0].to_dict()

    if missing_dict:
        logger.warning(f"Columns with missing values: {missing_dict}")

        high_missing = {k: v for k, v in missing_dict.items() if v > threshold}
        if high_missing:
            logger.warning(
                f"Columns with >{threshold*100}% missing values: {high_missing}"
            )
    else:
        logger.info("No missing values found")

    return missing_dict


def check_duplicates(df: pd.DataFrame) -> int:
    """
    Check for duplicate rows in DataFrame.

    Args:
        df: DataFrame to check

    Returns:
        Number of duplicate rows
    """
    logger.info("Checking for duplicate rows...")

    n_duplicates = df.duplicated().sum()

    if n_duplicates > 0:
        logger.warning(f"Found {n_duplicates} duplicate rows")
    else:
        logger.info("No duplicate rows found")

    return n_duplicates


def validate_target_variable(
    df: pd.DataFrame,
    target_col: str = "Churn"
) -> bool:
    """
    Validate target variable.

    Args:
        df: DataFrame to validate
        target_col: Name of target column

    Returns:
        True if validation passes
    """
    logger.info(f"Validating target variable: {target_col}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # Check for missing values in target
    missing_target = df[target_col].isnull().sum()
    if missing_target > 0:
        logger.warning(f"Target variable has {missing_target} missing values")

    # Check target distribution
    target_dist = df[target_col].value_counts()
    logger.info(f"Target distribution:\n{target_dist}")

    # Check for class imbalance
    if len(target_dist) == 2:
        minority_pct = target_dist.min() / target_dist.sum()
        if minority_pct < 0.1:
            logger.warning(
                f"Severe class imbalance detected: "
                f"minority class = {minority_pct*100:.2f}%"
            )
        elif minority_pct < 0.3:
            logger.info(
                f"Class imbalance detected: "
                f"minority class = {minority_pct*100:.2f}%"
            )

    logger.info("Target variable validation passed")
    return True


def validate_numerical_ranges(
    df: pd.DataFrame,
    numerical_cols: List[str],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> bool:
    """
    Validate numerical column ranges.

    Args:
        df: DataFrame to validate
        numerical_cols: List of numerical column names
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        True if validation passes
    """
    logger.info("Validating numerical ranges...")

    for col in numerical_cols:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()

            logger.info(f"{col}: min={col_min}, max={col_max}")

            if min_val is not None and col_min < min_val:
                logger.warning(f"{col} has values below {min_val}")

            if max_val is not None and col_max > max_val:
                logger.warning(f"{col} has values above {max_val}")

    logger.info("Numerical range validation completed")
    return True


def run_full_validation(
    df: pd.DataFrame,
    config: Optional[Dict] = None
) -> Dict[str, any]:
    """
    Run full data validation suite.

    Args:
        df: DataFrame to validate
        config: Configuration dictionary

    Returns:
        Dictionary with validation results
    """
    logger.info("Running full data validation...")

    results = {
        'shape': df.shape,
        'missing_values': check_missing_values(df),
        'duplicates': check_duplicates(df),
        'validation_passed': True
    }

    # Validate target if config provided
    if config and 'features' in config:
        target_col = config['features'].get('target', 'Churn')
        try:
            validate_target_variable(df, target_col)
        except Exception as e:
            logger.error(f"Target validation failed: {e}")
            results['validation_passed'] = False

    logger.info("Full validation completed")
    return results
