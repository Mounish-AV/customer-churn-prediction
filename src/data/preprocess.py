"""
Data preprocessing functions for the customer churn prediction project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.helpers import load_config, save_pickle

logger = setup_logger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data.

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data...")

    df_clean = df.copy()

    # Remove leading/trailing whitespaces from string columns
    string_cols = df_clean.select_dtypes(include=['object']).columns
    for col in string_cols:
        df_clean[col] = df_clean[col].str.strip()

    # Convert TotalCharges to numeric (it might have spaces for new customers)
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(
            df_clean['TotalCharges'],
            errors='coerce'
        )

    # Convert SeniorCitizen to string for consistency
    if 'SeniorCitizen' in df_clean.columns:
        df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

    # Convert target variable to binary
    if 'Churn' in df_clean.columns:
        df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})

    logger.info(f"Data cleaned. Shape: {df_clean.shape}")

    return df_clean


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'median',
    numerical_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        df: DataFrame with missing values
        strategy: Strategy for imputation ('mean', 'median', 'mode', 'drop')
        numerical_cols: List of numerical columns

    Returns:
        DataFrame with missing values handled
    """
    logger.info(f"Handling missing values using strategy: {strategy}")

    df_imputed = df.copy()

    # Check for missing values
    missing_before = df_imputed.isnull().sum().sum()
    logger.info(f"Missing values before imputation: {missing_before}")

    if strategy == 'drop':
        df_imputed = df_imputed.dropna()
    else:
        # Handle numerical columns
        if numerical_cols:
            for col in numerical_cols:
                if col in df_imputed.columns and df_imputed[col].isnull().any():
                    if strategy == 'mean':
                        fill_value = df_imputed[col].mean()
                    elif strategy == 'median':
                        fill_value = df_imputed[col].median()
                    else:
                        fill_value = df_imputed[col].mode()[0]

                    df_imputed.loc[:, col] = df_imputed[col].fillna(fill_value)
                    logger.info(f"Filled {col} with {strategy}: {fill_value}")

        # Handle categorical columns with mode
        categorical_cols = df_imputed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_imputed[col].isnull().any():
                mode_value = df_imputed[col].mode()[0]
                df_imputed[col].fillna(mode_value, inplace=True)
                logger.info(f"Filled {col} with mode: {mode_value}")

    missing_after = df_imputed.isnull().sum().sum()
    logger.info(f"Missing values after imputation: {missing_after}")

    return df_imputed


def remove_outliers(
    df: pd.DataFrame,
    numerical_cols: List[str],
    method: str = 'iqr',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Remove outliers from numerical columns.

    Args:
        df: DataFrame
        numerical_cols: List of numerical columns
        method: Method for outlier detection ('iqr', 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        DataFrame with outliers removed
    """
    logger.info(f"Removing outliers using method: {method}")

    df_clean = df.copy()
    initial_shape = df_clean.shape

    for col in numerical_cols:
        if col not in df_clean.columns:
            continue

        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            df_clean = df_clean[
                (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            ]

        elif method == 'zscore':
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            outliers = (z_scores > threshold).sum()
            df_clean = df_clean[z_scores <= threshold]

        if outliers > 0:
            logger.info(f"Removed {outliers} outliers from {col}")

    logger.info(f"Shape before: {initial_shape}, after: {df_clean.shape}")

    return df_clean


def split_data(
    df: pd.DataFrame,
    target_col: str = 'Churn',
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.

    Args:
        df: DataFrame to split
        target_col: Name of target column
        test_size: Proportion of test set
        val_size: Proportion of validation set
        random_state: Random seed
        stratify: Whether to stratify split by target

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Splitting data into train, validation, and test sets...")

    # First split: separate test set
    stratify_col = df[target_col] if stratify else None

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )

    # Second split: separate validation from train
    val_size_adjusted = val_size / (1 - test_size)
    stratify_col_train = train_val_df[target_col] if stratify else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_col_train
    )

    logger.info(f"Train set: {train_df.shape}")
    logger.info(f"Validation set: {val_df.shape}")
    logger.info(f"Test set: {test_df.shape}")

    # Log target distribution
    if target_col in train_df.columns:
        logger.info(f"Train target distribution:\n{train_df[target_col].value_counts()}")
        logger.info(f"Val target distribution:\n{val_df[target_col].value_counts()}")
        logger.info(f"Test target distribution:\n{test_df[target_col].value_counts()}")

    return train_df, val_df, test_df


def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numerical_cols: List[str],
    method: str = 'standard',
    save_scaler: bool = True,
    scaler_path: str = 'models/production/scaler.pkl'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scale numerical features.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        numerical_cols: List of numerical columns to scale
        method: Scaling method ('standard', 'minmax', 'robust')
        save_scaler: Whether to save the scaler
        scaler_path: Path to save scaler

    Returns:
        Tuple of scaled (train_df, val_df, test_df)
    """
    logger.info(f"Scaling features using method: {method}")

    # Select scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    # Create copies
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    # Fit on train and transform all sets
    cols_to_scale = [col for col in numerical_cols if col in train_df.columns]

    train_scaled[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    val_scaled[cols_to_scale] = scaler.transform(val_df[cols_to_scale])
    test_scaled[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

    logger.info(f"Scaled {len(cols_to_scale)} numerical features")

    # Save scaler
    if save_scaler:
        save_pickle(scaler, scaler_path)
        logger.info(f"Scaler saved to: {scaler_path}")

    return train_scaled, val_scaled, test_scaled


def preprocess_pipeline(
    df: pd.DataFrame,
    config_path: str = 'config.yaml'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run complete preprocessing pipeline.

    Args:
        df: Raw DataFrame
        config_path: Path to configuration file

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Running preprocessing pipeline...")

    # Load configuration
    config = load_config(config_path)

    # Clean data
    df_clean = clean_data(df)

    # Handle missing values
    numerical_cols = config['features']['numerical_features']
    df_imputed = handle_missing_values(
        df_clean,
        strategy=config['preprocessing']['handle_missing']['strategy'],
        numerical_cols=numerical_cols
    )

    # Remove outliers (optional, can be disabled)
    if config['preprocessing'].get('outlier_detection', {}).get('method'):
        df_no_outliers = remove_outliers(
            df_imputed,
            numerical_cols=numerical_cols,
            method=config['preprocessing']['outlier_detection']['method'],
            threshold=config['preprocessing']['outlier_detection']['threshold']
        )
    else:
        df_no_outliers = df_imputed

    # Split data
    train_df, val_df, test_df = split_data(
        df_no_outliers,
        target_col=config['features']['target'],
        test_size=config['split']['test_size'],
        val_size=config['split']['validation_size'],
        random_state=config['split']['random_state'],
        stratify=config['split']['stratify']
    )

    logger.info("Preprocessing pipeline completed")

    return train_df, val_df, test_df