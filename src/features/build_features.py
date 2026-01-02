"""
Feature building functions for the customer churn prediction project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.helpers import load_config, save_pickle

logger = setup_logger(__name__)


def encode_categorical_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: List[str],
    method: str = 'onehot',
    save_encoder: bool = True,
    encoder_path: str = 'models/production/encoder.pkl'
) -> tuple:
    """
    Encode categorical features.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        categorical_cols: List of categorical columns
        method: Encoding method ('onehot', 'label')
        save_encoder: Whether to save encoder
        encoder_path: Path to save encoder

    Returns:
        Tuple of encoded (train_df, val_df, test_df)
    """
    logger.info(f"Encoding categorical features using method: {method}")

    train_encoded = train_df.copy()
    val_encoded = val_df.copy()
    test_encoded = test_df.copy()

    # Filter columns that exist in dataframe
    cols_to_encode = [col for col in categorical_cols if col in train_df.columns]

    if method == 'onehot':
        # Use pandas get_dummies for simplicity
        train_encoded = pd.get_dummies(
            train_encoded,
            columns=cols_to_encode,
            drop_first=True,
            prefix=cols_to_encode
        )
        val_encoded = pd.get_dummies(
            val_encoded,
            columns=cols_to_encode,
            drop_first=True,
            prefix=cols_to_encode
        )
        test_encoded = pd.get_dummies(
            test_encoded,
            columns=cols_to_encode,
            drop_first=True,
            prefix=cols_to_encode
        )

        # Align columns across all sets
        all_columns = train_encoded.columns.tolist()

        # Add missing columns to val and test
        for col in all_columns:
            if col not in val_encoded.columns:
                val_encoded[col] = 0
            if col not in test_encoded.columns:
                test_encoded[col] = 0

        # Reorder columns to match train
        val_encoded = val_encoded[all_columns]
        test_encoded = test_encoded[all_columns]

        logger.info(f"One-hot encoded {len(cols_to_encode)} categorical features")
        logger.info(f"New shape: {train_encoded.shape}")

    elif method == 'label':
        encoders = {}
        for col in cols_to_encode:
            le = LabelEncoder()
            train_encoded[col] = le.fit_transform(train_df[col].astype(str))
            val_encoded[col] = le.transform(val_df[col].astype(str))
            test_encoded[col] = le.transform(test_df[col].astype(str))
            encoders[col] = le

        if save_encoder:
            save_pickle(encoders, encoder_path)
            logger.info(f"Label encoders saved to: {encoder_path}")

        logger.info(f"Label encoded {len(cols_to_encode)} categorical features")

    return train_encoded, val_encoded, test_encoded


def select_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'Churn',
    exclude_cols: Optional[List[str]] = None
) -> tuple:
    """
    Select features for modeling.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        target_col: Name of target column
        exclude_cols: Columns to exclude from features

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Selecting features for modeling...")

    # Default exclude columns
    if exclude_cols is None:
        exclude_cols = ['customerID']

    # Add target to exclude list
    if target_col not in exclude_cols:
        exclude_cols.append(target_col)

    # Get feature columns
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    # Split features and target
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df[target_col]
    y_val = val_df[target_col]
    y_test = test_df[target_col]

    logger.info(f"Selected {len(feature_cols)} features")
    logger.info(f"Feature columns: {feature_cols[:10]}...")  # Show first 10
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_feature_names(
    feature_names: List[str],
    filepath: str = 'models/production/feature_names.pkl'
) -> None:
    """
    Save feature names for later use.

    Args:
        feature_names: List of feature names
        filepath: Path to save feature names
    """
    save_pickle(feature_names, filepath)
    logger.info(f"Feature names saved to: {filepath}")


def build_features_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config_path: str = 'config.yaml'
) -> tuple:
    """
    Run complete feature building pipeline.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        config_path: Path to configuration file

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Running feature building pipeline...")

    # Load configuration
    config = load_config(config_path)

    # Encode categorical features
    categorical_cols = config['features']['categorical_features']
    encoding_method = config['features']['encoding']['method']

    train_encoded, val_encoded, test_encoded = encode_categorical_features(
        train_df, val_df, test_df,
        categorical_cols=categorical_cols,
        method=encoding_method
    )

    # Select features
    target_col = config['features']['target']
    X_train, X_val, X_test, y_train, y_val, y_test = select_features(
        train_encoded, val_encoded, test_encoded,
        target_col=target_col
    )

    # Save feature names
    save_feature_names(X_train.columns.tolist())

    logger.info("Feature building pipeline completed")

    return X_train, X_val, X_test, y_train, y_val, y_test
