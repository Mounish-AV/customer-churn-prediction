"""
Feature engineering functions for the customer churn prediction project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_tenure_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create tenure groups from tenure months.

    Args:
        df: DataFrame with tenure column

    Returns:
        DataFrame with tenure_group column
    """
    logger.info("Creating tenure groups...")

    df_new = df.copy()

    if 'tenure' in df_new.columns:
        df_new['tenure_group'] = pd.cut(
            df_new['tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['0-1 year', '1-2 years', '2-4 years', '4+ years']
        )
        logger.info("Tenure groups created")

    return df_new


def create_service_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on services subscribed.

    Args:
        df: DataFrame with service columns

    Returns:
        DataFrame with new service features
    """
    logger.info("Creating service features...")

    df_new = df.copy()

    # Total services subscribed
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    available_service_cols = [col for col in service_cols if col in df_new.columns]

    if available_service_cols:
        # Count number of services (Yes values)
        df_new['total_services'] = 0
        for col in available_service_cols:
            df_new['total_services'] += (df_new[col] == 'Yes').astype(int)

        logger.info(f"Created total_services feature from {len(available_service_cols)} columns")

    # Has internet service
    if 'InternetService' in df_new.columns:
        df_new['has_internet'] = (df_new['InternetService'] != 'No').astype(int)

    # Has phone service
    if 'PhoneService' in df_new.columns:
        df_new['has_phone'] = (df_new['PhoneService'] == 'Yes').astype(int)

    return df_new


def create_contract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on contract information.

    Args:
        df: DataFrame with contract columns

    Returns:
        DataFrame with new contract features
    """
    logger.info("Creating contract features...")

    df_new = df.copy()

    # Is on month-to-month contract
    if 'Contract' in df_new.columns:
        df_new['is_month_to_month'] = (df_new['Contract'] == 'Month-to-month').astype(int)

    # Has paperless billing
    if 'PaperlessBilling' in df_new.columns:
        df_new['has_paperless'] = (df_new['PaperlessBilling'] == 'Yes').astype(int)

    return df_new


def create_charge_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on charges.

    Args:
        df: DataFrame with charge columns

    Returns:
        DataFrame with new charge features
    """
    logger.info("Creating charge features...")

    df_new = df.copy()

    # Average monthly charges
    if 'TotalCharges' in df_new.columns and 'tenure' in df_new.columns:
        # Avoid division by zero
        df_new['avg_monthly_charges'] = df_new.apply(
            lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'],
            axis=1
        )

    # Charge ratio (monthly vs total)
    if 'MonthlyCharges' in df_new.columns and 'TotalCharges' in df_new.columns:
        df_new['charge_ratio'] = df_new.apply(
            lambda row: row['MonthlyCharges'] / row['TotalCharges'] if row['TotalCharges'] > 0 else 0,
            axis=1
        )

    return df_new


def create_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on demographics.

    Args:
        df: DataFrame with demographic columns

    Returns:
        DataFrame with new demographic features
    """
    logger.info("Creating demographic features...")

    df_new = df.copy()

    # Family status
    if 'Partner' in df_new.columns and 'Dependents' in df_new.columns:
        df_new['has_family'] = (
            ((df_new['Partner'] == 'Yes') | (df_new['Dependents'] == 'Yes'))
        ).astype(int)

    # Senior with family
    if 'SeniorCitizen' in df_new.columns and 'has_family' in df_new.columns:
        df_new['senior_with_family'] = (
            (df_new['SeniorCitizen'] == 'Yes') & (df_new['has_family'] == 1)
        ).astype(int)

    return df_new


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    Args:
        df: DataFrame to engineer features for

    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering features...")

    df_engineered = df.copy()

    # Apply all feature engineering functions
    df_engineered = create_tenure_groups(df_engineered)
    df_engineered = create_service_features(df_engineered)
    df_engineered = create_contract_features(df_engineered)
    df_engineered = create_charge_features(df_engineered)
    df_engineered = create_demographic_features(df_engineered)

    logger.info(f"Feature engineering completed. New shape: {df_engineered.shape}")

    return df_engineered
