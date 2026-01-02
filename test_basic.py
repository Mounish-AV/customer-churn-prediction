"""
Basic test script to verify core functionality without xgboost.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils.logger import setup_logger
from src.utils.helpers import load_config
from src.data.load_data import load_raw_data
from src.data.validate_data import run_full_validation
from src.data.preprocess import clean_data, handle_missing_values, split_data

logger = setup_logger(__name__, level="INFO")


def test_basic_pipeline():
    """Test basic data loading and preprocessing."""
    try:
        logger.info("=" * 60)
        logger.info("Testing Basic Pipeline Components")
        logger.info("=" * 60)
        
        # Test 1: Load configuration
        logger.info("\n[Test 1] Loading configuration...")
        config = load_config()
        logger.info(f"✓ Configuration loaded: {config['project']['name']}")
        
        # Test 2: Load raw data
        logger.info("\n[Test 2] Loading raw data...")
        df_raw = load_raw_data()
        logger.info(f"✓ Raw data loaded: {df_raw.shape}")
        logger.info(f"  Columns: {list(df_raw.columns)}")
        
        # Test 3: Validate data
        logger.info("\n[Test 3] Validating data...")
        validation_results = run_full_validation(df_raw, config)
        logger.info(f"✓ Validation passed: {validation_results['validation_passed']}")
        
        # Test 4: Clean data
        logger.info("\n[Test 4] Cleaning data...")
        df_clean = clean_data(df_raw)
        logger.info(f"✓ Data cleaned: {df_clean.shape}")
        
        # Test 5: Handle missing values
        logger.info("\n[Test 5] Handling missing values...")
        numerical_cols = config['features']['numerical_features']
        df_imputed = handle_missing_values(df_clean, numerical_cols=numerical_cols)
        logger.info(f"✓ Missing values handled: {df_imputed.shape}")
        logger.info(f"  Missing values remaining: {df_imputed.isnull().sum().sum()}")
        
        # Test 6: Split data
        logger.info("\n[Test 6] Splitting data...")
        train_df, val_df, test_df = split_data(df_imputed)
        logger.info(f"✓ Data split successfully:")
        logger.info(f"  Train: {train_df.shape}")
        logger.info(f"  Val: {val_df.shape}")
        logger.info(f"  Test: {test_df.shape}")
        
        # Test 7: Check target distribution
        logger.info("\n[Test 7] Checking target distribution...")
        target_col = config['features']['target']
        logger.info(f"  Train target distribution:")
        logger.info(f"{train_df[target_col].value_counts()}")
        
        logger.info("\n" + "=" * 60)
        logger.info("All basic tests passed successfully!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_basic_pipeline()
    sys.exit(0 if success else 1)

