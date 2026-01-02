"""
Test script to verify the entire ML pipeline works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils.logger import setup_logger
from src.utils.helpers import load_config
from src.data.load_data import load_raw_data, save_processed_data
from src.data.validate_data import run_full_validation
from src.data.preprocess import preprocess_pipeline, scale_features
from src.features.feature_engineering import engineer_features
from src.features.build_features import build_features_pipeline
from src.models.train_model import train_baseline_model, train_all_models, save_production_model
from src.models.evaluate_model import evaluate_model, compare_models, select_best_model
from src.visualization.visualize import (
    plot_confusion_matrix, plot_roc_curve, plot_feature_importance,
    plot_correlation_matrix
)

logger = setup_logger(__name__, level="INFO")


def main():
    """Run the complete ML pipeline."""
    try:
        logger.info("=" * 60)
        logger.info("Starting Customer Churn Prediction Pipeline")
        logger.info("=" * 60)
        
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Step 1: Load raw data
        logger.info("\n[Step 1] Loading raw data...")
        df_raw = load_raw_data()
        logger.info(f"Raw data shape: {df_raw.shape}")
        
        # Step 2: Validate data
        logger.info("\n[Step 2] Validating data...")
        validation_results = run_full_validation(df_raw, config)
        logger.info(f"Validation passed: {validation_results['validation_passed']}")
        
        # Step 3: Preprocess data
        logger.info("\n[Step 3] Preprocessing data...")
        train_df, val_df, test_df = preprocess_pipeline(df_raw)
        
        # Save processed data
        save_processed_data(train_df, "train")
        save_processed_data(val_df, "validation")
        save_processed_data(test_df, "test")
        logger.info("Processed data saved")
        
        # Step 4: Feature engineering
        logger.info("\n[Step 4] Engineering features...")
        train_df = engineer_features(train_df)
        val_df = engineer_features(val_df)
        test_df = engineer_features(test_df)
        
        # Step 5: Scale features
        logger.info("\n[Step 5] Scaling features...")
        numerical_cols = config['features']['numerical_features']
        train_df, val_df, test_df = scale_features(
            train_df, val_df, test_df,
            numerical_cols=numerical_cols,
            method=config['preprocessing']['scaling']['method']
        )
        
        # Step 6: Build features
        logger.info("\n[Step 6] Building features...")
        X_train, X_val, X_test, y_train, y_val, y_test = build_features_pipeline(
            train_df, val_df, test_df
        )
        
        logger.info(f"Feature shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Step 7: Train baseline model
        logger.info("\n[Step 7] Training baseline model...")
        baseline_model = train_baseline_model(X_train, y_train)
        
        # Evaluate baseline
        baseline_metrics = evaluate_model(baseline_model, X_test, y_test)
        logger.info(f"Baseline model ROC-AUC: {baseline_metrics['roc_auc']:.4f}")
        
        # Step 8: Train all models
        logger.info("\n[Step 8] Training all models...")
        models = train_all_models(X_train, y_train)
        logger.info(f"Trained {len(models)} models")
        
        # Step 9: Compare models
        logger.info("\n[Step 9] Comparing models...")
        comparison_df = compare_models(models, X_val, y_val)
        logger.info(f"\nModel Comparison:\n{comparison_df}")
        
        # Step 10: Select best model
        logger.info("\n[Step 10] Selecting best model...")
        best_model_name, best_model = select_best_model(
            models, X_val, y_val,
            metric=config['selection']['primary_metric']
        )
        
        # Step 11: Evaluate best model on test set
        logger.info("\n[Step 11] Evaluating best model on test set...")
        test_metrics = evaluate_model(best_model, X_test, y_test)
        
        # Step 12: Save production model
        logger.info("\n[Step 12] Saving production model...")
        save_production_model(best_model, 'best_model')
        
        # Step 13: Generate visualizations
        logger.info("\n[Step 13] Generating visualizations...")
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        plot_confusion_matrix(y_test, y_pred, 'reports/figures/confusion_matrix.png')
        plot_roc_curve(y_test, y_proba, 'reports/figures/roc_curve.png')
        
        if hasattr(best_model, 'feature_importances_'):
            plot_feature_importance(
                best_model, X_train.columns.tolist(),
                save_path='reports/figures/feature_importance.png'
            )
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        logger.info(f"\nBest Model: {best_model_name}")
        logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
        logger.info(f"Test F1-Score: {test_metrics['f1']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

