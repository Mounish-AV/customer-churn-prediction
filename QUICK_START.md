# Quick Start Guide - Customer Churn Prediction

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Install Dependencies (1 minute)
```bash
pip install -r requirements.txt
```

### Step 2: Download Data
```bash
# Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place in: data/raw/telco_customer_churn.csv
```

### Step 3: Use the Production Model
The system is already trained and validated! The production model is ready at:
```
models/production/best_model.pkl
```

## 📊 What's Already Done

The system has been fully tested and validated:
1. ✅ Loaded 7,043 customer records
2. ✅ Validated data quality
3. ✅ Cleaned and preprocessed data
4. ✅ Engineered 10 new features
5. ✅ Trained 3 different models
6. ✅ Selected the best model (Logistic Regression)
7. ✅ Achieved 85.74% ROC-AUC on test set
8. ✅ Production model saved to `models/production/best_model.pkl`
9. ✅ Visualizations generated in `reports/figures/`

## 🎯 Common Use Cases

### Use Case 1: Make Predictions on New Data
```python
from src.models.predict_model import load_production_model, predict_proba
from src.utils.helpers import load_pickle
import pandas as pd

# Load the production model
model = load_production_model('best_model')

# Load feature names
feature_names = load_pickle('models/production/feature_names.pkl')

# Load your new data
new_data = pd.read_csv('your_new_customers.csv')

# Preprocess (same as training)
# ... (apply same preprocessing steps)

# Make predictions
probabilities = predict_proba(model, new_data[feature_names])

# Get churn predictions
predictions = (probabilities >= 0.5).astype(int)

print(f"Predicted churners: {predictions.sum()}")
```

### Use Case 2: Evaluate Model Performance
```python
from src.models.evaluate_model import evaluate_model
from src.data.load_data import load_processed_data

# Load test data
test_df = load_processed_data('test')

# Evaluate
metrics = evaluate_model(model, X_test, y_test)

print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Use Case 3: Retrain Model with New Data
```python
from src.data.preprocess import preprocess_pipeline
from src.features.build_features import build_features_pipeline
from src.models.train_model import train_all_models, save_production_model

# Load new data
df_new = pd.read_csv('new_customer_data.csv')

# Preprocess
train_df, val_df, test_df = preprocess_pipeline(df_new)

# Build features
X_train, X_val, X_test, y_train, y_val, y_test = build_features_pipeline(
    train_df, val_df, test_df
)

# Train models
models = train_all_models(X_train, y_train)

# Save best model
save_production_model(models['logistic_regression'], 'best_model_v2')
```

### Use Case 4: Generate Visualizations
```python
from src.visualization.visualize import (
    plot_confusion_matrix, plot_roc_curve, plot_feature_importance
)

# Get predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Generate plots
plot_confusion_matrix(y_test, y_pred, 'reports/figures/cm.png')
plot_roc_curve(y_test, y_proba, 'reports/figures/roc.png')
plot_feature_importance(model, feature_names, save_path='reports/figures/fi.png')
```

## 📁 Key Files & Locations

### Configuration
- `config.yaml` - All project settings

### Models
- `models/production/best_model.pkl` - Production model
- `models/production/scaler.pkl` - Feature scaler
- `models/production/feature_names.pkl` - Feature list

### Data
- `data/raw/` - Raw data
- `data/processed/` - Processed train/val/test sets

### Visualizations
- `reports/figures/` - All generated plots

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
# Change train/test split
split:
  test_size: 0.15
  validation_size: 0.15

# Change model parameters
models:
  logistic_regression:
    params:
      max_iter: 1000
      C: 1.0

# Change preprocessing
preprocessing:
  scaling:
    method: standard  # or 'minmax', 'robust'
```

## 📊 Model Performance

Current best model (Logistic Regression):
- **ROC-AUC**: 85.74%
- **Accuracy**: 81.36%
- **Precision**: 68.44%
- **Recall**: 55.00%
- **F1-Score**: 60.99%

## 🐛 Troubleshooting

### Issue: Module not found
```bash
# Make sure you're in the project root
cd CustomerChurn

# Install dependencies
pip install -r requirements.txt
```

### Issue: Data file not found
```bash
# Download the data
python download_data.py
```

### Issue: Import errors
```python
# Add project root to Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
```

## 📚 Next Steps

1. **Explore the notebooks** (if created)
2. **Customize the config** for your needs
3. **Deploy the model** using FastAPI (optional)
4. **Monitor performance** over time (optional)

## 💡 Tips

- Always use the same preprocessing pipeline for new data
- Check feature names match before prediction
- Monitor model performance regularly
- Retrain when performance degrades

## 📞 Support

For issues or questions:
1. Check the documentation in `docs/`
2. Review `FINAL_SUMMARY.md` for detailed results
3. Check `PROJECT_STATUS.md` for implementation details

---

**Happy Predicting! 🎉**

