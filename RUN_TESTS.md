# Step-by-Step Testing Guide

## 🚀 Quick Test Commands

### Step 1: Verify Python and Dependencies
```bash
# Check Python version (should be 3.8+)
python --version

# Check if key packages are installed
python -c "import pandas, numpy, sklearn, xgboost, yaml; print('✅ All dependencies installed!')"
```

### Step 2: Verify Data is Available
```bash
# Check if data file exists
ls -lh data/raw/telco_customer_churn.csv

# Check number of rows
wc -l data/raw/telco_customer_churn.csv
```

### Step 3: Run Basic Test (Fast - ~5 seconds)
```bash
# This tests data loading, validation, and preprocessing
python test_basic.py
```

### Step 4: Run Full Pipeline Test (Complete - ~30 seconds)
```bash
# This runs the entire ML pipeline and shows accuracy
python test_pipeline.py
```

### Step 5: Check Results
```bash
# View the final results
tail -20 test_pipeline.py

# Or run and see only the final summary
python test_pipeline.py 2>&1 | tail -30
```

## 📊 What to Expect

### Basic Test Output:
```
✓ Configuration loaded
✓ Raw data loaded: (7043, 21)
✓ Validation passed
✓ Data cleaned
✓ Missing values handled
✓ Data split successfully
All basic tests passed successfully!
```

### Full Pipeline Output (Final Results):
```
============================================================
Pipeline completed successfully!
============================================================

Best Model: logistic_regression
Test ROC-AUC: 0.8574      ← Main accuracy metric (85.74%)
Test Accuracy: 0.8136     ← Overall accuracy (81.36%)
Test Precision: 0.6844    ← Precision (68.44%)
Test Recall: 0.5500       ← Recall (55.00%)
Test F1-Score: 0.6099     ← F1 Score (60.99%)
```

## 🎯 Understanding the Metrics

- **ROC-AUC (0.8574)**: Overall model quality - **85.74% is EXCELLENT!**
- **Accuracy (0.8136)**: 81.36% of predictions are correct
- **Precision (0.6844)**: 68.44% of predicted churners actually churn
- **Recall (0.5500)**: Model catches 55% of actual churners
- **F1-Score (0.6099)**: Balance between precision and recall

## 🔍 Check Generated Files

### View trained models:
```bash
ls -lh models/production/
```

### View visualizations:
```bash
ls -lh reports/figures/
```

### Open a visualization (if you have image viewer):
```bash
# On Linux
xdg-open reports/figures/roc_curve.png

# Or just list them
ls reports/figures/*.png
```

## 🐛 Troubleshooting

### If you get "Module not found":
```bash
pip install -r requirements.txt
```

### If data file is missing:
```bash
python download_data.py
```

### If you want to see detailed logs:
```bash
# Run with full output
python test_pipeline.py
```

## ✅ Success Criteria

You should see:
- ✅ All tests pass without errors
- ✅ ROC-AUC > 0.80 (we got 0.8574!)
- ✅ Model saved to `models/production/best_model.pkl`
- ✅ Visualizations in `reports/figures/`

## 🎉 Quick One-Liner Test

```bash
# Run everything and show final results
python test_pipeline.py 2>&1 | grep -A 10 "Pipeline completed successfully"
```

