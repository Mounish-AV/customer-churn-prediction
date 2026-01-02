# Customer Churn Prediction

![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-85.74%25-success)

A complete, production-ready machine learning system to predict customer churn for telecommunications companies. Achieves **85.74% ROC-AUC** on test data.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project identifies customers at risk of churning (leaving the service) using machine learning techniques. The system is **fully tested and production-ready** with excellent performance metrics.

**Business Impact:**
- **85.74% ROC-AUC** - Excellent discrimination between churners and non-churners
- **68.44% Precision** - High confidence in churn predictions
- **55% Recall** - Catches majority of at-risk customers
- Enables targeted retention campaigns with measurable ROI

**Key Features:**
- ✅ Complete end-to-end ML pipeline (tested and verified)
- ✅ 3 ML algorithms trained and compared (Logistic Regression, Random Forest, XGBoost)
- ✅ Production model deployed and ready to use
- ✅ Comprehensive data validation and preprocessing
- ✅ Advanced feature engineering (10 engineered features)
- ✅ Automated model evaluation and selection
- ✅ Visualization and reporting capabilities

## 📊 Dataset

We use the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) containing:
- 7,043 customer records
- 21 features (demographics, services, account info)
- Binary target: Churn (Yes/No)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (tested with Python 3.12.7)
- pip or conda

### Installation & Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset from Kaggle
# Visit: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Download and place in: data/raw/telco_customer_churn.csv

# 3. Run the pipeline (see src/ modules for usage)
```

**The system has been tested and validated with:**
```
✅ Best Model: Logistic Regression
✅ Test ROC-AUC: 0.8574 (85.74%)
✅ Test Accuracy: 0.8136 (81.36%)
✅ Production model: models/production/best_model.pkl
```

### Making Predictions

```python
from src.models.predict_model import load_production_model, predict_proba
import pandas as pd

# Load the trained model
model = load_production_model('best_model')

# Load your data
new_customers = pd.read_csv('your_data.csv')

# Make predictions
churn_probabilities = predict_proba(model, new_customers)
```

> 📖 For detailed usage examples, see [QUICK_START.md](QUICK_START.md)

## 📁 Project Structure

```
CustomerChurn/
├── config.yaml                    # Complete configuration
├── requirements.txt               # All dependencies
├── download_data.py               # Data download script
├── test_basic.py                  # Basic pipeline test ✅
├── test_pipeline.py               # Full pipeline test ✅
│
├── data/
│   ├── raw/                       # Raw dataset (7,043 records)
│   └── processed/                 # Train/val/test splits
│
├── models/
│   ├── baseline/                  # Baseline models
│   ├── experiments/               # All trained models
│   └── production/                # Best model (85.74% ROC-AUC)
│       ├── best_model.pkl         # Production model
│       ├── scaler.pkl             # Feature scaler
│       └── feature_names.pkl      # Feature list
│
├── reports/figures/               # Visualizations
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
└── src/                           # Source code (11 modules)
    ├── data/                      # Loading, validation, preprocessing
    ├── features/                  # Feature engineering
    ├── models/                    # Training, prediction, evaluation
    ├── utils/                     # Logging, helpers
    └── visualization/             # Plotting functions
```

> 📖 For complete file inventory, see [FILES_CREATED.md](FILES_CREATED.md)

## 🔄 Pipeline Architecture

The system implements a complete ML pipeline with the following stages:

### 1. Data Loading & Validation
- Load 7,043 customer records from CSV
- Validate data quality (missing values, duplicates, distributions)
- Check for data integrity issues

### 2. Data Preprocessing
- Clean data and handle missing values (11 imputed)
- Remove outliers using IQR method
- Split into train (70%), validation (15%), test (15%)
- Stratified sampling to maintain class balance

### 3. Feature Engineering
- **Tenure Groups**: Categorize customer tenure
- **Service Features**: Total services, internet/phone flags
- **Contract Features**: Month-to-month indicator, paperless billing
- **Charge Features**: Average charges, charge ratios
- **Demographic Features**: Family status, senior indicators
- **Result**: 21 → 42 features after engineering and encoding

### 4. Model Training & Selection
- Train 3 algorithms: Logistic Regression, Random Forest, XGBoost
- Compare models on validation set using ROC-AUC
- Select best model (Logistic Regression: 83.66% val ROC-AUC)
- Evaluate on held-out test set (85.74% test ROC-AUC)

### 5. Model Deployment
- Save production model with scaler and feature names
- Generate visualizations (confusion matrix, ROC curve)
- Ready for batch or real-time predictions

## 📈 Results

### Model Performance (Test Set)

**Best Model: Logistic Regression**

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | **85.74%** | ⭐ Excellent discrimination ability |
| **Accuracy** | **81.36%** | Overall prediction accuracy |
| **Precision** | **68.44%** | 68% of predicted churners actually churn |
| **Recall** | **55.00%** | Catches 55% of actual churners |
| **F1-Score** | **60.99%** | Balanced precision-recall metric |

### Model Comparison (Validation Set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| **Logistic Regression** | 79.75% | 65.58% | 50.18% | 56.85% | **83.66%** ✅ |
| Random Forest | 79.94% | 67.16% | 48.04% | 56.02% | 82.81% |
| XGBoost | 79.19% | 63.44% | 51.25% | 56.69% | 82.72% |

**Winner**: Logistic Regression selected based on highest ROC-AUC

### Business Impact

With this model, you can:
- 🎯 **Identify high-risk customers** with 85.74% accuracy (ROC-AUC)
- 💰 **Target retention campaigns** to 68% true churners (Precision)
- 📊 **Reduce churn** by proactively reaching 55% of at-risk customers (Recall)
- 💡 **Optimize marketing spend** by focusing on customers most likely to churn

> 🔍 For complete analysis, see [FINAL_SUMMARY.md](FINAL_SUMMARY.md)

## 🛠️ Technologies Used

- **ML Libraries:** scikit-learn (1.5.1), XGBoost (3.1.2)
- **Data Processing:** Pandas (2.3.3), NumPy (1.26.4)
- **Visualization:** Matplotlib (3.9.2), Seaborn (0.13.2)
- **Configuration:** PyYAML
- **Python Version:** 3.8+ (tested with 3.12.7)

## 📚 Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)** - First-time user orientation guide
- **[QUICK_START.md](QUICK_START.md)** - 5-minute getting started guide
- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - Complete project summary with results
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Detailed implementation status
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Mounish V**
- GitHub: [@Mounish-AV](https://github.com/Mounish-AV)

## 🙏 Acknowledgments

- Telco Customer Churn dataset from Kaggle
- CRISP-DM methodology documentation
- Open source community

---

⭐ If you find this project helpful, please consider giving it a star!
