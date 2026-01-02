# Customer Churn Prediction

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An end-to-end machine learning project to predict customer churn for telecommunications companies, built using the CRISP-DM methodology.

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

This project identifies customers at risk of churning (leaving the service) using machine learning techniques. The goal is to help businesses:
- Reduce customer attrition by 15-20%
- Identify high-risk customers proactively
- Optimize retention campaign targeting

**Key Features:**
- Complete CRISP-DM implementation
- Multiple ML algorithms comparison
- Production-ready API deployment
- Comprehensive model monitoring

## 📊 Dataset

We use the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) containing:
- 7,043 customer records
- 21 features (demographics, services, account info)
- Binary target: Churn (Yes/No)

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/Mounish-AV/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Place telco_customer_churn.csv in data/raw/
```

### Running the Project

```bash
# Run notebooks in sequence
jupyter notebook notebooks/

# Or run the entire pipeline
python src/pipeline.py

# Start the API server
cd deployment/api
uvicorn app:app --reload
```

## 📁 Project Structure

```
customer-churn-prediction/
├── data/                   # Raw and processed data
├── notebooks/              # Jupyter notebooks (01-06)
├── src/                    # Source code modules
├── models/                 # Trained models
├── reports/                # Analysis reports and figures
├── deployment/             # API and Docker files
├── tests/                  # Unit tests
└── docs/                   # Additional documentation
```

> 📖 For detailed file descriptions, see [docs/project_structure.md](docs/project_structure.md)

## 🔄 Methodology

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:

```
[Business Understanding] → [Data Understanding] → [Data Preparation]
          ↑                                              ↓
   [Deployment] ← [Evaluation] ← [Modeling]
```

### Notebooks Overview

| Notebook | Phase | Description |
|----------|-------|-------------|
| `01_business_understanding.ipynb` | Phase 1 | Problem definition & success metrics |
| `02_data_understanding.ipynb` | Phase 2 | Exploratory data analysis |
| `03_data_preparation.ipynb` | Phase 3 | Data cleaning & feature engineering |
| `04_modeling.ipynb` | Phase 4 | Model training & comparison |
| `05_evaluation.ipynb` | Phase 5 | Model evaluation & selection |
| `06_deployment_demo.ipynb` | Phase 6 | Deployment demonstration |

## 📈 Results

**Model Performance (Test Set):**
- Accuracy: 82%
- Precision: 78%
- Recall: 85%
- F1-Score: 81%
- ROC-AUC: 0.87

> 🔍 Detailed results and model comparison available in [reports/model_evaluation_report.md](reports/model_evaluation_report.md)

## 🛠️ Technologies Used

- **ML Libraries:** scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **API:** FastAPI
- **Deployment:** Docker
- **Testing:** pytest

## 🤝 Contributing

Contributions are welcome! Please check out our [Contributing Guidelines](CONTRIBUTING.md).

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Mounish AV**
- GitHub: [@Mounish-AV](https://github.com/Mounish-AV)

## 🙏 Acknowledgments

- Telco Customer Churn dataset from Kaggle
- CRISP-DM methodology documentation
- Open source community

---

⭐ If you find this project helpful, please consider giving it a star!
