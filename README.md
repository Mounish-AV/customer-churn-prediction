# Customer Churn Prediction - End-to-End ML Project

## Project Overview
This project predicts customer churn for a telecommunications company using the CRISP-DM methodology. We'll use the Telco Customer Churn dataset to identify customers likely to leave the service.

---

## Project Structure

```
customer-churn-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
│
├── data/
│   ├── raw/
│   │   └── telco_customer_churn.csv
│   ├── processed/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── validation.csv
│   └── external/
│       └── data_dictionary.txt
│
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_deployment_demo.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   ├── validate_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   ├── predict_model.py
│   │   └── evaluate_model.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualize.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── helpers.py
│
├── models/
│   ├── baseline/
│   │   └── logistic_regression.pkl
│   ├── experiments/
│   │   ├── random_forest_v1.pkl
│   │   ├── xgboost_v1.pkl
│   │   └── neural_network_v1.pkl
│   └── production/
│       ├── best_model.pkl
│       ├── scaler.pkl
│       └── feature_names.pkl
│
├── reports/
│   ├── figures/
│   │   ├── data_distribution.png
│   │   ├── correlation_matrix.png
│   │   ├── feature_importance.png
│   │   └── roc_curves.png
│   ├── business_understanding_report.md
│   ├── data_quality_report.md
│   ├── model_evaluation_report.md
│   └── final_presentation.pdf
│
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_features.py
│   └── test_models.py
│
├── deployment/
│   ├── api/
│   │   ├── app.py
│   │   ├── schemas.py
│   │   └── requirements.txt
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── monitoring/
│       ├── monitor_performance.py
│       └── drift_detection.py
│
└── docs/
    ├── project_charter.md
    ├── data_dictionary.md
    ├── model_card.md
    ├── deployment_guide.md
    └── user_guide.md
```

---

## File Descriptions

### Root Level Files

**README.md**
- Project overview and objectives
- Installation instructions
- Quick start guide
- Project structure explanation
- Contributors and license

**requirements.txt**
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==1.7.6
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
fastapi==0.103.0
pydantic==2.3.0
pytest==7.4.0
pyyaml==6.0.1
```

**config.yaml**
- Project configurations
- Hyperparameters
- File paths
- Model settings

**.gitignore**
- Ignore data files, models, cache
- Python artifacts

---

### Data Directory

**data/raw/**
- Original, immutable data
- `telco_customer_churn.csv`: Raw dataset

**data/processed/**
- Cleaned and transformed data
- `train.csv`: Training set (70%)
- `validation.csv`: Validation set (15%)
- `test.csv`: Test set (15%)

**data/external/**
- External reference data
- Data dictionaries and metadata

---

### Notebooks Directory

**01_business_understanding.ipynb**
- Problem definition
- Success metrics
- Stakeholder analysis
- Business impact assessment
- Project timeline

**02_data_understanding.ipynb**
- Initial data exploration
- Descriptive statistics
- Data quality assessment
- Missing value analysis
- Target variable distribution
- Correlation analysis

**03_data_preparation.ipynb**
- Data cleaning steps
- Handling missing values
- Outlier treatment
- Feature engineering
- Encoding categorical variables
- Train-test split
- Data scaling/normalization

**04_modeling.ipynb**
- Baseline model creation
- Multiple algorithm testing
- Hyperparameter tuning
- Cross-validation
- Model comparison
- Feature importance analysis

**05_evaluation.ipynb**
- Model performance metrics
- Confusion matrix analysis
- ROC-AUC curves
- Precision-Recall analysis
- Business metric evaluation
- Model interpretation (SHAP values)
- Error analysis

**06_deployment_demo.ipynb**
- Loading production model
- Making predictions
- API usage examples
- Monitoring examples

---

### Source Code (src/) Directory

**src/data/load_data.py**
- Functions to load raw data
- Database connection handlers
- API data fetchers

**src/data/validate_data.py**
- Data schema validation
- Quality checks
- Assertion tests

**src/data/preprocess.py**
- Data cleaning functions
- Missing value imputation
- Outlier handling
- Data type conversions

**src/features/build_features.py**
- Feature extraction
- Feature selection
- Feature transformation

**src/features/feature_engineering.py**
- Creating interaction features
- Polynomial features
- Domain-specific features

**src/models/train_model.py**
- Model training pipelines
- Hyperparameter tuning functions
- Cross-validation logic

**src/models/predict_model.py**
- Prediction functions
- Batch prediction handlers
- Real-time prediction logic

**src/models/evaluate_model.py**
- Metric calculation functions
- Model comparison utilities
- Performance reporting

**src/visualization/visualize.py**
- Plotting functions
- Dashboard creation
- Report generation

**src/utils/logger.py**
- Logging configuration
- Custom logger setup

**src/utils/helpers.py**
- Common utility functions
- Helper decorators
- Configuration loaders

---

### Models Directory

**models/baseline/**
- Simple baseline models for comparison

**models/experiments/**
- Different model versions during experimentation

**models/production/**
- Final selected model
- Associated preprocessing artifacts (scalers, encoders)
- Feature metadata

---

### Reports Directory

**reports/figures/**
- All visualizations and plots

**business_understanding_report.md**
- Problem statement
- Business objectives
- Success criteria
- Constraints and assumptions

**data_quality_report.md**
- Data profiling results
- Quality issues found
- Data cleaning decisions

**model_evaluation_report.md**
- Model performance summary
- Comparison of different models
- Final model selection rationale
- Recommendations

**final_presentation.pdf**
- Executive summary
- Key findings
- Business recommendations

---

### Tests Directory

**test_data_processing.py**
- Unit tests for data functions
- Data validation tests

**test_features.py**
- Feature engineering tests
- Feature validation

**test_models.py**
- Model training tests
- Prediction tests
- Performance tests

---

### Deployment Directory

**deployment/api/app.py**
- FastAPI application
- Prediction endpoints
- Health check endpoints

**deployment/api/schemas.py**
- Input/output data models
- Validation schemas

**deployment/docker/Dockerfile**
- Container configuration
- Dependencies installation

**deployment/docker/docker-compose.yml**
- Multi-container setup
- Service orchestration

**deployment/monitoring/monitor_performance.py**
- Model performance tracking
- Logging predictions
- Alerting logic

**deployment/monitoring/drift_detection.py**
- Data drift detection
- Concept drift monitoring
- Retraining triggers

---

### Documentation Directory

**project_charter.md**
- Project scope and goals
- Team roles
- Timeline and milestones

**data_dictionary.md**
- Feature descriptions
- Data types
- Value ranges

**model_card.md**
- Model details
- Intended use
- Performance metrics
- Limitations and biases
- Ethical considerations

**deployment_guide.md**
- Deployment instructions
- Configuration steps
- Monitoring setup

**user_guide.md**
- How to use the system
- API documentation
- Troubleshooting

---

## CRISP-DM Phase Mapping

### Phase 1: Business Understanding
- `notebooks/01_business_understanding.ipynb`
- `docs/project_charter.md`
- `reports/business_understanding_report.md`

### Phase 2: Data Understanding
- `notebooks/02_data_understanding.ipynb`
- `src/data/validate_data.py`
- `reports/data_quality_report.md`
- `docs/data_dictionary.md`

### Phase 3: Data Preparation
- `notebooks/03_data_preparation.ipynb`
- `src/data/preprocess.py`
- `src/features/build_features.py`
- `src/features/feature_engineering.py`
- `data/processed/`

### Phase 4: Modeling
- `notebooks/04_modeling.ipynb`
- `src/models/train_model.py`
- `models/experiments/`

### Phase 5: Evaluation
- `notebooks/05_evaluation.ipynb`
- `src/models/evaluate_model.py`
- `reports/model_evaluation_report.md`
- `reports/figures/`

### Phase 6: Deployment
- `notebooks/06_deployment_demo.ipynb`
- `deployment/api/`
- `deployment/docker/`
- `deployment/monitoring/`
- `models/production/`
- `docs/deployment_guide.md`

---

## Getting Started

### 1. Clone the repository
```bash
git clone <repository-url>
cd customer-churn-prediction
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Place the Telco Customer Churn dataset in `data/raw/`

### 5. Run notebooks sequentially
Start with `01_business_understanding.ipynb` and proceed through each phase

### 6. Run the API locally
```bash
cd deployment/api
uvicorn app:app --reload
```

---

## Key Metrics

- **Business Metric**: Customer retention rate improvement
- **Model Metrics**: Precision, Recall, F1-Score, ROC-AUC
- **Target**: 85%+ accuracy, 80%+ recall for churn class

---

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests: `pytest tests/`
4. Submit a pull request

---

## License

MIT License

---

## Contact

Project Lead: [Your Name]
Email: [your.email@example.com]
