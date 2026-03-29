# Fraud Detection for Business Operations

A comprehensive fraud detection system for identifying anomalous transactions and behaviors in business operations. This system combines multiple machine learning approaches including supervised classification, unsupervised anomaly detection, and ensemble methods to provide robust fraud detection capabilities.

## DISCLAIMER

**This system is for research and educational purposes only. It should not be used for automated decision-making without human review.**

## Features

- **Multiple Detection Methods**: Combines Random Forest, Gradient Boosting, Logistic Regression, Isolation Forest, and One-Class SVM
- **Ensemble Learning**: Advanced ensemble methods for improved performance
- **Comprehensive Evaluation**: Both ML metrics and business-relevant KPIs
- **Explainability**: SHAP-based explanations for model decisions
- **Interactive Demo**: Streamlit-based web interface
- **Production Ready**: Proper structure, configuration, and documentation

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Fraud-Detection-for-Business-Operations.git
cd Fraud-Detection-for-Business-Operations
```

2. Install dependencies:
```bash
pip install -e .
```

3. Install development dependencies (optional):
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Generate Data and Train Models

```bash
# Generate synthetic data and train models
python scripts/train.py --data-size 5000 --fraud-rate 0.1
```

### 2. Run Interactive Demo

```bash
# Start Streamlit demo
streamlit run demo/app.py
```

### 3. Make Predictions

```bash
# Make predictions on new data
python scripts/predict.py --model models/fraud_detection_model.pkl --input data/new_transactions.csv
```

## Usage

### Data Generation

The system can generate synthetic transaction data for training and testing:

```python
from fraud_detection import TransactionDataGenerator

# Generate 10,000 transactions with 10% fraud rate
generator = TransactionDataGenerator(random_state=42, fraud_rate=0.1)
data = generator.generate_transactions(n_transactions=10000)
```

### Model Training

```python
from fraud_detection import FraudDetectionPipeline
from fraud_detection.data import DataProcessor

# Prepare data
processor = DataProcessor()
X = data.drop(['is_fraud', 'transaction_id', 'customer_id', 'timestamp'], axis=1)
y = data['is_fraud']

# Split data
X_train, X_test, y_train, y_test = processor.split_data(
    processor.prepare_features(pd.concat([X, y], axis=1), fit=True)
)

# Train pipeline
pipeline = FraudDetectionPipeline(random_state=42)
pipeline.fit(X_train, y_train, X_test, y_test)
```

### Making Predictions

```python
# Single prediction
fraud_prob = pipeline.predict_proba(new_transaction)[0]
prediction = pipeline.predict(new_transaction)[0]

# Batch predictions
predictions = pipeline.predict(batch_data)
probabilities = pipeline.predict_proba(batch_data)
```

### Model Evaluation

```python
from fraud_detection import FraudDetectionEvaluator

# Evaluate model
evaluator = FraudDetectionEvaluator(pipeline)
results = evaluator.evaluate(X_test, y_test, X_test)

# Get performance metrics
ml_metrics = results['ml_metrics']
business_metrics = results['business_metrics']
```

### Explainability

```python
from fraud_detection.viz import FraudExplainer

# Create explainer
explainer = FraudExplainer(pipeline, background_data=X_train)

# Explain individual predictions
explanations = explainer.explain_prediction(X_test)

# Global feature importance
global_explanation = explainer.explain_global(X_test)
```

## Configuration

The system uses YAML configuration files. See `configs/default.yaml` for available options:

```yaml
# Data generation settings
data:
  n_transactions: 10000
  fraud_rate: 0.1
  random_state: 42

# Model configurations
models:
  random_forest:
    n_estimators: 100
    max_depth: 10
    class_weight: "balanced"
  
  gradient_boosting:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 6

# Evaluation settings
evaluation:
  test_size: 0.2
  investigation_cost: 10.0
  fraud_loss_per_transaction: 100.0
```

## Data Schema

### Transaction Data

| Column | Type | Description |
|--------|------|-------------|
| transaction_id | int | Unique transaction identifier |
| customer_id | int | Customer identifier |
| timestamp | datetime | Transaction timestamp |
| amount | float | Transaction amount |
| merchant_category | str | Merchant category |
| is_foreign | int | Foreign transaction flag |
| is_high_risk_country | int | High-risk country flag |
| is_weekend | int | Weekend transaction flag |
| is_night | int | Night transaction flag |
| device_type | str | Device type (mobile/desktop/tablet) |
| channel | str | Transaction channel |
| is_high_risk_merchant | int | High-risk merchant flag |
| frequency | int | Daily transaction frequency |
| is_fraud | int | Fraud label (0/1) |

### Customer Profiles

| Column | Type | Description |
|--------|------|-------------|
| customer_id | int | Customer identifier |
| age_group | str | Age group |
| income_bracket | str | Income bracket |
| account_age_days | int | Account age in days |
| credit_score | int | Credit score |
| avg_monthly_transactions | int | Average monthly transactions |
| avg_transaction_amount | float | Average transaction amount |

## Model Architecture

The system uses a multi-layered approach:

1. **Data Processing**: Feature engineering, encoding, and scaling
2. **Base Models**: Multiple individual models (RF, GB, LR, IF, OCSVM)
3. **Ensemble Layer**: Combines base model predictions
4. **Evaluation**: Comprehensive performance assessment
5. **Explainability**: SHAP-based model interpretation

## Evaluation Metrics

### Machine Learning Metrics

- **AUC**: Area Under the ROC Curve
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Specificity**: True Negatives / (True Negatives + False Positives)

### Business Metrics

- **Fraud Detection Rate**: Percentage of fraud cases detected
- **Alert Rate**: Percentage of transactions flagged for review
- **Precision at K**: Precision of top K flagged transactions
- **ROI**: Return on investment from fraud prevention
- **Investigation Cost**: Cost of investigating flagged transactions

## Interactive Demo

The Streamlit demo provides:

- **Data Overview**: Transaction statistics and visualizations
- **Model Training**: Interactive model training interface
- **Predictions**: Single transaction and batch prediction tools
- **Evaluation**: Comprehensive model performance analysis
- **Explanations**: SHAP-based model interpretability

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src/fraud_detection
```

## Development

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking
- **Pre-commit**: Git hooks for quality checks

### Setup Development Environment

```bash
# Install pre-commit hooks
pre-commit install

# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

## Project Structure

```
fraud-detection-business/
├── src/fraud_detection/          # Main package
│   ├── data/                     # Data generation and processing
│   ├── models/                   # Model implementations
│   ├── eval/                     # Evaluation modules
│   ├── viz/                      # Visualization and explainability
│   └── utils/                    # Utility functions
├── configs/                      # Configuration files
├── scripts/                      # Command-line scripts
├── tests/                        # Test files
├── demo/                         # Streamlit demo
├── assets/                       # Generated plots and reports
├── data/                         # Data files
├── models/                       # Trained models
└── logs/                         # Log files
```

## Limitations

- **Synthetic Data**: Uses generated data for demonstration
- **Simplified Features**: Real-world systems may require more complex features
- **No Real-time Processing**: Designed for batch processing
- **Limited Scalability**: Not optimized for high-volume production use

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{fraud_detection_2024,
  title={Fraud Detection for Business Operations},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Fraud-Detection-for-Business-Operations}
}
```

## Support

For questions and support, please open an issue on GitHub.

## Changelog

### Version 1.0.0
- Initial release
- Multiple fraud detection models
- Ensemble learning
- SHAP explainability
- Interactive Streamlit demo
- Comprehensive evaluation framework
# Fraud-Detection-for-Business-Operations
