#!/usr/bin/env python3
"""Example usage of the fraud detection system.

This script demonstrates how to use the fraud detection system
for training models and making predictions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from fraud_detection import (
    TransactionDataGenerator,
    FraudDetectionPipeline,
    FraudDetectionEvaluator
)
from fraud_detection.viz import FraudExplainer


def main():
    """Main example function."""
    print("🔍 Fraud Detection System Example")
    print("=" * 50)
    
    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic transaction data...")
    generator = TransactionDataGenerator(random_state=42, fraud_rate=0.1)
    data = generator.generate_transactions(n_transactions=5000)
    
    print(f"   Generated {len(data)} transactions")
    print(f"   Fraud rate: {data['is_fraud'].mean():.1%}")
    print(f"   Average amount: ${data['amount'].mean():.2f}")
    
    # Step 2: Prepare data for training
    print("\n2. Preparing data for training...")
    from fraud_detection.data.processor import DataProcessor
    
    processor = DataProcessor()
    X = data.drop(['is_fraud', 'transaction_id', 'customer_id', 'timestamp'], axis=1, errors='ignore')
    y = data['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(
        processor.prepare_features(pd.concat([X, y], axis=1), fit=True)
    )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Step 3: Train the fraud detection pipeline
    print("\n3. Training fraud detection models...")
    pipeline = FraudDetectionPipeline(random_state=42)
    pipeline.fit(X_train, y_train, X_test, y_test)
    
    print("   Models trained successfully!")
    print(f"   Available models: {list(pipeline.models.keys())}")
    
    # Step 4: Make predictions
    print("\n4. Making predictions...")
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    
    print(f"   Predicted fraud rate: {predictions.mean():.1%}")
    print(f"   Average fraud probability: {probabilities.mean():.3f}")
    
    # Step 5: Evaluate model performance
    print("\n5. Evaluating model performance...")
    evaluator = FraudDetectionEvaluator(pipeline)
    results = evaluator.evaluate(X_test, y_test, X_test)
    
    # Display key metrics
    ml_metrics = results['ml_metrics']
    business_metrics = results['business_metrics']
    
    print("   Machine Learning Metrics:")
    print(f"     AUC: {ml_metrics['roc_auc']:.3f}")
    print(f"     Precision: {ml_metrics['precision']:.3f}")
    print(f"     Recall: {ml_metrics['recall']:.3f}")
    print(f"     F1 Score: {ml_metrics['f1_score']:.3f}")
    
    print("   Business Metrics:")
    print(f"     Fraud Detection Rate: {business_metrics['fraud_detection_rate']:.1%}")
    print(f"     Alert Rate: {business_metrics['alert_rate']:.1%}")
    print(f"     ROI: {business_metrics['roi']:.1%}")
    
    # Step 6: Model explainability
    print("\n6. Generating model explanations...")
    try:
        explainer = FraudExplainer(pipeline, X_train)
        
        # Global feature importance
        global_explanation = explainer.explain_global(X_test)
        top_features = global_explanation['global_importance'][:5]
        
        print("   Top 5 Most Important Features:")
        for i, feature in enumerate(top_features, 1):
            print(f"     {i}. {feature['feature']}: {feature['mean_abs_shap']:.4f}")
        
        # Individual prediction explanation
        individual_explanation = explainer.explain_prediction(X_test.iloc[[0]])
        explanation = individual_explanation['explanations'][0]
        
        print(f"\n   Example Prediction Explanation:")
        print(f"     Fraud Probability: {explanation['fraud_probability']:.3f}")
        print("     Top Contributing Features:")
        for feature in explanation['top_features'][:3]:
            print(f"       - {feature['feature']}: {feature['shap_value']:.4f}")
            
    except ImportError:
        print("   SHAP not available. Install with: pip install shap")
    
    # Step 7: Model comparison
    print("\n7. Model Performance Comparison:")
    performance = pipeline.get_model_performance()
    
    if not performance.empty:
        for _, row in performance.iterrows():
            print(f"   {row['model']}: AUC={row['auc']:.3f}, F1={row['f1']:.3f}")
    
    # Step 8: Save model
    print("\n8. Saving trained model...")
    model_path = "models/example_fraud_detection_model.pkl"
    Path("models").mkdir(exist_ok=True)
    pipeline.save_model(model_path)
    print(f"   Model saved to {model_path}")
    
    # Step 9: Example of loading and using saved model
    print("\n9. Loading saved model for new predictions...")
    new_pipeline = FraudDetectionPipeline()
    new_pipeline.load_model(model_path)
    
    # Make prediction on new transaction
    new_transaction = pd.DataFrame([{
        'amount': 1500.0,
        'frequency': 2,
        'is_foreign': 1,
        'is_high_risk_country': 1,
        'is_weekend': 1,
        'is_night': 0,
        'merchant_category': 'online',
        'device_type': 'mobile',
        'channel': 'online',
        'is_high_risk_merchant': 1
    }])
    
    fraud_prob = new_pipeline.predict_proba(new_transaction)[0]
    prediction = new_pipeline.predict(new_transaction)[0]
    
    print(f"   New Transaction Prediction:")
    print(f"     Amount: ${new_transaction['amount'].iloc[0]:.2f}")
    print(f"     Fraud Probability: {fraud_prob:.3f}")
    print(f"     Prediction: {'🚨 FRAUD' if prediction else '✅ NORMAL'}")
    
    print("\n✅ Example completed successfully!")
    print("\n📝 Next Steps:")
    print("   - Run 'streamlit run demo/app.py' for interactive demo")
    print("   - Check 'assets/' directory for generated plots and reports")
    print("   - Review 'DISCLAIMER.md' for important usage information")


if __name__ == "__main__":
    main()
