"""Tests for fraud detection models."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fraud_detection.models import FraudDetectionPipeline
from fraud_detection.data import TransactionDataGenerator, DataProcessor


class TestFraudDetectionPipeline:
    """Test cases for FraudDetectionPipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        generator = TransactionDataGenerator(random_state=42, fraud_rate=0.1)
        return generator.generate_transactions(n_transactions=1000)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = FraudDetectionPipeline(random_state=42)
        assert pipeline.random_state == 42
        assert not pipeline.is_trained
    
    def test_pipeline_training(self, sample_data):
        """Test pipeline training."""
        # Prepare data
        processor = DataProcessor()
        X = sample_data.drop(['is_fraud', 'transaction_id', 'customer_id', 'timestamp'], axis=1, errors='ignore')
        y = sample_data['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = processor.split_data(
            processor.prepare_features(pd.concat([X, y], axis=1), fit=True)
        )
        
        # Train pipeline
        pipeline = FraudDetectionPipeline(random_state=42)
        pipeline.fit(X_train, y_train, X_test, y_test)
        
        # Check training
        assert pipeline.is_trained
        assert len(pipeline.models) > 0
        assert len(pipeline.feature_importance) > 0
    
    def test_pipeline_prediction(self, sample_data):
        """Test pipeline prediction."""
        # Prepare data
        processor = DataProcessor()
        X = sample_data.drop(['is_fraud', 'transaction_id', 'customer_id', 'timestamp'], axis=1, errors='ignore')
        y = sample_data['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = processor.split_data(
            processor.prepare_features(pd.concat([X, y], axis=1), fit=True)
        )
        
        # Train pipeline
        pipeline = FraudDetectionPipeline(random_state=42)
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        
        # Check predictions
        assert len(predictions) == len(X_test)
        assert len(probabilities) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)
        assert all(0 <= prob <= 1 for prob in probabilities)
    
    def test_pipeline_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        # Prepare data
        processor = DataProcessor()
        X = sample_data.drop(['is_fraud', 'transaction_id', 'customer_id', 'timestamp'], axis=1, errors='ignore')
        y = sample_data['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = processor.split_data(
            processor.prepare_features(pd.concat([X, y], axis=1), fit=True)
        )
        
        # Train pipeline
        pipeline = FraudDetectionPipeline(random_state=42)
        pipeline.fit(X_train, y_train)
        
        # Get feature importance
        importance = pipeline.get_feature_importance()
        
        # Check importance
        assert not importance.empty
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert 'model' in importance.columns
    
    def test_pipeline_model_performance(self, sample_data):
        """Test model performance evaluation."""
        # Prepare data
        processor = DataProcessor()
        X = sample_data.drop(['is_fraud', 'transaction_id', 'customer_id', 'timestamp'], axis=1, errors='ignore')
        y = sample_data['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = processor.split_data(
            processor.prepare_features(pd.concat([X, y], axis=1), fit=True)
        )
        
        # Train pipeline
        pipeline = FraudDetectionPipeline(random_state=42)
        pipeline.fit(X_train, y_train, X_test, y_test)
        
        # Get performance
        performance = pipeline.get_model_performance()
        
        # Check performance
        if not performance.empty:
            assert 'model' in performance.columns
            assert 'auc' in performance.columns
            assert 'precision' in performance.columns
            assert 'recall' in performance.columns
            assert 'f1' in performance.columns
    
    def test_pipeline_save_load(self, sample_data, tmp_path):
        """Test model saving and loading."""
        # Prepare data
        processor = DataProcessor()
        X = sample_data.drop(['is_fraud', 'transaction_id', 'customer_id', 'timestamp'], axis=1, errors='ignore')
        y = sample_data['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = processor.split_data(
            processor.prepare_features(pd.concat([X, y], axis=1), fit=True)
        )
        
        # Train pipeline
        pipeline = FraudDetectionPipeline(random_state=42)
        pipeline.fit(X_train, y_train)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        pipeline.save_model(str(model_path))
        
        # Load model
        new_pipeline = FraudDetectionPipeline()
        new_pipeline.load_model(str(model_path))
        
        # Check loaded model
        assert new_pipeline.is_trained
        assert len(new_pipeline.models) == len(pipeline.models)
        
        # Test predictions are the same
        original_pred = pipeline.predict(X_test)
        loaded_pred = new_pipeline.predict(X_test)
        np.testing.assert_array_equal(original_pred, loaded_pred)


if __name__ == "__main__":
    pytest.main([__file__])
