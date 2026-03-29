"""Tests for data generation and processing modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fraud_detection.data import TransactionDataGenerator, DataProcessor


class TestTransactionDataGenerator:
    """Test cases for TransactionDataGenerator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = TransactionDataGenerator(random_state=42, fraud_rate=0.1)
        assert generator.random_state == 42
        assert generator.fraud_rate == 0.1
    
    def test_generate_transactions(self):
        """Test transaction generation."""
        generator = TransactionDataGenerator(random_state=42, fraud_rate=0.1)
        data = generator.generate_transactions(n_transactions=1000)
        
        # Check basic structure
        assert len(data) == 1000
        assert 'is_fraud' in data.columns
        assert 'amount' in data.columns
        assert 'customer_id' in data.columns
        
        # Check fraud rate
        fraud_rate = data['is_fraud'].mean()
        assert 0.05 <= fraud_rate <= 0.15  # Allow some variance
    
    def test_generate_customer_profiles(self):
        """Test customer profile generation."""
        generator = TransactionDataGenerator(random_state=42)
        profiles = generator.generate_customer_profiles(n_customers=100)
        
        assert len(profiles) == 100
        assert 'customer_id' in profiles.columns
        assert 'age_group' in profiles.columns
        assert 'income_bracket' in profiles.columns


class TestDataProcessor:
    """Test cases for DataProcessor."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = DataProcessor(target_column='is_fraud')
        assert processor.target_column == 'is_fraud'
    
    def test_prepare_features(self):
        """Test feature preparation."""
        # Create sample data
        data = pd.DataFrame({
            'amount': [100, 200, 300, 400, 500],
            'frequency': [5, 10, 15, 20, 25],
            'merchant_category': ['grocery', 'retail', 'online', 'travel', 'other'],
            'is_fraud': [0, 1, 0, 1, 0]
        })
        
        processor = DataProcessor()
        processed_data = processor.prepare_features(data, fit=True)
        
        # Check that categorical features are encoded
        assert 'merchant_category' not in processed_data.columns
        assert len(processed_data.columns) > len(data.columns) - 1  # Should have more features after processing
    
    def test_split_data(self):
        """Test data splitting."""
        # Create sample data
        data = pd.DataFrame({
            'amount': [100, 200, 300, 400, 500] * 20,
            'frequency': [5, 10, 15, 20, 25] * 20,
            'is_fraud': [0, 1, 0, 1, 0] * 20
        })
        
        processor = DataProcessor()
        processed_data = processor.prepare_features(data, fit=True)
        
        X_train, X_test, y_train, y_test = processor.split_data(processed_data, test_size=0.2)
        
        # Check split sizes
        assert len(X_train) + len(X_test) == len(processed_data)
        assert len(y_train) + len(y_test) == len(processed_data)
        
        # Check that features are separated from target
        assert 'is_fraud' not in X_train.columns
        assert 'is_fraud' not in X_test.columns
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        # Create sample data
        data = pd.DataFrame({
            'amount': [100, 200, 300, 400, 500],
            'frequency': [5, 10, 15, 20, 25],
            'merchant_category': ['grocery', 'retail', 'online', 'travel', 'other'],
            'is_fraud': [0, 1, 0, 1, 0]
        })
        
        processor = DataProcessor()
        processed_data = processor.prepare_features(data, fit=True)
        
        summary = processor.get_data_summary(processed_data)
        
        # Check summary structure
        assert 'total_transactions' in summary
        assert 'fraud_rate' in summary
        assert 'feature_count' in summary
        assert 'feature_stats' in summary
        
        # Check values
        assert summary['total_transactions'] == 5
        assert summary['fraud_rate'] == 0.4
        assert summary['feature_count'] > 0


if __name__ == "__main__":
    pytest.main([__file__])
