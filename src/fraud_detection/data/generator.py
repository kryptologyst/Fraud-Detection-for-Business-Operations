"""Transaction data generator for fraud detection.

This module provides functionality to generate synthetic transaction data
for training and testing fraud detection models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import random


class TransactionDataGenerator:
    """Generate synthetic transaction data for fraud detection.
    
    This class creates realistic transaction datasets with various features
    that can be used to train and evaluate fraud detection models.
    
    Attributes:
        random_state: Random seed for reproducibility
        fraud_rate: Proportion of fraudulent transactions (default: 0.1)
    """
    
    def __init__(self, random_state: int = 42, fraud_rate: float = 0.1) -> None:
        """Initialize the transaction data generator.
        
        Args:
            random_state: Random seed for reproducibility
            fraud_rate: Proportion of fraudulent transactions (0.0 to 1.0)
        """
        self.random_state = random_state
        self.fraud_rate = fraud_rate
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Define risk countries and merchant categories
        self.risk_countries = [
            'CN', 'RU', 'NG', 'BR', 'MX', 'IN', 'PK', 'BD', 'VN', 'TH'
        ]
        self.merchant_categories = [
            'grocery', 'gas_station', 'restaurant', 'retail', 'online',
            'travel', 'entertainment', 'healthcare', 'education', 'other'
        ]
        self.high_risk_merchants = ['online', 'travel', 'entertainment']
        
    def generate_transactions(
        self, 
        n_transactions: int = 10000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Generate synthetic transaction data.
        
        Args:
            n_transactions: Number of transactions to generate
            start_date: Start date for transaction timestamps
            end_date: End date for transaction timestamps
            
        Returns:
            DataFrame with transaction data including fraud labels
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
            
        # Generate base transaction data
        transactions = []
        
        for i in range(n_transactions):
            # Determine if transaction is fraudulent
            is_fraud = np.random.random() < self.fraud_rate
            
            # Generate transaction features
            transaction = self._generate_single_transaction(
                transaction_id=i,
                is_fraud=is_fraud,
                start_date=start_date,
                end_date=end_date
            )
            transactions.append(transaction)
            
        df = pd.DataFrame(transactions)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        return df
    
    def _generate_single_transaction(
        self,
        transaction_id: int,
        is_fraud: bool,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Generate a single transaction with realistic features.
        
        Args:
            transaction_id: Unique transaction identifier
            is_fraud: Whether this transaction is fraudulent
            start_date: Start date for timestamp generation
            end_date: End date for timestamp generation
            
        Returns:
            Dictionary with transaction features
        """
        # Generate timestamp
        time_range = (end_date - start_date).total_seconds()
        random_seconds = np.random.uniform(0, time_range)
        timestamp = start_date + timedelta(seconds=random_seconds)
        
        # Generate customer ID (some customers are more likely to commit fraud)
        customer_id = np.random.randint(1, 1001)
        
        # Generate merchant and location features
        merchant_category = np.random.choice(self.merchant_categories)
        is_foreign = np.random.choice([0, 1], p=[0.7, 0.3])
        is_high_risk_country = 0
        if is_foreign:
            is_high_risk_country = np.random.choice([0, 1], p=[0.6, 0.4])
            
        # Generate amount (fraudulent transactions tend to be larger)
        if is_fraud:
            # Fraudulent transactions: higher amounts, more likely to be foreign
            amount = np.random.lognormal(mean=6.5, sigma=1.2)  # Higher mean
            if np.random.random() < 0.7:  # 70% of fraud is foreign
                is_foreign = 1
                is_high_risk_country = np.random.choice([0, 1], p=[0.3, 0.7])
        else:
            # Normal transactions: smaller amounts
            amount = np.random.lognormal(mean=4.8, sigma=1.0)
            
        # Cap extreme amounts
        amount = min(amount, 50000)
        
        # Generate frequency (transactions per day for this customer)
        if is_fraud:
            frequency = np.random.poisson(2.5)  # Lower frequency for fraud
        else:
            frequency = np.random.poisson(5.0)  # Higher frequency for normal
            
        # Generate time-based features
        is_weekend = 1 if timestamp.weekday() >= 5 else 0
        is_night = 1 if timestamp.hour < 6 or timestamp.hour > 22 else 0
        
        # Generate device and channel features
        device_type = np.random.choice(['mobile', 'desktop', 'tablet'], p=[0.6, 0.3, 0.1])
        channel = np.random.choice(['online', 'pos', 'atm'], p=[0.4, 0.5, 0.1])
        
        # Generate additional risk indicators
        is_high_risk_merchant = 1 if merchant_category in self.high_risk_merchants else 0
        
        # For fraud, increase risk indicators
        if is_fraud:
            if np.random.random() < 0.6:  # 60% of fraud at night
                is_night = 1
            if np.random.random() < 0.4:  # 40% of fraud on weekends
                is_weekend = 1
            if np.random.random() < 0.5:  # 50% of fraud from high-risk merchants
                is_high_risk_merchant = 1
                
        return {
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': merchant_category,
            'is_foreign': is_foreign,
            'is_high_risk_country': is_high_risk_country,
            'is_weekend': is_weekend,
            'is_night': is_night,
            'device_type': device_type,
            'channel': channel,
            'is_high_risk_merchant': is_high_risk_merchant,
            'frequency': frequency,
            'is_fraud': int(is_fraud)
        }
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the transaction data.
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            DataFrame with additional derived features
        """
        # Sort by customer and timestamp for rolling calculations
        df = df.sort_values(['customer_id', 'timestamp']).reset_index(drop=True)
        
        # Customer-level features
        customer_stats = df.groupby('customer_id').agg({
            'amount': ['mean', 'std', 'max'],
            'frequency': 'mean',
            'is_fraud': 'sum'
        }).round(2)
        
        customer_stats.columns = [
            'customer_avg_amount', 'customer_std_amount', 'customer_max_amount',
            'customer_avg_frequency', 'customer_fraud_count'
        ]
        
        # Merge customer stats back
        df = df.merge(customer_stats, left_on='customer_id', right_index=True, how='left')
        
        # Fill NaN values for customers with single transactions
        df['customer_std_amount'] = df['customer_std_amount'].fillna(0)
        
        # Amount-based risk indicators
        df['amount_zscore'] = (df['amount'] - df['customer_avg_amount']) / (df['customer_std_amount'] + 1e-6)
        df['is_high_amount'] = (df['amount'] > df['customer_avg_amount'] + 2 * df['customer_std_amount']).astype(int)
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Risk score (simple heuristic)
        risk_factors = [
            'is_foreign', 'is_high_risk_country', 'is_weekend', 'is_night',
            'is_high_risk_merchant', 'is_high_amount'
        ]
        df['risk_score'] = df[risk_factors].sum(axis=1)
        
        return df
    
    def generate_customer_profiles(self, n_customers: int = 1000) -> pd.DataFrame:
        """Generate customer profile data.
        
        Args:
            n_customers: Number of customers to generate
            
        Returns:
            DataFrame with customer profile information
        """
        customers = []
        
        for i in range(1, n_customers + 1):
            # Generate customer demographics (anonymized)
            age_group = np.random.choice(['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
            income_bracket = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
            
            # Generate account features
            account_age_days = np.random.randint(30, 3650)  # 1 month to 10 years
            credit_score = np.random.normal(650, 100)
            credit_score = max(300, min(850, credit_score))  # Clamp to valid range
            
            # Generate transaction patterns
            avg_monthly_transactions = np.random.poisson(20)
            avg_transaction_amount = np.random.lognormal(4.5, 0.8)
            
            customers.append({
                'customer_id': i,
                'age_group': age_group,
                'income_bracket': income_bracket,
                'account_age_days': account_age_days,
                'credit_score': round(credit_score),
                'avg_monthly_transactions': avg_monthly_transactions,
                'avg_transaction_amount': round(avg_transaction_amount, 2)
            })
            
        return pd.DataFrame(customers)
