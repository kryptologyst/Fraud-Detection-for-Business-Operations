"""Data processing utilities for fraud detection.

This module provides data preprocessing and feature engineering functionality
for fraud detection models.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings


class DataProcessor:
    """Data processor for fraud detection datasets.
    
    This class handles data preprocessing, feature engineering, and
    train/test splitting for fraud detection models.
    
    Attributes:
        scaler: StandardScaler for numerical features
        label_encoders: Dictionary of LabelEncoders for categorical features
        feature_columns: List of feature column names
        target_column: Name of the target column
    """
    
    def __init__(self, target_column: str = 'is_fraud') -> None:
        """Initialize the data processor.
        
        Args:
            target_column: Name of the target column
        """
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        fit: bool = True
    ) -> pd.DataFrame:
        """Prepare features for model training.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the encoders/scalers
            
        Returns:
            DataFrame with processed features
        """
        df_processed = df.copy()
        
        # Identify feature columns (exclude target and metadata)
        exclude_columns = [
            self.target_column, 'transaction_id', 'customer_id', 'timestamp'
        ]
        self.feature_columns = [
            col for col in df_processed.columns 
            if col not in exclude_columns
        ]
        
        # Separate categorical and numerical features
        self.categorical_columns = df_processed[self.feature_columns].select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        self.numerical_columns = df_processed[self.feature_columns].select_dtypes(
            include=[np.number]
        ).columns.tolist()
        
        # Handle categorical features
        for col in self.categorical_columns:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(
                    df_processed[col].astype(str)
                )
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    try:
                        df_processed[col] = self.label_encoders[col].transform(
                            df_processed[col].astype(str)
                        )
                    except ValueError:
                        # Handle unseen categories by assigning them to the most common class
                        warnings.warn(f"Unseen categories in {col}, assigning to most common class")
                        most_common = self.label_encoders[col].classes_[0]
                        df_processed[col] = df_processed[col].replace(
                            df_processed[~df_processed[col].isin(self.label_encoders[col].classes_)][col].unique(),
                            most_common
                        )
                        df_processed[col] = self.label_encoders[col].transform(
                            df_processed[col].astype(str)
                        )
        
        # Handle missing values
        df_processed = self._handle_missing_values(df_processed)
        
        # Scale numerical features
        if self.numerical_columns:
            if fit:
                df_processed[self.numerical_columns] = self.scaler.fit_transform(
                    df_processed[self.numerical_columns]
                )
            else:
                df_processed[self.numerical_columns] = self.scaler.transform(
                    df_processed[self.numerical_columns]
                )
        
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        # For numerical columns, fill with median
        for col in self.numerical_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns, fill with mode
        for col in self.categorical_columns:
            if df[col].isnull().any():
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col] = df[col].fillna(mode_value[0])
                else:
                    df[col] = df[col].fillna(0)
        
        return df
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets.
        
        Args:
            df: Processed DataFrame
            test_size: Proportion of data for testing
            stratify: Whether to stratify by target variable
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        if stratify and y.nunique() > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state,
                stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state
            )
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance_data(
        self, 
        feature_importance: np.ndarray
    ) -> pd.DataFrame:
        """Create DataFrame with feature importance information.
        
        Args:
            feature_importance: Array of feature importance values
            
        Returns:
            DataFrame with feature names and importance scores
        """
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def inverse_transform_categorical(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Inverse transform categorical features back to original values.
        
        Args:
            df: DataFrame with encoded categorical features
            columns: Specific columns to inverse transform (all if None)
            
        Returns:
            DataFrame with original categorical values
        """
        df_inverse = df.copy()
        
        if columns is None:
            columns = self.categorical_columns
        
        for col in columns:
            if col in self.label_encoders:
                df_inverse[col] = self.label_encoders[col].inverse_transform(
                    df_inverse[col].astype(int)
                )
        
        return df_inverse
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data summary information
        """
        summary = {
            'total_transactions': len(df),
            'fraud_rate': df[self.target_column].mean(),
            'fraud_count': df[self.target_column].sum(),
            'normal_count': (df[self.target_column] == 0).sum(),
            'feature_count': len(self.feature_columns),
            'categorical_features': len(self.categorical_columns),
            'numerical_features': len(self.numerical_columns),
            'missing_values': df[self.feature_columns].isnull().sum().sum(),
            'duplicate_transactions': df.duplicated().sum()
        }
        
        # Add feature-wise statistics
        summary['feature_stats'] = {}
        for col in self.feature_columns:
            if col in self.numerical_columns:
                summary['feature_stats'][col] = {
                    'type': 'numerical',
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'missing': df[col].isnull().sum()
                }
            else:
                summary['feature_stats'][col] = {
                    'type': 'categorical',
                    'unique_values': df[col].nunique(),
                    'most_common': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                    'missing': df[col].isnull().sum()
                }
        
        return summary
