"""Explainability module for fraud detection.

This module provides SHAP-based explainability for fraud detection models,
including feature importance analysis and individual prediction explanations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

from ..models.pipeline import FraudDetectionPipeline


class FraudExplainer:
    """SHAP-based explainer for fraud detection models.
    
    This class provides comprehensive explainability for fraud detection
    models using SHAP (SHapley Additive exPlanations) values.
    
    Attributes:
        pipeline: Trained fraud detection pipeline
        explainer: SHAP explainer object
        background_data: Background data for SHAP explainer
        feature_names: List of feature names
    """
    
    def __init__(
        self,
        pipeline: FraudDetectionPipeline,
        background_data: Optional[pd.DataFrame] = None,
        max_background_samples: int = 100
    ) -> None:
        """Initialize the fraud explainer.
        
        Args:
            pipeline: Trained fraud detection pipeline
            background_data: Background data for SHAP explainer
            max_background_samples: Maximum number of background samples
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for explainability. Install with: pip install shap")
        
        self.pipeline = pipeline
        self.explainer = None
        self.background_data = background_data
        self.max_background_samples = max_background_samples
        self.feature_names = pipeline.data_processor.feature_columns
        
        # Initialize explainer
        self._initialize_explainer()
    
    def _initialize_explainer(self) -> None:
        """Initialize the SHAP explainer."""
        if not self.pipeline.is_trained:
            raise ValueError("Pipeline must be trained before creating explainer")
        
        # Prepare background data
        if self.background_data is not None:
            # Sample background data if too large
            if len(self.background_data) > self.max_background_samples:
                background_sample = self.background_data.sample(
                    n=self.max_background_samples, 
                    random_state=42
                )
            else:
                background_sample = self.background_data
            
            # Process background data
            background_processed = self.pipeline.data_processor.prepare_features(
                background_sample, fit=False
            )
        else:
            # Use a small sample of normal transactions as background
            background_processed = self._create_default_background()
        
        # Create explainer for the ensemble model
        if 'ensemble' in self.pipeline.models:
            # For ensemble model, we need to create a wrapper function
            def ensemble_predict(X):
                # Get base model predictions
                model_predictions = {}
                for name, model in self.pipeline.models.items():
                    if name == 'ensemble':
                        continue
                        
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X)[:, 1]
                    else:
                        anomaly_scores = model.decision_function(X)
                        min_score = anomaly_scores.min()
                        max_score = anomaly_scores.max()
                        if max_score > min_score:
                            pred_proba = (anomaly_scores - min_score) / (max_score - min_score)
                        else:
                            pred_proba = np.zeros_like(anomaly_scores)
                    
                    model_predictions[name] = pred_proba
                
                # Get ensemble predictions
                ensemble_features = pd.DataFrame(model_predictions)
                return self.pipeline.models['ensemble'].predict_proba(ensemble_features)[:, 1]
            
            self.explainer = shap.Explainer(ensemble_predict, background_processed)
        else:
            # Use the main model (random forest as fallback)
            main_model = self.pipeline.models.get('random_forest')
            if main_model is None:
                raise ValueError("No suitable model found for SHAP explanation")
            
            self.explainer = shap.Explainer(main_model, background_processed)
    
    def _create_default_background(self) -> pd.DataFrame:
        """Create default background data for SHAP explainer.
        
        Returns:
            Default background data
        """
        # Create a small sample of normal-looking transactions
        n_samples = min(50, self.max_background_samples)
        
        # Generate synthetic normal transactions
        from ..data.generator import TransactionDataGenerator
        generator = TransactionDataGenerator(random_state=42, fraud_rate=0.0)
        background_data = generator.generate_transactions(n_samples)
        
        # Process the data
        background_processed = self.pipeline.data_processor.prepare_features(
            background_data, fit=False
        )
        
        return background_processed
    
    def explain_prediction(
        self, 
        X: pd.DataFrame, 
        max_display: int = 10
    ) -> Dict[str, Any]:
        """Explain individual predictions using SHAP values.
        
        Args:
            X: Input features for explanation
            max_display: Maximum number of features to display
            
        Returns:
            Dictionary with SHAP values and explanations
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")
        
        # Process input data
        X_processed = self.pipeline.data_processor.prepare_features(X, fit=False)
        
        # Calculate SHAP values
        shap_values = self.explainer(X_processed)
        
        # Get predictions
        predictions = self.pipeline.predict_proba(X)
        
        # Create explanation data
        explanations = []
        
        for i in range(len(X)):
            # Get SHAP values for this prediction
            if hasattr(shap_values, 'values'):
                # For newer SHAP versions
                sample_shap_values = shap_values.values[i]
                base_value = shap_values.base_values[i]
            else:
                # For older SHAP versions
                sample_shap_values = shap_values[i].values
                base_value = shap_values[i].base_values
            
            # Create feature importance data
            feature_importance = []
            for j, feature in enumerate(self.feature_names):
                feature_importance.append({
                    'feature': feature,
                    'shap_value': sample_shap_values[j],
                    'value': X_processed.iloc[i, j]
                })
            
            # Sort by absolute SHAP value
            feature_importance.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            
            # Get top features
            top_features = feature_importance[:max_display]
            
            explanations.append({
                'prediction_id': i,
                'fraud_probability': predictions[i],
                'base_value': base_value,
                'top_features': top_features,
                'all_features': feature_importance
            })
        
        return {
            'explanations': explanations,
            'shap_values': shap_values,
            'feature_names': self.feature_names
        }
    
    def explain_global(
        self, 
        X: pd.DataFrame, 
        max_display: int = 20
    ) -> Dict[str, Any]:
        """Explain global model behavior using SHAP values.
        
        Args:
            X: Input features for global explanation
            max_display: Maximum number of features to display
            
        Returns:
            Dictionary with global SHAP explanations
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")
        
        # Process input data
        X_processed = self.pipeline.data_processor.prepare_features(X, fit=False)
        
        # Calculate SHAP values
        shap_values = self.explainer(X_processed)
        
        # Get global feature importance
        if hasattr(shap_values, 'values'):
            # For newer SHAP versions
            mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
        else:
            # For older SHAP versions
            mean_abs_shap = np.mean(np.abs([sv.values for sv in shap_values]), axis=0)
        
        # Create feature importance ranking
        feature_importance = []
        for i, feature in enumerate(self.feature_names):
            feature_importance.append({
                'feature': feature,
                'mean_abs_shap': mean_abs_shap[i],
                'rank': i + 1
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['mean_abs_shap'], reverse=True)
        
        return {
            'global_importance': feature_importance[:max_display],
            'shap_values': shap_values,
            'feature_names': self.feature_names
        }
    
    def plot_waterfall(
        self, 
        X: pd.DataFrame, 
        sample_idx: int = 0,
        max_display: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """Plot SHAP waterfall for individual prediction.
        
        Args:
            X: Input features
            sample_idx: Index of sample to explain
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")
        
        # Process input data
        X_processed = self.pipeline.data_processor.prepare_features(X, fit=False)
        
        # Calculate SHAP values
        shap_values = self.explainer(X_processed)
        
        # Create waterfall plot
        plt.figure(figsize=(10, 8))
        
        if hasattr(shap, 'waterfall_plot'):
            # Use newer SHAP waterfall plot
            shap.waterfall_plot(shap_values[sample_idx], max_display=max_display)
        else:
            # Fallback to summary plot
            shap.summary_plot(shap_values, X_processed, max_display=max_display)
        
        plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_summary(
        self, 
        X: pd.DataFrame, 
        max_display: int = 20,
        save_path: Optional[str] = None
    ) -> None:
        """Plot SHAP summary plot.
        
        Args:
            X: Input features
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")
        
        # Process input data
        X_processed = self.pipeline.data_processor.prepare_features(X, fit=False)
        
        # Calculate SHAP values
        shap_values = self.explainer(X_processed)
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_processed, max_display=max_display)
        plt.title('SHAP Summary Plot')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(
        self, 
        X: pd.DataFrame, 
        max_display: int = 20,
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature importance based on SHAP values.
        
        Args:
            X: Input features
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")
        
        # Process input data
        X_processed = self.pipeline.data_processor.prepare_features(X, fit=False)
        
        # Calculate SHAP values
        shap_values = self.explainer(X_processed)
        
        # Get mean absolute SHAP values
        if hasattr(shap_values, 'values'):
            mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
        else:
            mean_abs_shap = np.mean(np.abs([sv.values for sv in shap_values]), axis=0)
        
        # Create feature importance plot
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=True).tail(max_display)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Feature Importance (SHAP)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def get_feature_contributions(
        self, 
        X: pd.DataFrame, 
        sample_idx: int = 0
    ) -> pd.DataFrame:
        """Get detailed feature contributions for a specific sample.
        
        Args:
            X: Input features
            sample_idx: Index of sample to analyze
            
        Returns:
            DataFrame with feature contributions
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")
        
        # Process input data
        X_processed = self.pipeline.data_processor.prepare_features(X, fit=False)
        
        # Calculate SHAP values
        shap_values = self.explainer(X_processed)
        
        # Get SHAP values for the specific sample
        if hasattr(shap_values, 'values'):
            sample_shap_values = shap_values.values[sample_idx]
            base_value = shap_values.base_values[sample_idx]
        else:
            sample_shap_values = shap_values[sample_idx].values
            base_value = shap_values[sample_idx].base_values
        
        # Create contributions DataFrame
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_processed.iloc[sample_idx].values,
            'shap_value': sample_shap_values,
            'abs_shap_value': np.abs(sample_shap_values)
        })
        
        # Sort by absolute SHAP value
        contributions = contributions.sort_values('abs_shap_value', ascending=False)
        
        # Add cumulative contribution
        contributions['cumulative_contribution'] = np.cumsum(contributions['shap_value'])
        contributions['cumulative_abs_contribution'] = np.cumsum(contributions['abs_shap_value'])
        
        return contributions
    
    def generate_explanation_report(
        self, 
        X: pd.DataFrame, 
        output_dir: str = "assets/explanations"
    ) -> str:
        """Generate comprehensive explanation report.
        
        Args:
            X: Input features to explain
            output_dir: Directory to save explanation plots
            
        Returns:
            Path to the generated report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate global explanations
        global_explanation = self.explain_global(X)
        
        # Generate individual explanations
        individual_explanations = self.explain_prediction(X)
        
        # Create plots
        self.plot_summary(X, save_path=str(output_path / "shap_summary.png"))
        self.plot_feature_importance(X, save_path=str(output_path / "shap_feature_importance.png"))
        
        # Create detailed explanations for top samples
        for i in range(min(5, len(X))):
            self.plot_waterfall(X, sample_idx=i, save_path=str(output_path / f"shap_waterfall_{i}.png"))
        
        # Generate text report
        report_lines = []
        report_lines.append("# Fraud Detection Model Explainability Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Global feature importance
        report_lines.append("## Global Feature Importance (SHAP)")
        report_lines.append("")
        for i, feature_info in enumerate(global_explanation['global_importance'][:10]):
            report_lines.append(f"{i+1}. **{feature_info['feature']}**: {feature_info['mean_abs_shap']:.4f}")
        report_lines.append("")
        
        # Individual predictions
        report_lines.append("## Individual Prediction Explanations")
        report_lines.append("")
        
        for i, explanation in enumerate(individual_explanations['explanations'][:5]):
            report_lines.append(f"### Prediction {i+1}")
            report_lines.append(f"**Fraud Probability**: {explanation['fraud_probability']:.3f}")
            report_lines.append("")
            report_lines.append("**Top Contributing Features**:")
            report_lines.append("")
            
            for j, feature in enumerate(explanation['top_features'][:5]):
                report_lines.append(f"- **{feature['feature']}**: {feature['shap_value']:.4f} (value: {feature['value']:.4f})")
            
            report_lines.append("")
        
        # Disclaimer
        report_lines.append("## Disclaimer")
        report_lines.append("This explanation is for research and educational purposes only.")
        report_lines.append("It should not be used for automated decision-making without human review.")
        
        # Save report
        report_content = "\n".join(report_lines)
        report_file = output_path / "explanation_report.md"
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        return str(report_file)
