"""Comprehensive evaluator for fraud detection models.

This module provides a comprehensive evaluator that combines multiple
evaluation approaches for fraud detection systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .metrics import FraudDetectionMetrics
from ..models.pipeline import FraudDetectionPipeline


class FraudDetectionEvaluator:
    """Comprehensive evaluator for fraud detection models.
    
    This class provides comprehensive evaluation capabilities for fraud
    detection models, including performance metrics, business impact
    analysis, and visualization.
    
    Attributes:
        pipeline: Trained fraud detection pipeline
        results: Dictionary storing evaluation results
        plots_dir: Directory to save evaluation plots
    """
    
    def __init__(
        self,
        pipeline: FraudDetectionPipeline,
        plots_dir: Optional[str] = None
    ) -> None:
        """Initialize the evaluator.
        
        Args:
            pipeline: Trained fraud detection pipeline
            plots_dir: Directory to save evaluation plots
        """
        self.pipeline = pipeline
        self.results: Dict[str, Any] = {}
        self.plots_dir = Path(plots_dir) if plots_dir else Path("assets/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_test_original: Optional[pd.DataFrame] = None,
        thresholds: Optional[List[float]] = None,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """Perform comprehensive evaluation of the fraud detection pipeline.
        
        Args:
            X_test: Test features
            y_test: Test labels
            X_test_original: Original test features (before processing)
            thresholds: List of thresholds to evaluate
            save_plots: Whether to save evaluation plots
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        print("Starting comprehensive evaluation...")
        
        # Get predictions
        y_pred, y_pred_proba = self.pipeline.predict(X_test, return_probabilities=True)
        
        # Store original test data if provided
        if X_test_original is not None:
            X_test_eval = X_test_original.copy()
        else:
            X_test_eval = X_test.copy()
        
        # Add predictions to test data
        X_test_eval['y_true'] = y_test
        X_test_eval['y_pred'] = y_pred
        X_test_eval['y_pred_proba'] = y_pred_proba
        
        # Evaluate at default threshold
        metrics = FraudDetectionMetrics(y_test, y_pred, y_pred_proba)
        
        # Get all metrics
        self.results['ml_metrics'] = metrics.get_ml_metrics()
        self.results['business_metrics'] = metrics.get_business_metrics()
        self.results['calibration_metrics'] = metrics.get_calibration_metrics()
        self.results['classification_report'] = metrics.get_classification_report()
        
        # Threshold analysis
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        self.results['threshold_analysis'] = metrics.get_threshold_analysis(thresholds)
        
        # Curve data
        self.results['precision_recall_curve'] = metrics.get_precision_recall_curve()
        self.results['roc_curve'] = metrics.get_roc_curve()
        
        # Feature importance
        self.results['feature_importance'] = self.pipeline.get_feature_importance()
        
        # Model performance comparison
        self.results['model_performance'] = self.pipeline.get_model_performance()
        
        # Generate visualizations
        if save_plots:
            self._generate_plots(X_test_eval)
        
        print("Evaluation completed!")
        return self.results
    
    def _generate_plots(self, X_test_eval: pd.DataFrame) -> None:
        """Generate comprehensive evaluation plots.
        
        Args:
            X_test_eval: Test data with predictions
        """
        print("Generating evaluation plots...")
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix()
        
        # 2. ROC Curve
        self._plot_roc_curve()
        
        # 3. Precision-Recall Curve
        self._plot_precision_recall_curve()
        
        # 4. Threshold Analysis
        self._plot_threshold_analysis()
        
        # 5. Feature Importance
        self._plot_feature_importance()
        
        # 6. Model Performance Comparison
        self._plot_model_performance()
        
        # 7. Calibration Plot
        self._plot_calibration()
        
        # 8. Fraud Distribution Analysis
        self._plot_fraud_distribution(X_test_eval)
        
        print(f"Plots saved to {self.plots_dir}")
    
    def _plot_confusion_matrix(self) -> None:
        """Plot confusion matrix."""
        cm = np.array([
            [self.results['ml_metrics']['specificity'] * (1 - self.results['ml_metrics']['fpr']),
             self.results['ml_metrics']['fpr']],
            [self.results['ml_metrics']['fnr'],
             self.results['ml_metrics']['sensitivity']]
        ])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=['Predicted Normal', 'Predicted Fraud'],
                   yticklabels=['Actual Normal', 'Actual Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self) -> None:
        """Plot ROC curve."""
        roc_data = self.results['roc_curve']
        
        plt.figure(figsize=(8, 6))
        plt.plot(roc_data['fpr'], roc_data['tpr'], 
                label=f'ROC Curve (AUC = {self.results["ml_metrics"]["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self) -> None:
        """Plot precision-recall curve."""
        pr_data = self.results['precision_recall_curve']
        
        plt.figure(figsize=(8, 6))
        plt.plot(pr_data['recall'], pr_data['precision'],
                label=f'PR Curve (AUC = {self.results["ml_metrics"]["pr_auc"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_threshold_analysis(self) -> None:
        """Plot threshold analysis."""
        threshold_data = self.results['threshold_analysis']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Precision vs Threshold
        axes[0, 0].plot(threshold_data['threshold'], threshold_data['precision'], 'b-o')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_title('Precision vs Threshold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recall vs Threshold
        axes[0, 1].plot(threshold_data['threshold'], threshold_data['recall'], 'r-o')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Recall vs Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score vs Threshold
        axes[1, 0].plot(threshold_data['threshold'], threshold_data['f1_score'], 'g-o')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score vs Threshold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Specificity vs Threshold
        axes[1, 1].plot(threshold_data['threshold'], threshold_data['specificity'], 'm-o')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Specificity')
        axes[1, 1].set_title('Specificity vs Threshold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self) -> None:
        """Plot feature importance."""
        if self.results['feature_importance'].empty:
            return
        
        # Get top 15 features
        top_features = self.results['feature_importance'].head(15)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_features, x='importance', y='feature', hue='model')
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Top 15 Feature Importance by Model')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_performance(self) -> None:
        """Plot model performance comparison."""
        if self.results['model_performance'].empty:
            return
        
        perf_data = self.results['model_performance']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # AUC comparison
        axes[0, 0].bar(perf_data['model'], perf_data['auc'])
        axes[0, 0].set_ylabel('AUC')
        axes[0, 0].set_title('Model AUC Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Precision comparison
        axes[0, 1].bar(perf_data['model'], perf_data['precision'])
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Model Precision Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Recall comparison
        axes[1, 0].bar(perf_data['model'], perf_data['recall'])
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Model Recall Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        axes[1, 1].bar(perf_data['model'], perf_data['f1'])
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Model F1 Score Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration(self) -> None:
        """Plot calibration curve."""
        cal_data = self.results['calibration_metrics']['calibration_curve']
        
        if cal_data is None:
            return
        
        plt.figure(figsize=(8, 6))
        plt.plot(cal_data['mean_predicted_value'], cal_data['fraction_of_positives'],
                'b-', label='Model Calibration')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Curve (ECE = {self.results["calibration_metrics"]["ece"]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'calibration_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_fraud_distribution(self, X_test_eval: pd.DataFrame) -> None:
        """Plot fraud distribution analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Fraud by amount
        if 'amount' in X_test_eval.columns:
            fraud_amounts = X_test_eval[X_test_eval['y_true'] == 1]['amount']
            normal_amounts = X_test_eval[X_test_eval['y_true'] == 0]['amount']
            
            axes[0, 0].hist([normal_amounts, fraud_amounts], bins=50, alpha=0.7,
                           label=['Normal', 'Fraud'], density=True)
            axes[0, 0].set_xlabel('Transaction Amount')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('Transaction Amount Distribution')
            axes[0, 0].legend()
            axes[0, 0].set_yscale('log')
        
        # Fraud by time of day
        if 'hour' in X_test_eval.columns:
            fraud_hours = X_test_eval[X_test_eval['y_true'] == 1]['hour']
            normal_hours = X_test_eval[X_test_eval['y_true'] == 0]['hour']
            
            axes[0, 1].hist([normal_hours, fraud_hours], bins=24, alpha=0.7,
                           label=['Normal', 'Fraud'], density=True)
            axes[0, 1].set_xlabel('Hour of Day')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Fraud Distribution by Hour')
            axes[0, 1].legend()
        
        # Prediction probability distribution
        fraud_probs = X_test_eval[X_test_eval['y_true'] == 1]['y_pred_proba']
        normal_probs = X_test_eval[X_test_eval['y_true'] == 0]['y_pred_proba']
        
        axes[1, 0].hist([normal_probs, fraud_probs], bins=50, alpha=0.7,
                       label=['Normal', 'Fraud'], density=True)
        axes[1, 0].set_xlabel('Predicted Fraud Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Prediction Probability Distribution')
        axes[1, 0].legend()
        
        # Risk score distribution
        if 'risk_score' in X_test_eval.columns:
            fraud_risk = X_test_eval[X_test_eval['y_true'] == 1]['risk_score']
            normal_risk = X_test_eval[X_test_eval['y_true'] == 0]['risk_score']
            
            axes[1, 1].hist([normal_risk, fraud_risk], bins=20, alpha=0.7,
                           label=['Normal', 'Fraud'], density=True)
            axes[1, 1].set_xlabel('Risk Score')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Risk Score Distribution')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'fraud_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get model performance leaderboard.
        
        Returns:
            DataFrame with model performance ranking
        """
        if self.results['model_performance'].empty:
            return pd.DataFrame()
        
        # Create leaderboard with multiple metrics
        leaderboard = self.results['model_performance'].copy()
        
        # Calculate composite score (weighted average)
        weights = {'auc': 0.3, 'precision': 0.25, 'recall': 0.25, 'f1': 0.2}
        leaderboard['composite_score'] = sum(
            leaderboard[metric] * weight for metric, weight in weights.items()
        )
        
        # Sort by composite score
        leaderboard = leaderboard.sort_values('composite_score', ascending=False)
        leaderboard['rank'] = range(1, len(leaderboard) + 1)
        
        return leaderboard
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Report content as string
        """
        report = []
        report.append("# Fraud Detection Model Evaluation Report")
        report.append("=" * 50)
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        ml_metrics = self.results['ml_metrics']
        business_metrics = self.results['business_metrics']
        
        report.append(f"- **Overall Performance**: AUC = {ml_metrics['roc_auc']:.3f}")
        report.append(f"- **Fraud Detection Rate**: {business_metrics['fraud_detection_rate']:.1%}")
        report.append(f"- **Alert Rate**: {business_metrics['alert_rate']:.1%}")
        report.append(f"- **Precision at K**: {business_metrics['precision_at_k']:.1%}")
        report.append("")
        
        # Model Performance
        report.append("## Model Performance")
        report.append("")
        leaderboard = self.get_leaderboard()
        if not leaderboard.empty:
            report.append(leaderboard.to_string(index=False))
        report.append("")
        
        # Business Impact
        report.append("## Business Impact")
        report.append("")
        report.append(f"- **Investigation Cost**: ${business_metrics['investigation_cost']:,.2f}")
        report.append(f"- **Net Savings**: ${business_metrics['net_savings']:,.2f}")
        report.append(f"- **ROI**: {business_metrics['roi']:.1%}")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        if ml_metrics['precision'] < 0.5:
            report.append("- **High False Positive Rate**: Consider adjusting threshold or improving feature engineering")
        if ml_metrics['recall'] < 0.7:
            report.append("- **Low Fraud Detection Rate**: Consider ensemble methods or additional data sources")
        if business_metrics['roi'] < 0:
            report.append("- **Negative ROI**: Review investigation costs and fraud prevention value")
        
        report.append("")
        report.append("## Disclaimer")
        report.append("This system is for research and educational purposes only.")
        report.append("It should not be used for automated decision-making without human review.")
        
        report_content = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
        
        return report_content
