"""Fraud detection metrics and evaluation utilities.

This module provides comprehensive metrics for evaluating fraud detection
models, including both ML metrics and business-relevant KPIs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    average_precision_score, roc_curve
)
from sklearn.calibration import calibration_curve
import warnings


class FraudDetectionMetrics:
    """Comprehensive metrics for fraud detection evaluation.
    
    This class provides both standard ML metrics and business-relevant
    metrics for fraud detection systems.
    
    Attributes:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold used
    """
    
    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> None:
        """Initialize the metrics calculator.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold used
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.threshold = threshold
        
        # Calculate confusion matrix
        self.confusion_matrix = confusion_matrix(y_true, y_pred)
        self.tn, self.fp, self.fn, self.tp = self.confusion_matrix.ravel()
    
    def get_ml_metrics(self) -> Dict[str, float]:
        """Get standard machine learning metrics.
        
        Returns:
            Dictionary of ML metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        metrics['precision'] = precision_score(self.y_true, self.y_pred, zero_division=0)
        metrics['recall'] = recall_score(self.y_true, self.y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(self.y_true, self.y_pred, zero_division=0)
        
        # AUC metrics
        try:
            metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_pred_proba)
        except ValueError:
            metrics['roc_auc'] = 0.5  # Default for single class
        
        try:
            metrics['pr_auc'] = average_precision_score(self.y_true, self.y_pred_proba)
        except ValueError:
            metrics['pr_auc'] = 0.0
        
        # Specificity and sensitivity
        metrics['specificity'] = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0
        metrics['sensitivity'] = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        
        # False positive and false negative rates
        metrics['fpr'] = self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0
        metrics['fnr'] = self.fn / (self.fn + self.tp) if (self.fn + self.tp) > 0 else 0
        
        # Positive and negative predictive values
        metrics['ppv'] = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        metrics['npv'] = self.tn / (self.tn + self.fn) if (self.tn + self.fn) > 0 else 0
        
        return metrics
    
    def get_business_metrics(
        self,
        transaction_amounts: Optional[np.ndarray] = None,
        investigation_cost: float = 10.0,
        fraud_loss_per_transaction: float = 100.0
    ) -> Dict[str, float]:
        """Get business-relevant metrics for fraud detection.
        
        Args:
            transaction_amounts: Array of transaction amounts
            investigation_cost: Cost to investigate a flagged transaction
            fraud_loss_per_transaction: Average loss per fraudulent transaction
            
        Returns:
            Dictionary of business metrics
        """
        metrics = {}
        
        # Alert workload metrics
        total_transactions = len(self.y_true)
        total_alerts = self.tp + self.fp
        metrics['alert_rate'] = total_alerts / total_transactions if total_transactions > 0 else 0
        metrics['precision_at_k'] = self.tp / total_alerts if total_alerts > 0 else 0
        
        # Fraud detection metrics
        total_fraud = self.tp + self.fn
        metrics['fraud_detection_rate'] = self.tp / total_fraud if total_fraud > 0 else 0
        metrics['fraud_miss_rate'] = self.fn / total_fraud if total_fraud > 0 else 0
        
        # Cost-based metrics
        if transaction_amounts is not None:
            # Calculate financial impact
            detected_fraud_amount = transaction_amounts[self.y_true == 1][self.y_pred[self.y_true == 1] == 1].sum()
            missed_fraud_amount = transaction_amounts[self.y_true == 1][self.y_pred[self.y_true == 1] == 0].sum()
            
            metrics['detected_fraud_amount'] = detected_fraud_amount
            metrics['missed_fraud_amount'] = missed_fraud_amount
            metrics['total_fraud_amount'] = transaction_amounts[self.y_true == 1].sum()
            metrics['fraud_recovery_rate'] = detected_fraud_amount / metrics['total_fraud_amount'] if metrics['total_fraud_amount'] > 0 else 0
        
        # Investigation costs
        investigation_cost_total = total_alerts * investigation_cost
        metrics['investigation_cost'] = investigation_cost_total
        metrics['cost_per_detected_fraud'] = investigation_cost_total / self.tp if self.tp > 0 else float('inf')
        
        # Expected cost savings
        if transaction_amounts is not None:
            fraud_prevention_value = detected_fraud_amount
            net_savings = fraud_prevention_value - investigation_cost_total
            metrics['net_savings'] = net_savings
            metrics['roi'] = net_savings / investigation_cost_total if investigation_cost_total > 0 else 0
        else:
            # Use average fraud loss if transaction amounts not available
            fraud_prevention_value = self.tp * fraud_loss_per_transaction
            net_savings = fraud_prevention_value - investigation_cost_total
            metrics['net_savings'] = net_savings
            metrics['roi'] = net_savings / investigation_cost_total if investigation_cost_total > 0 else 0
        
        return metrics
    
    def get_calibration_metrics(self, n_bins: int = 10) -> Dict[str, Any]:
        """Get calibration metrics for probability predictions.
        
        Args:
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary of calibration metrics
        """
        try:
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                self.y_true, self.y_pred_proba, n_bins=n_bins
            )
            
            # Calculate Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (self.y_pred_proba > bin_lower) & (self.y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = self.y_true[in_bin].mean()
                    avg_confidence_in_bin = self.y_pred_proba[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            # Calculate Brier score
            brier_score = np.mean((self.y_pred_proba - self.y_true) ** 2)
            
            return {
                'ece': ece,
                'brier_score': brier_score,
                'calibration_curve': {
                    'fraction_of_positives': fraction_of_positives,
                    'mean_predicted_value': mean_predicted_value
                }
            }
            
        except Exception as e:
            warnings.warn(f"Could not calculate calibration metrics: {e}")
            return {
                'ece': float('nan'),
                'brier_score': float('nan'),
                'calibration_curve': None
            }
    
    def get_threshold_analysis(
        self, 
        thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """Analyze performance at different thresholds.
        
        Args:
            thresholds: List of thresholds to analyze (default: 0.1 to 0.9)
            
        Returns:
            DataFrame with metrics at each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        results = []
        
        for threshold in thresholds:
            y_pred_thresh = (self.y_pred_proba >= threshold).astype(int)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred_thresh).ravel()
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            })
        
        return pd.DataFrame(results)
    
    def get_precision_recall_curve(self) -> Dict[str, np.ndarray]:
        """Get precision-recall curve data.
        
        Returns:
            Dictionary with precision, recall, and thresholds
        """
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_proba)
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds
        }
    
    def get_roc_curve(self) -> Dict[str, np.ndarray]:
        """Get ROC curve data.
        
        Returns:
            Dictionary with fpr, tpr, and thresholds
        """
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    
    def get_classification_report(self) -> str:
        """Get detailed classification report.
        
        Returns:
            String classification report
        """
        return classification_report(self.y_true, self.y_pred, zero_division=0)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all metrics.
        
        Returns:
            Dictionary with all calculated metrics
        """
        summary = {
            'ml_metrics': self.get_ml_metrics(),
            'business_metrics': self.get_business_metrics(),
            'calibration_metrics': self.get_calibration_metrics(),
            'confusion_matrix': {
                'tn': int(self.tn),
                'fp': int(self.fp),
                'fn': int(self.fn),
                'tp': int(self.tp)
            },
            'threshold_used': self.threshold
        }
        
        return summary
