"""Additional plotting utilities for fraud detection.

This module provides additional plotting utilities for fraud detection
visualization and analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class FraudDetectionPlots:
    """Additional plotting utilities for fraud detection.
    
    This class provides specialized plotting functions for fraud detection
    analysis and visualization.
    
    Attributes:
        style: Plotting style configuration
        color_palette: Color palette for plots
    """
    
    def __init__(self, style: str = 'seaborn-v0_8') -> None:
        """Initialize the plotting utilities.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.color_palette = sns.color_palette("husl")
        
        # Set default figure size
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_transaction_timeline(
        self, 
        df: pd.DataFrame, 
        time_col: str = 'timestamp',
        fraud_col: str = 'is_fraud',
        amount_col: str = 'amount',
        save_path: Optional[str] = None
    ) -> None:
        """Plot transaction timeline with fraud indicators.
        
        Args:
            df: DataFrame with transaction data
            time_col: Name of timestamp column
            fraud_col: Name of fraud indicator column
            amount_col: Name of amount column
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # Sort by timestamp
        df_sorted = df.sort_values(time_col)
        
        # Plot 1: Transaction amounts over time
        fraud_transactions = df_sorted[df_sorted[fraud_col] == 1]
        normal_transactions = df_sorted[df_sorted[fraud_col] == 0]
        
        axes[0].scatter(normal_transactions[time_col], normal_transactions[amount_col], 
                       alpha=0.6, s=20, label='Normal', color='blue')
        axes[0].scatter(fraud_transactions[time_col], fraud_transactions[amount_col], 
                       alpha=0.8, s=30, label='Fraud', color='red')
        
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Transaction Amount')
        axes[0].set_title('Transaction Amounts Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Fraud rate over time (rolling window)
        df_sorted['fraud_rate'] = df_sorted[fraud_col].rolling(window=100, center=True).mean()
        
        axes[1].plot(df_sorted[time_col], df_sorted['fraud_rate'], 
                    color='red', linewidth=2, label='Fraud Rate (100-transaction window)')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Fraud Rate')
        axes[1].set_title('Fraud Rate Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_fraud_by_category(
        self, 
        df: pd.DataFrame, 
        category_col: str = 'merchant_category',
        fraud_col: str = 'is_fraud',
        save_path: Optional[str] = None
    ) -> None:
        """Plot fraud distribution by category.
        
        Args:
            df: DataFrame with transaction data
            category_col: Name of category column
            fraud_col: Name of fraud indicator column
            save_path: Path to save the plot
        """
        # Calculate fraud rates by category
        category_stats = df.groupby(category_col).agg({
            fraud_col: ['count', 'sum', 'mean']
        }).round(3)
        
        category_stats.columns = ['total_transactions', 'fraud_count', 'fraud_rate']
        category_stats = category_stats.sort_values('fraud_rate', ascending=True)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Fraud rate by category
        bars1 = axes[0].barh(category_stats.index, category_stats['fraud_rate'])
        axes[0].set_xlabel('Fraud Rate')
        axes[0].set_title('Fraud Rate by Category')
        axes[0].grid(True, alpha=0.3)
        
        # Color bars by fraud rate
        for i, bar in enumerate(bars1):
            bar.set_color(plt.cm.Reds(category_stats['fraud_rate'].iloc[i] / category_stats['fraud_rate'].max()))
        
        # Plot 2: Total transactions vs fraud count
        scatter = axes[1].scatter(category_stats['total_transactions'], 
                                 category_stats['fraud_count'],
                                 s=100, alpha=0.7, 
                                 c=category_stats['fraud_rate'], 
                                 cmap='Reds')
        
        axes[1].set_xlabel('Total Transactions')
        axes[1].set_ylabel('Fraud Count')
        axes[1].set_title('Total Transactions vs Fraud Count')
        axes[1].grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[1])
        cbar.set_label('Fraud Rate')
        
        # Add category labels
        for i, category in enumerate(category_stats.index):
            axes[1].annotate(category, 
                           (category_stats['total_transactions'].iloc[i], 
                            category_stats['fraud_count'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_risk_score_distribution(
        self, 
        df: pd.DataFrame, 
        risk_col: str = 'risk_score',
        fraud_col: str = 'is_fraud',
        save_path: Optional[str] = None
    ) -> None:
        """Plot risk score distribution by fraud status.
        
        Args:
            df: DataFrame with risk scores
            risk_col: Name of risk score column
            fraud_col: Name of fraud indicator column
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Histogram
        fraud_scores = df[df[fraud_col] == 1][risk_col]
        normal_scores = df[df[fraud_col] == 0][risk_col]
        
        axes[0].hist([normal_scores, fraud_scores], bins=30, alpha=0.7, 
                    label=['Normal', 'Fraud'], density=True, color=['blue', 'red'])
        axes[0].set_xlabel('Risk Score')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Risk Score Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Box plot
        data_for_box = [normal_scores, fraud_scores]
        box_plot = axes[1].boxplot(data_for_box, labels=['Normal', 'Fraud'], patch_artist=True)
        
        # Color the boxes
        box_plot['boxes'][0].set_facecolor('blue')
        box_plot['boxes'][1].set_facecolor('red')
        
        axes[1].set_ylabel('Risk Score')
        axes[1].set_title('Risk Score Distribution (Box Plot)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_interactive_dashboard(
        self, 
        df: pd.DataFrame, 
        save_path: Optional[str] = None
    ) -> None:
        """Create interactive dashboard using Plotly.
        
        Args:
            df: DataFrame with transaction data
            save_path: Path to save the HTML file
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Transaction Amounts Over Time', 'Fraud Rate by Category',
                          'Risk Score Distribution', 'Fraud by Hour of Day'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Transaction amounts over time
        fraud_mask = df['is_fraud'] == 1
        normal_mask = df['is_fraud'] == 0
        
        fig.add_trace(
            go.Scatter(x=df[normal_mask]['timestamp'], y=df[normal_mask]['amount'],
                      mode='markers', name='Normal', marker=dict(color='blue', size=4)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df[fraud_mask]['timestamp'], y=df[fraud_mask]['amount'],
                      mode='markers', name='Fraud', marker=dict(color='red', size=6)),
            row=1, col=1
        )
        
        # Plot 2: Fraud rate by category
        category_stats = df.groupby('merchant_category')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        category_stats.columns = ['category', 'total', 'fraud_count', 'fraud_rate']
        category_stats = category_stats.sort_values('fraud_rate', ascending=True)
        
        fig.add_trace(
            go.Bar(x=category_stats['fraud_rate'], y=category_stats['category'],
                   orientation='h', name='Fraud Rate', marker=dict(color='red')),
            row=1, col=2
        )
        
        # Plot 3: Risk score distribution
        fig.add_trace(
            go.Histogram(x=df[normal_mask]['risk_score'], name='Normal', 
                        opacity=0.7, marker=dict(color='blue')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=df[fraud_mask]['risk_score'], name='Fraud', 
                        opacity=0.7, marker=dict(color='red')),
            row=2, col=1
        )
        
        # Plot 4: Fraud by hour of day
        if 'hour' in df.columns:
            hour_stats = df.groupby('hour')['is_fraud'].mean().reset_index()
            
            fig.add_trace(
                go.Scatter(x=hour_stats['hour'], y=hour_stats['is_fraud'],
                          mode='lines+markers', name='Fraud Rate by Hour',
                          line=dict(color='red', width=2)),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Fraud Detection Dashboard",
            showlegend=True,
            height=800,
            width=1200
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Amount", row=1, col=1)
        fig.update_xaxes(title_text="Fraud Rate", row=1, col=2)
        fig.update_yaxes(title_text="Category", row=1, col=2)
        fig.update_xaxes(title_text="Risk Score", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Hour", row=2, col=2)
        fig.update_yaxes(title_text="Fraud Rate", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def plot_correlation_heatmap(
        self, 
        df: pd.DataFrame, 
        numeric_cols: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot correlation heatmap for numeric features.
        
        Args:
            df: DataFrame with numeric features
            numeric_cols: List of numeric columns to include
            save_path: Path to save the plot
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_model_comparison(
        self, 
        results: Dict[str, Dict[str, float]], 
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot model performance comparison.
        
        Args:
            results: Dictionary with model results
            metrics: List of metrics to compare
            save_path: Path to save the plot
        """
        if metrics is None:
            metrics = ['auc', 'precision', 'recall', 'f1']
        
        # Prepare data
        model_names = list(results.keys())
        metric_data = {metric: [results[model].get(metric, 0) for model in model_names] 
                      for metric in metrics}
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                bars = axes[i].bar(model_names, metric_data[metric])
                axes[i].set_title(f'{metric.upper()} Comparison')
                axes[i].set_ylabel(metric.upper())
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, metric_data[metric]):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_cost_benefit_analysis(
        self, 
        threshold_analysis: pd.DataFrame,
        investigation_cost: float = 10.0,
        fraud_loss: float = 100.0,
        save_path: Optional[str] = None
    ) -> None:
        """Plot cost-benefit analysis for different thresholds.
        
        Args:
            threshold_analysis: DataFrame with threshold analysis results
            investigation_cost: Cost per investigation
            fraud_loss: Loss per missed fraud
            save_path: Path to save the plot
        """
        # Calculate costs
        threshold_analysis['investigation_cost'] = (threshold_analysis['tp'] + threshold_analysis['fp']) * investigation_cost
        threshold_analysis['fraud_loss'] = threshold_analysis['fn'] * fraud_loss
        threshold_analysis['total_cost'] = threshold_analysis['investigation_cost'] + threshold_analysis['fraud_loss']
        threshold_analysis['net_benefit'] = (threshold_analysis['tp'] * fraud_loss) - threshold_analysis['total_cost']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Total cost vs threshold
        axes[0, 0].plot(threshold_analysis['threshold'], threshold_analysis['total_cost'], 'b-o')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Total Cost')
        axes[0, 0].set_title('Total Cost vs Threshold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Net benefit vs threshold
        axes[0, 1].plot(threshold_analysis['threshold'], threshold_analysis['net_benefit'], 'g-o')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Net Benefit')
        axes[0, 1].set_title('Net Benefit vs Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cost breakdown
        axes[1, 0].plot(threshold_analysis['threshold'], threshold_analysis['investigation_cost'], 
                       'r-o', label='Investigation Cost')
        axes[1, 0].plot(threshold_analysis['threshold'], threshold_analysis['fraud_loss'], 
                       'b-o', label='Fraud Loss')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Cost')
        axes[1, 0].set_title('Cost Breakdown')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: ROI
        threshold_analysis['roi'] = threshold_analysis['net_benefit'] / threshold_analysis['investigation_cost']
        axes[1, 1].plot(threshold_analysis['threshold'], threshold_analysis['roi'], 'm-o')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('ROI')
        axes[1, 1].set_title('Return on Investment vs Threshold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
