"""Fraud Detection for Business Operations.

A comprehensive fraud detection system for identifying anomalous transactions
and behaviors in business operations.

DISCLAIMER: This system is for research and educational purposes only.
It should not be used for automated decision-making without human review.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"

from .data import TransactionDataGenerator
from .models import FraudDetectionPipeline
from .eval import FraudDetectionEvaluator

__all__ = [
    "TransactionDataGenerator",
    "FraudDetectionPipeline", 
    "FraudDetectionEvaluator",
]
