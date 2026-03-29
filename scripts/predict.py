#!/usr/bin/env python3
"""Prediction script for fraud detection system.

This script provides a command-line interface for making fraud predictions
using trained models.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fraud_detection import FraudDetectionPipeline


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Make fraud predictions")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to trained model file"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to input CSV file with transaction data"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="predictions.csv",
        help="Path to output CSV file with predictions"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5,
        help="Classification threshold"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load model
        logger.info(f"Loading model from {args.model}")
        pipeline = FraudDetectionPipeline()
        pipeline.load_model(args.model)
        
        # Load input data
        logger.info(f"Loading input data from {args.input}")
        data = pd.read_csv(args.input)
        
        # Prepare features
        X = data.drop(['is_fraud', 'transaction_id', 'customer_id', 'timestamp'], axis=1, errors='ignore')
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = pipeline.predict(X, threshold=args.threshold)
        probabilities = pipeline.predict_proba(X)
        
        # Create results DataFrame
        results = data.copy()
        results['predicted_fraud'] = predictions
        results['fraud_probability'] = probabilities
        
        # Add risk level
        results['risk_level'] = pd.cut(
            probabilities, 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
        
        # Save results
        logger.info(f"Saving predictions to {args.output}")
        results.to_csv(args.output, index=False)
        
        # Print summary
        logger.info("Prediction Summary:")
        logger.info(f"  Total transactions: {len(results)}")
        logger.info(f"  Predicted fraud: {predictions.sum()}")
        logger.info(f"  Fraud rate: {predictions.mean():.1%}")
        logger.info(f"  Average fraud probability: {probabilities.mean():.3f}")
        
        # Risk level distribution
        risk_dist = results['risk_level'].value_counts()
        logger.info("Risk Level Distribution:")
        for level, count in risk_dist.items():
            logger.info(f"  {level}: {count} ({count/len(results):.1%})")
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()
