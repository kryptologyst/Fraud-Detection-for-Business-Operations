#!/usr/bin/env python3
"""Training script for fraud detection system.

This script provides a command-line interface for training fraud detection
models with various configurations.
"""

import argparse
import sys
from pathlib import Path
import yaml
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fraud_detection import (
    TransactionDataGenerator,
    FraudDetectionPipeline,
    FraudDetectionEvaluator
)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-size", 
        type=int, 
        default=None,
        help="Number of transactions to generate (overrides config)"
    )
    parser.add_argument(
        "--fraud-rate", 
        type=float, 
        default=None,
        help="Fraud rate (overrides config)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models",
        help="Output directory for trained models"
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
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_size:
        config['data']['n_transactions'] = args.data_size
    if args.fraud_rate:
        config['data']['fraud_rate'] = args.fraud_rate
    
    logger.info("Starting fraud detection model training")
    logger.info(f"Configuration: {config}")
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate data
        logger.info("Generating transaction data...")
        generator = TransactionDataGenerator(
            random_state=config['data']['random_state'],
            fraud_rate=config['data']['fraud_rate']
        )
        
        data = generator.generate_transactions(
            n_transactions=config['data']['n_transactions']
        )
        
        logger.info(f"Generated {len(data)} transactions with {data['is_fraud'].mean():.1%} fraud rate")
        
        # Prepare data
        from fraud_detection.data.processor import DataProcessor
        
        processor = DataProcessor()
        X = data.drop(['is_fraud', 'transaction_id', 'customer_id', 'timestamp'], axis=1, errors='ignore')
        y = data['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = processor.split_data(
            processor.prepare_features(pd.concat([X, y], axis=1), fit=True)
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train pipeline
        logger.info("Training fraud detection pipeline...")
        pipeline = FraudDetectionPipeline(
            model_configs=config['models'],
            random_state=config['data']['random_state']
        )
        
        pipeline.fit(X_train, y_train, X_test, y_test)
        
        # Evaluate model
        logger.info("Evaluating model performance...")
        evaluator = FraudDetectionEvaluator(pipeline)
        results = evaluator.evaluate(X_test, y_test, X_test)
        
        # Print results
        ml_metrics = results['ml_metrics']
        business_metrics = results['business_metrics']
        
        logger.info("Model Performance:")
        logger.info(f"  AUC: {ml_metrics['roc_auc']:.3f}")
        logger.info(f"  Precision: {ml_metrics['precision']:.3f}")
        logger.info(f"  Recall: {ml_metrics['recall']:.3f}")
        logger.info(f"  F1 Score: {ml_metrics['f1_score']:.3f}")
        
        logger.info("Business Impact:")
        logger.info(f"  Fraud Detection Rate: {business_metrics['fraud_detection_rate']:.1%}")
        logger.info(f"  Alert Rate: {business_metrics['alert_rate']:.1%}")
        logger.info(f"  ROI: {business_metrics['roi']:.1%}")
        
        # Save model
        model_path = output_dir / f"fraud_detection_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        pipeline.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Save evaluation results
        results_path = output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        logger.info(f"Evaluation results saved to {results_path}")
        
        # Generate report
        report_path = output_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_content = evaluator.generate_report(str(report_path))
        logger.info(f"Training report saved to {report_path}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
