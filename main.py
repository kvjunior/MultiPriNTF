"""
MultiPriNTF Main Entry Point
======================

This script serves as the main entry point for the MultiPriNTF system,
coordinating data processing, model training, and evaluation.
"""

import argparse
import logging
import torch
import yaml
from pathlib import Path
from datetime import datetime

from src.models.multimodal_architecture import create_optimized_model
from src.utils.data_loader import create_data_loaders
from src.preprocessing import create_preprocessor
from trainer import create_trainer
from src.models.security_module import create_secure_model
from src.models.evaluation import create_evaluator

def setup_logging(output_dir: Path):
    """Configure logging system."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'MultiPriNTF.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_system(config: dict):
    """Setup system based on configuration."""
    # Set random seeds for reproducibility
    torch.manual_seed(config['system']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['system']['seed'])
        
    # Configure CUDA settings
    if config['system']['device'] == 'cuda':
        torch.backends.cudnn.benchmark = config['system']['cudnn_benchmark']
        torch.backends.cudnn.deterministic = config['system']['cuda_deterministic']

def main(args):
    """Main execution function."""
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(config['output']['save_path']) / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting MultiPriNTF system - Experiment: {args.experiment}")
    
    # Setup system
    setup_system(config)
    
    try:
        # Preprocess data if needed
        if args.preprocess:
            logger.info("Starting data preprocessing...")
            preprocessor = create_preprocessor(
                image_dir=config['data']['image']['dir'],
                transaction_file=config['data']['transaction']['file'],
                output_dir=config['data']['cache_dir'],
                num_gpus=config['system']['num_gpus']
            )
            preprocessor.process_all()
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_data_loaders(
            image_dir=config['data']['image']['dir'],
            transaction_file=config['data']['transaction']['file'],
            batch_size=config['data']['dataloader']['batch_size'],
            num_gpus=config['system']['num_gpus']
        )
        
        # Create model
        logger.info("Creating model...")
        model = create_optimized_model(
            visual_dim=config['model']['visual_encoder']['input_dim'],
            transaction_dim=config['model']['transaction_encoder']['input_dim'],
            fusion_dim=config['model']['fusion']['dim'],
            num_gpus=config['system']['num_gpus']
        )
        
        # Apply security wrapper if enabled
        if config['security']['encryption']['enabled']:
            logger.info("Applying security wrapper...")
            model = create_secure_model(
                model,
                epsilon=config['security']['privacy']['epsilon'],
                delta=config['security']['privacy']['delta']
            )
        
        # Create trainer
        logger.info("Setting up trainer...")
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config_path=args.config,
            experiment_name=args.experiment
        )
        
        # Training
        if not args.evaluate_only:
            logger.info("Starting training...")
            trainer.train()
        
        # Evaluation
        if args.evaluate or args.evaluate_only:
            logger.info("Starting evaluation...")
            evaluator = create_evaluator(
                model=model,
                val_loader=val_loader,
                num_gpus=config['system']['num_gpus'],
                output_dir=output_dir / 'evaluation'
            )
            metrics = evaluator.evaluate_model()
            
            # Log evaluation results
            logger.info("Evaluation Results:")
            for metric_type, values in metrics.items():
                logger.info(f"{metric_type}: {values}")
        
        logger.info("MultiPriNTF execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise
        
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MultiPriNTF System')
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default=f'MultiPriNTF{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        help='Experiment name'
    )
    
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Run data preprocessing'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run evaluation after training'
    )
    
    parser.add_argument(
        '--evaluate_only',
        action='store_true',
        help='Run only evaluation (no training)'
    )
    
    args = parser.parse_args()
    
    main(args)