"""
MultiPriNTF Training System
====================

This module implements an advanced training system for the MultiPriNTF architecture,
optimized for multi-GPU training with comprehensive monitoring and error handling.

Key Features:
1. Distributed training support
2. Advanced metrics tracking
3. Dynamic learning rate adjustment
4. Resource optimization
5. Checkpoint management
6. Comprehensive logging
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import yaml
import logging
from pathlib import Path
import numpy as np
from datetime import datetime
import time
import json
from tqdm import tqdm
import h5py
import wandb
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

class OptimizedTrainer:
    """
    Advanced training system optimized for multi-GPU NFT market analysis.
    Implements comprehensive training features with resource optimization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config_path: str,
        experiment_name: str = "MultiPriNTF_training"
    ):
        """
        Initialize the training system.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config_path: Path to configuration file
            experiment_name: Name of experiment
        """
        self.config = self._load_config(config_path)
        self.experiment_name = experiment_name
        
        # Setup model and devices
        self.model = self._setup_model(model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Initialize training components
        self._initialize_training_components()
        
        # Setup logging and monitoring
        self._setup_logging()
        
        # Initialize metrics tracking
        self.metrics = self._initialize_metrics()
        
        # Initialize checkpoint management
        self._setup_checkpointing()
        
    def _load_config(self, config_path: str) -> dict:
        """
        Load and validate configuration.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self._validate_config(config)
            return config
        except Exception as e:
            raise ValueError(f"Error loading config: {e}")

    def _validate_config(self, config: dict):
        """
        Validate configuration parameters.

        Args:
            config: Configuration dictionary
        """
        required_sections = [
            'system', 'data', 'model', 'training',
            'security', 'evaluation', 'logging'
        ]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

    def _setup_model(self, model: nn.Module) -> nn.Module:
        """
        Setup model for multi-GPU training.

        Args:
            model: Model to setup

        Returns:
            Configured model
        """
        model = model.cuda()
        
        if self.config['system']['distributed']:
            model = DistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device()
            )
        
        return model

    def _initialize_training_components(self):
        """Initialize training components with optimized configuration."""
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['optimizer']['learning_rate'],
            betas=(
                self.config['training']['optimizer']['beta1'],
                self.config['training']['optimizer']['beta2']
            ),
            eps=self.config['training']['optimizer']['eps'],
            weight_decay=self.config['training']['optimizer']['weight_decay']
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['scheduler']['T_max'],
            eta_min=self.config['training']['scheduler']['min_lr']
        )
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler()
        
        # Setup loss weights
        self.loss_weights = self.config['training']['loss']
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['training']['early_stopping']['patience'],
            min_delta=self.config['training']['early_stopping']['min_delta']
        )

    def _setup_logging(self):
        """Configure comprehensive logging system."""
        # Create output directory
        self.output_dir = Path(self.config['output']['save_path']) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        
        # Initialize TensorBoard
        if self.config['logging']['tensorboard']['enabled']:
            self.writer = SummaryWriter(
                log_dir=str(Path(self.config['logging']['tensorboard']['save_path']) / self.experiment_name)
            )
        
        # Initialize W&B
        if self.config['logging']['wandb']['enabled']:
            wandb.init(
                project=self.config['logging']['wandb']['project'],
                entity=self.config['logging']['wandb']['entity'],
                config=self.config,
                name=self.experiment_name
            )

    def _initialize_metrics(self) -> Dict:
        """
        Initialize metrics tracking system.

        Returns:
            Metrics dictionary
        """
        return {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epoch_times': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0,
            'gpu_metrics': defaultdict(list),
            'memory_metrics': defaultdict(list)
        }

    def _setup_checkpointing(self):
        """Configure checkpoint management system."""
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize checkpoint tracking
        self.best_checkpoints = []
        self.checkpoint_frequency = self.config['training']['checkpointing']['frequency']
        self.max_checkpoints = self.config['training']['checkpointing']['keep_last']

@torch.cuda.amp.autocast()
    def _calculate_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate weighted loss components.

        Args:
            outputs: Model outputs
            targets: Target values

        Returns:
            Tuple of (total loss, loss components)
        """
        loss_components = {}
        
        # Reconstruction loss
        loss_components['recon_loss'] = F.mse_loss(
            outputs['reconstructed'],
            targets['images']
        )
        
        # KL divergence loss
        loss_components['kl_loss'] = -0.5 * torch.sum(
            1 + outputs['log_var'] - outputs['mu'].pow(2) - outputs['log_var'].exp()
        )
        
        # Market prediction loss
        loss_components['market_loss'] = F.mse_loss(
            outputs['market_prediction'],
            targets['prices']
        )
        
        # Attribute prediction loss
        loss_components['attr_loss'] = F.binary_cross_entropy_with_logits(
            outputs['attribute_prediction'],
            targets['attributes']
        )
        
        # Calculate total weighted loss
        total_loss = (
            self.loss_weights['reconstruction_weight'] * loss_components['recon_loss'] +
            self.loss_weights['kl_weight'] * loss_components['kl_loss'] +
            self.loss_weights['market_weight'] * loss_components['market_loss'] +
            self.loss_weights['attribute_weight'] * loss_components['attr_loss']
        )
        
        return total_loss, loss_components

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        epoch_metrics = defaultdict(float)
        epoch_start_time = time.time()
        
        for batch_idx, (images, types, attributes, prices) in enumerate(
            tqdm(self.train_loader, desc=f"Epoch {epoch}")
        ):
            try:
                # Move data to GPU
                images = images.cuda()
                types = types.cuda()
                attributes = attributes.cuda()
                prices = prices.cuda()
                
                # Forward pass with mixed precision
                with autocast():
                    outputs = self.model(images, types, attributes)
                    loss, loss_components = self._calculate_loss(
                        outputs,
                        {
                            'images': images,
                            'attributes': attributes,
                            'prices': prices
                        }
                    )
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update metrics
                for k, v in loss_components.items():
                    epoch_metrics[k] += v.item()
                epoch_metrics['total_loss'] += loss.item()
                
                # Log batch metrics
                if batch_idx % self.config['logging']['tensorboard']['log_freq'] == 0:
                    self._log_batch_metrics(epoch, batch_idx, epoch_metrics)
                    
            except Exception as e:
                logging.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Calculate epoch averages
        for k in epoch_metrics.keys():
            epoch_metrics[k] /= len(self.train_loader)
        
        epoch_metrics['epoch_time'] = time.time() - epoch_start_time
        return epoch_metrics

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        val_metrics = defaultdict(float)
        
        for images, types, attributes, prices in tqdm(
            self.val_loader,
            desc="Validation"
        ):
            try:
                # Move data to GPU
                images = images.cuda()
                types = types.cuda()
                attributes = attributes.cuda()
                prices = prices.cuda()
                
                # Forward pass
                with autocast():
                    outputs = self.model(images, types, attributes)
                    loss, loss_components = self._calculate_loss(
                        outputs,
                        {
                            'images': images,
                            'attributes': attributes,
                            'prices': prices
                        }
                    )
                
                # Update metrics
                for k, v in loss_components.items():
                    val_metrics[k] += v.item()
                val_metrics['total_loss'] += loss.item()
                
            except Exception as e:
                logging.error(f"Error in validation batch: {e}")
                continue
        
        # Calculate validation averages
        for k in val_metrics.keys():
            val_metrics[k] /= len(self.val_loader)
        
        return val_metrics

    def train(self):
        """Main training loop with comprehensive monitoring."""
        logging.info(f"Starting training with experiment name: {self.experiment_name}")
        
        try:
            for epoch in range(self.config['training']['epochs']):
                epoch_start_time = time.time()
                
                # Training phase
                train_metrics = self.train_epoch(epoch)
                
                # Validation phase
                val_metrics = self.validate_epoch()
                
                # Update learning rate
                self.scheduler.step()
                
                # Log epoch metrics
                self._log_epoch_metrics(epoch, train_metrics, val_metrics)
                
                # Save checkpoint if best model
                if val_metrics['total_loss'] < self.metrics['best_val_loss']:
                    self.metrics['best_val_loss'] = val_metrics['total_loss']
                    self.metrics['best_epoch'] = epoch
                    self._save_checkpoint(epoch, val_metrics['total_loss'])
                
                # Early stopping check
                if self.early_stopping.should_stop(val_metrics['total_loss']):
                    logging.info(f"Early stopping triggered at epoch {epoch}")
                    break
                
                # Update metrics history
                self._update_metrics_history(
                    epoch,
                    train_metrics,
                    val_metrics,
                    time.time() - epoch_start_time
                )
                
            # Save final training metrics
            self._save_training_metrics()
            
        except Exception as e:
            logging.error(f"Training error: {e}")
            raise
            
        finally:
            self._cleanup()

    def _log_batch_metrics(self, epoch: int, batch_idx: int, metrics: Dict[str, float]):
        """Log batch-level metrics to monitoring systems."""
        if self.config['logging']['tensorboard']['enabled']:
            step = epoch * len(self.train_loader) + batch_idx
            for k, v in metrics.items():
                self.writer.add_scalar(f'batch/{k}', v, step)
        
        if self.config['logging']['wandb']['enabled']:
            wandb.log({f'batch/{k}': v for k, v in metrics.items()})

    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log epoch-level metrics with comprehensive monitoring."""
        # Console logging
        logging.info(
            f"Epoch {epoch}: "
            f"Train Loss = {train_metrics['total_loss']:.4f}, "
            f"Val Loss = {val_metrics['total_loss']:.4f}, "
            f"LR = {self.optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # TensorBoard logging
        if self.config['logging']['tensorboard']['enabled']:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f'train/{k}', v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f'val/{k}', v, epoch)
            
        # W&B logging
        if self.config['logging']['wandb']['enabled']:
            wandb.log({
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
                'epoch': epoch
            })

def _save_checkpoint(self, epoch: int, val_loss: float):
        """
        Save model checkpoint with comprehensive metadata.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'metrics_history': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save checkpoint with version control
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        try:
            torch.save(checkpoint, checkpoint_path)
            self.best_checkpoints.append((checkpoint_path, val_loss))
            self._cleanup_old_checkpoints()
            logging.info(f"Saved checkpoint at epoch {epoch}")
            
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the best ones."""
        # Sort checkpoints by validation loss
        self.best_checkpoints.sort(key=lambda x: x[1])
        
        # Remove excess checkpoints
        while len(self.best_checkpoints) > self.max_checkpoints:
            checkpoint_path, _ = self.best_checkpoints.pop()
            try:
                checkpoint_path.unlink()
            except Exception as e:
                logging.error(f"Error removing checkpoint {checkpoint_path}: {e}")

    def _update_metrics_history(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float
    ):
        """
        Update training metrics history.

        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
            epoch_time: Epoch duration
        """
        self.metrics['train_losses'].append(train_metrics['total_loss'])
        self.metrics['val_losses'].append(val_metrics['total_loss'])
        self.metrics['learning_rates'].append(
            self.optimizer.param_groups[0]['lr']
        )
        self.metrics['epoch_times'].append(epoch_time)
        
        # Update GPU metrics
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.metrics['gpu_metrics'][f'gpu_{i}_memory'].append(
                    torch.cuda.memory_allocated(i) / 1024**3
                )
                self.metrics['gpu_metrics'][f'gpu_{i}_utilization'].append(
                    torch.cuda.utilization(i)
                )

    def _save_training_metrics(self):
        """Save comprehensive training metrics and visualizations."""
        metrics_path = self.output_dir / 'training_metrics.json'
        
        # Prepare metrics for saving
        save_metrics = {
            'train_history': {
                'losses': self.metrics['train_losses'],
                'learning_rates': self.metrics['learning_rates']
            },
            'validation_history': {
                'losses': self.metrics['val_losses'],
                'best_epoch': self.metrics['best_epoch'],
                'best_loss': self.metrics['best_val_loss']
            },
            'performance_metrics': {
                'epoch_times': self.metrics['epoch_times'],
                'gpu_metrics': dict(self.metrics['gpu_metrics'])
            },
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save metrics
        try:
            with open(metrics_path, 'w') as f:
                json.dump(save_metrics, f, indent=2)
                
            # Generate training visualizations
            self._generate_training_plots()
            
        except Exception as e:
            logging.error(f"Failed to save metrics: {e}")

    def _generate_training_plots(self):
        """Generate comprehensive training visualization plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn')
        
        # Create plots directory
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Loss plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.metrics['train_losses'], label='Train Loss')
        plt.plot(self.metrics['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(plots_dir / 'loss_history.png')
        plt.close()
        
        # Learning rate plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.metrics['learning_rates'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.savefig(plots_dir / 'lr_schedule.png')
        plt.close()
        
        # GPU metrics plot
        if torch.cuda.is_available():
            plt.figure(figsize=(12, 6))
            for gpu_id in range(torch.cuda.device_count()):
                plt.plot(
                    self.metrics['gpu_metrics'][f'gpu_{gpu_id}_memory'],
                    label=f'GPU {gpu_id}'
                )
            plt.xlabel('Epoch')
            plt.ylabel('Memory Usage (GB)')
            plt.title('GPU Memory Usage')
            plt.legend()
            plt.savefig(plots_dir / 'gpu_memory.png')
            plt.close()

    def _cleanup(self):
        """Cleanup resources and connections."""
        try:
            # Close TensorBoard writer
            if hasattr(self, 'writer'):
                self.writer.close()
                
            # Close wandb run
            if wandb.run is not None:
                wandb.finish()
                
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

class EarlyStopping:
    """
    Early stopping implementation with dynamic patience.
    """
    
    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change in monitored value
            mode: 'min' or 'max' for loss or metric monitoring
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def should_stop(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            Boolean indicating if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
            
        if self.mode == 'min':
            improvement = self.best_loss - val_loss > self.min_delta
        else:
            improvement = val_loss - self.best_loss > self.min_delta
            
        if improvement:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config_path: str,
    experiment_name: str = "MultiPriNTF_training"
) -> OptimizedTrainer:
    """
    Create and configure a trainer instance.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config_path: Path to configuration file
        experiment_name: Name of experiment
        
    Returns:
        Configured OptimizedTrainer instance
    """
    return OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config_path=config_path,
        experiment_name=experiment_name
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MultiPriNTF Model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='MultiPriNTF_training',
        help='Experiment name'
    )
    args = parser.parse_args()
    
    # Import your model and data loaders
    from model import MultiPriNTF
    from data_loader import create_data_loaders
    
    # Create model and data loaders
    model = MultiPriNTF()
    train_loader, val_loader = create_data_loaders()
    
    # Create and start trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config_path=args.config,
        experiment_name=args.experiment
    )
    
    # Start training
    trainer.train()