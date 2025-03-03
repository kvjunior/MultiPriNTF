"""
MultiPriNTF Training Callback System
=============================

This module implements a comprehensive callback system for training monitoring and 
optimization of the MultiPriNTF (Hybrid Variational Autoencoder-Transformer) architecture.

The callback system provides:
1. Training monitoring and optimization
2. Checkpoint management with versioning
3. Early stopping with adaptive patience
4. Performance monitoring across multiple GPUs
5. Integration with TensorBoard and Weights & Biases
6. Comprehensive metric tracking and analysis
7. Resource utilization optimization

Theoretical Foundations
---------------------
The callbacks implement various optimization techniques including:
- Adaptive learning rate adjustment using validation metrics
- Dynamic batch size scaling based on memory utilization
- Multi-modal loss balancing for stable convergence
- Gradient norm monitoring for training stability
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json
import time
from datetime import datetime
import psutil
import GPUtil
from collections import deque
import wandb
import h5py
import threading
from queue import Queue

class CallbackManager:
    """
    Manages and coordinates multiple training callbacks with error handling
    and resource optimization.
    """
    def __init__(self, callbacks: List["BaseCallback"]):
        """
        Initialize callback manager with error handling and logging.
        
        Args:
            callbacks: List of callback instances to manage
        """
        self.callbacks = callbacks
        self.logger = logging.getLogger(__name__)
        self._initialize_logging()
        
    def _initialize_logging(self):
        """Configure logging system for callback management."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def on_training_start(self, trainer: Any):
        """
        Execute callbacks at training start with comprehensive error handling.
        
        Args:
            trainer: Training instance containing model and optimization state
        """
        for callback in self.callbacks:
            try:
                callback.on_training_start(trainer)
            except Exception as e:
                self.logger.error(
                    f"Error in {callback.__class__.__name__}.on_training_start: {e}"
                )
                self._handle_callback_error(callback, e)
                
    def on_training_end(self, trainer: Any):
        """
        Execute callbacks at training end with resource cleanup.
        
        Args:
            trainer: Training instance containing model and optimization state
        """
        for callback in self.callbacks:
            try:
                callback.on_training_end(trainer)
            except Exception as e:
                self.logger.error(
                    f"Error in {callback.__class__.__name__}.on_training_end: {e}"
                )
                self._handle_callback_error(callback, e)
        self._cleanup_resources()
                
    def on_epoch_start(self, trainer: Any, epoch: int):
        """
        Execute callbacks at epoch start with performance monitoring.
        
        Args:
            trainer: Training instance containing model and optimization state
            epoch: Current epoch number
        """
        self._monitor_resources()
        for callback in self.callbacks:
            try:
                callback.on_epoch_start(trainer, epoch)
            except Exception as e:
                self.logger.error(
                    f"Error in {callback.__class__.__name__}.on_epoch_start: {e}"
                )
                self._handle_callback_error(callback, e)
                
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]):
        """
        Execute callbacks at epoch end with metric logging.
        
        Args:
            trainer: Training instance containing model and optimization state
            epoch: Current epoch number
            logs: Dictionary containing training metrics
        """
        for callback in self.callbacks:
            try:
                callback.on_epoch_end(trainer, epoch, logs)
            except Exception as e:
                self.logger.error(
                    f"Error in {callback.__class__.__name__}.on_epoch_end: {e}"
                )
                self._handle_callback_error(callback, e)
        self._log_metrics(epoch, logs)
                
    def on_batch_start(self, trainer: Any, batch: int):
        """
        Execute callbacks at batch start with memory optimization.
        
        Args:
            trainer: Training instance containing model and optimization state
            batch: Current batch number
        """
        self._optimize_memory()
        for callback in self.callbacks:
            try:
                callback.on_batch_start(trainer, batch)
            except Exception as e:
                self.logger.error(
                    f"Error in {callback.__class__.__name__}.on_batch_start: {e}"
                )
                self._handle_callback_error(callback, e)
                
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict[str, float]):
        """
        Execute callbacks at batch end with performance logging.
        
        Args:
            trainer: Training instance containing model and optimization state
            batch: Current batch number
            logs: Dictionary containing training metrics
        """
        for callback in self.callbacks:
            try:
                callback.on_batch_end(trainer, batch, logs)
            except Exception as e:
                self.logger.error(
                    f"Error in {callback.__class__.__name__}.on_batch_end: {e}"
                )
                self._handle_callback_error(callback, e)
        self._log_batch_metrics(batch, logs)
        
    def _handle_callback_error(self, callback: "BaseCallback", error: Exception):
        """
        Handle callback errors with appropriate recovery actions.
        
        Args:
            callback: Failed callback instance
            error: Exception that occurred
        """
        # Log error details
        self.logger.error(f"Callback error details: {str(error)}")
        
        # Attempt recovery based on error type
        if isinstance(error, RuntimeError):
            self._handle_runtime_error(callback)
        elif isinstance(error, MemoryError):
            self._handle_memory_error()
            
        # Notify monitoring systems
        self._notify_error(callback, error)
        
    def _handle_runtime_error(self, callback: "BaseCallback"):
        """
        Handle runtime errors in callbacks.
        
        Args:
            callback: Failed callback instance
        """
        # Attempt to reset callback state
        if hasattr(callback, 'reset'):
            callback.reset()
            
        # Clear CUDA cache if GPU-related error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _handle_memory_error(self):
        """Handle memory-related errors with resource optimization."""
        # Clear memory caches
        torch.cuda.empty_cache()
        
        # Reduce batch size if possible
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'batch_size'):
            self.trainer.batch_size = max(1, self.trainer.batch_size // 2)
            
    def _notify_error(self, callback: "BaseCallback", error: Exception):
        """
        Notify monitoring systems of callback errors.
        
        Args:
            callback: Failed callback instance
            error: Exception that occurred
        """
        if wandb.run is not None:
            wandb.alert(
                title="Callback Error",
                text=f"Error in {callback.__class__.__name__}: {str(error)}"
            )
            
    def _monitor_resources(self):
        """Monitor system resource utilization."""
        # Monitor GPU utilization
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_used = torch.cuda.memory_allocated(i) / 1024**3
                self.logger.info(f"GPU {i} memory used: {memory_used:.2f}GB")
                
        # Monitor CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        self.logger.info(
            f"CPU utilization: {cpu_percent}%, Memory utilization: {memory_percent}%"
        )
        
    def _optimize_memory(self):
        """Optimize memory usage during training."""
        if torch.cuda.is_available():
            # Clear unused memory
            torch.cuda.empty_cache()
            
            # Check memory pressure
            for i in range(torch.cuda.device_count()):
                memory_used = torch.cuda.memory_allocated(i) / 1024**3
                if memory_used > 0.9 * torch.cuda.get_device_properties(i).total_memory:
                    self.logger.warning(f"High memory usage on GPU {i}")
                    
    def _log_metrics(self, epoch: int, logs: Dict[str, float]):
        """
        Log training metrics to monitoring systems.
        
        Args:
            epoch: Current epoch number
            logs: Dictionary containing metrics
        """
        # Log to TensorBoard
        if hasattr(self, 'writer'):
            for key, value in logs.items():
                self.writer.add_scalar(f'epoch/{key}', value, epoch)
                
        # Log to Weights & Biases
        if wandb.run is not None:
            wandb.log({f'epoch/{k}': v for k, v in logs.items()}, step=epoch)
            
    def _log_batch_metrics(self, batch: int, logs: Dict[str, float]):
        """
        Log batch-level metrics to monitoring systems.
        
        Args:
            batch: Current batch number
            logs: Dictionary containing metrics
        """
        # Log to TensorBoard
        if hasattr(self, 'writer'):
            for key, value in logs.items():
                self.writer.add_scalar(f'batch/{key}', value, batch)
                
        # Log to Weights & Biases
        if wandb.run is not None:
            wandb.log({f'batch/{k}': v for k, v in logs.items()}, step=batch)
            
    def _cleanup_resources(self):
        """Clean up resources and close connections."""
        # Close TensorBoard writer
        if hasattr(self, 'writer'):
            self.writer.close()
            
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Close wandb run
        if wandb.run is not None:
            wandb.finish()

class BaseCallback:
    """
    Base class for all callbacks with core functionality and error handling.
    """
    def __init__(self):
        """Initialize base callback with logging configuration."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def on_training_start(self, trainer: Any):
        """Handle training start event."""
        pass
        
    def on_training_end(self, trainer: Any):
        """Handle training end event."""
        pass
        
    def on_epoch_start(self, trainer: Any, epoch: int):
        """Handle epoch start event."""
        pass
        
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]):
        """Handle epoch end event."""
        pass
        
    def on_batch_start(self, trainer: Any, batch: int):
        """Handle batch start event."""
        pass
        
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict[str, float]):
        """Handle batch end event."""
        pass
        
    def reset(self):
        """Reset callback state."""
        pass

class ModelCheckpoint(BaseCallback):
    """
    Advanced model checkpointing with versioning and optimization.
    """
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        max_save: int = 3,
        mode: str = 'min'
    ):
        """
        Initialize checkpoint callback with configuration options.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor for saving decisions
            save_best_only: Only save if monitored metric improves
            save_weights_only: Only save model weights
            max_save: Maximum number of checkpoints to keep
            mode: 'min' or 'max' for metric monitoring
        """
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.max_save = max_save
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.saved_checkpoints = []
        
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]):
        """
        Handle checkpoint saving at epoch end.
        
        Args:
            trainer: Training instance
            epoch: Current epoch number
            logs: Dictionary containing metrics
        """
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
            
        filepath = self.filepath / f'checkpoint_epoch_{epoch}.pt'
        
        if self.save_best_only:
            if (self.mode == 'min' and current_value < self.best_value) or \
               (self.mode == 'max' and current_value > self.best_value):
                self.best_value = current_value
                self._save_checkpoint(trainer, filepath, epoch, logs)
        else:
            self._save_checkpoint(trainer, filepath, epoch, logs)
            
        self._cleanup_old_checkpoints()
        
    def _save_checkpoint(
        self,
        trainer: Any,
        filepath: Path,
        epoch: int,
        logs: Dict[str, float]
    ):
        """
        Save model checkpoint with optimized storage.
        
        Args:
            trainer: Training instance
            filepath: Path to save checkpoint
            epoch: Current epoch number
            logs: Dictionary containing metrics
        """
        try:
            # Prepare checkpoint data
            checkpoint = {
                'epoch': epoch,
                'logs': logs,
                'model_state_dict': trainer.model.state_dict() if not self.save_weights_only 
                    else trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict() 
                    if hasattr(trainer, 'scheduler') else None
            }
            
            # Save with error handling
            torch.save(checkpoint, filepath)
            self.saved_checkpoints.append(filepath)
            self.logger.info(f"Saved checkpoint to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain storage efficiency."""
        if len(self.saved_checkpoints) > self.max_save:
            old_checkpoint = self.saved_checkpoints.pop(0)
            try:
                old_checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                self.logger.error(f"Failed to remove checkpoint: {e}")

class EarlyStopping(BaseCallback):
    """
    Early stopping implementation with adaptive patience.
    """
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = 'min',
        min_epochs: int = 20,
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping with configuration.
        
        Args:
            monitor: Metric to monitor
            min_delta: Minimum change in monitored value
            patience: Number of epochs with no improvement before stopping
            mode: 'min' or 'max' for metric monitoring
            min_epochs: Minimum number of epochs before stopping
            restore_best_weights: Restore model to best weights when stopping
        """
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.min_epochs = min_epochs
        self.restore_best_weights = restore_best_weights
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]):
        """
        Check for early stopping conditions at epoch end.
        
        Args:
            trainer: Training instance
            epoch: Current epoch number
            logs: Dictionary containing metrics
        """
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
            
        if epoch < self.min_epochs:
            return
            
        if (self.mode == 'min' and current_value < self.best_value - self.min_delta) or \
           (self.mode == 'max' and current_value > self.best_value + self.min_delta):
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = trainer.model.state_dict()
        else:
            self.wait += 1
            
            # Adapt patience based on training stability
            if self.wait > self.patience // 2:
                std_value = np.std([
                    logs.get(self.monitor) for logs in trainer.history[-self.patience:]
                ])
                if std_value < self.min_delta:
                    self.patience = max(self.patience - 1, 5)
                    
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    trainer.model.load_state_dict(self.best_weights)
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(best epoch: {self.best_epoch})"
                )

class PerformanceMonitor(BaseCallback):
    """
    Advanced performance monitoring system for training optimization.
    Tracks system metrics, resource utilization, and training efficiency.
    """
    def __init__(self, log_interval: int = 10):
        """
        Initialize performance monitoring system.
        
        Args:
            log_interval: Frequency of metric logging in batches
        """
        super().__init__()
        self.log_interval = log_interval
        self.metrics_history = {
            'gpu_utilization': deque(maxlen=1000),
            'gpu_memory': deque(maxlen=1000),
            'cpu_utilization': deque(maxlen=1000),
            'ram_usage': deque(maxlen=1000),
            'batch_time': deque(maxlen=1000)
        }
        self.batch_start_time = None
        
        # Initialize GPU monitoring
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.gpu_handles = [torch.cuda.device(i) for i in range(self.num_gpus)]
        
    def on_batch_start(self, trainer: Any, batch: int):
        """
        Record batch start time and initial resource state.
        
        Args:
            trainer: Training instance
            batch: Current batch number
        """
        self.batch_start_time = time.time()
        
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict[str, float]):
        """
        Monitor and log performance metrics at batch end.
        
        Args:
            trainer: Training instance
            batch: Current batch number
            logs: Dictionary containing metrics
        """
        if batch % self.log_interval == 0:
            # Calculate batch processing time
            batch_time = time.time() - self.batch_start_time
            
            # Get GPU metrics
            gpu_metrics = self._get_gpu_metrics()
            
            # Get CPU and memory metrics
            cpu_util = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent
            
            # Update metrics history
            self.metrics_history['gpu_utilization'].append(gpu_metrics['utilization'])
            self.metrics_history['gpu_memory'].append(gpu_metrics['memory'])
            self.metrics_history['cpu_utilization'].append(cpu_util)
            self.metrics_history['ram_usage'].append(ram_usage)
            self.metrics_history['batch_time'].append(batch_time)
            
            # Log current metrics
            self.logger.info(
                f"Batch {batch} - "
                f"GPU Util: {gpu_metrics['utilization']:.1f}%, "
                f"GPU Mem: {gpu_metrics['memory']:.1f}MB, "
                f"CPU Util: {cpu_util:.1f}%, "
                f"RAM: {ram_usage:.1f}%, "
                f"Batch Time: {batch_time:.3f}s"
            )
            
            # Check for performance bottlenecks
            self._check_bottlenecks(trainer)
            
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """
        Collect GPU performance metrics.
        
        Returns:
            Dictionary containing GPU metrics
        """
        if not torch.cuda.is_available():
            return {'utilization': 0.0, 'memory': 0.0}
            
        util_sum = 0
        mem_sum = 0
        
        for gpu_id in range(self.num_gpus):
            gpu = GPUtil.getGPUs()[gpu_id]
            util_sum += gpu.load * 100
            mem_sum += gpu.memoryUsed
            
        return {
            'utilization': util_sum / self.num_gpus,
            'memory': mem_sum / self.num_gpus
        }
        
    def _check_bottlenecks(self, trainer: Any):
        """
        Check for and address performance bottlenecks.
        
        Args:
            trainer: Training instance
        """
        # Check GPU memory pressure
        if torch.cuda.is_available():
            for gpu_id in range(self.num_gpus):
                memory_used = torch.cuda.memory_allocated(gpu_id) / 1024**3
                if memory_used > 0.9 * torch.cuda.get_device_properties(gpu_id).total_memory:
                    self.logger.warning(f"High GPU memory usage detected on GPU {gpu_id}")
                    self._handle_memory_pressure(trainer)
                    
        # Check CPU bottlenecks
        if np.mean(list(self.metrics_history['cpu_utilization'])) > 90:
            self.logger.warning("High CPU utilization detected")
            self._handle_cpu_bottleneck(trainer)
            
    def _handle_memory_pressure(self, trainer: Any):
        """
        Handle GPU memory pressure situations.
        
        Args:
            trainer: Training instance
        """
        # Clear cache
        torch.cuda.empty_cache()
        
        # Reduce batch size if possible
        if hasattr(trainer, 'batch_size'):
            trainer.batch_size = max(1, trainer.batch_size // 2)
            self.logger.info(f"Reduced batch size to {trainer.batch_size}")
            
    def _handle_cpu_bottleneck(self, trainer: Any):
        """
        Handle CPU bottleneck situations.
        
        Args:
            trainer: Training instance
        """
        # Adjust number of worker processes
        if hasattr(trainer.dataloader, 'num_workers'):
            trainer.dataloader.num_workers = max(1, trainer.dataloader.num_workers - 1)
            self.logger.info(
                f"Reduced dataloader workers to {trainer.dataloader.num_workers}"
            )

class TensorBoardCallback(BaseCallback):
    """
    Enhanced TensorBoard logging with comprehensive metric tracking.
    """
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logging system.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]):
        """
        Log epoch-level metrics to TensorBoard.
        
        Args:
            trainer: Training instance
            epoch: Current epoch number
            logs: Dictionary containing metrics
        """
        # Log training metrics
        for key, value in logs.items():
            self.writer.add_scalar(f'metrics/{key}', value, epoch)
            
        # Log learning rate
        lr = trainer.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('training/learning_rate', lr, epoch)
        
        # Log model parameters distribution
        for name, param in trainer.model.named_parameters():
            self.writer.add_histogram(f'parameters/{name}', param.data, epoch)
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad, epoch)
                
        # Log memory usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                self.writer.add_scalar(
                    f'system/gpu_{i}_memory',
                    memory_allocated,
                    epoch
                )
                
    def on_training_end(self, trainer: Any):
        """Clean up TensorBoard writer."""
        self.writer.close()

class WandBCallback(BaseCallback):
    """
    Weights & Biases integration for experiment tracking.
    """
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize W&B logging system.
        
        Args:
            project: W&B project name
            entity: W&B entity name
            config: Configuration dictionary for logging
        """
        super().__init__()
        self.project = project
        self.entity = entity
        self.config = config or {}
        
    def on_training_start(self, trainer: Any):
        """Initialize W&B run."""
        wandb.init(
            project=self.project,
            entity=self.entity,
            config=self.config
        )
        
        # Log model architecture
        wandb.watch(
            trainer.model,
            log="all",
            log_freq=100
        )
        
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]):
        """
        Log epoch-level metrics to W&B.
        
        Args:
            trainer: Training instance
            epoch: Current epoch number
            logs: Dictionary containing metrics
        """
        # Log metrics
        wandb.log(logs, step=epoch)
        
        # Log system metrics
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                wandb.log({
                    f"system/gpu_{i}_memory": torch.cuda.memory_allocated(i) / 1024**3
                }, step=epoch)
                
    def on_training_end(self, trainer: Any):
        """Clean up W&B run."""
        wandb.finish()

class MetricsHistory(BaseCallback):
    """
    Track and analyze training metrics history.
    """
    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize metrics tracking system.
        
        Args:
            filepath: Path to save metrics history
        """
        super().__init__()
        self.filepath = Path(filepath) if filepath else None
        self.history = defaultdict(list)
        
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]):
        """
        Update metrics history at epoch end.
        
        Args:
            trainer: Training instance
            epoch: Current epoch number
            logs: Dictionary containing metrics
        """
        # Update history
        for key, value in logs.items():
            self.history[key].append(value)
            
        # Save history if filepath provided
        if self.filepath:
            self._save_history()
            
    def _save_history(self):
        """Save metrics history to JSON file."""
        history_dict = {
            k: np.array(v).tolist() for k, v in self.history.items()
        }
        
        with open(self.filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)

class LRSchedulerCallback(BaseCallback):
    """
    Learning rate scheduler with monitoring and adjustment.
    """
    def __init__(
        self,
        scheduler_type: str = 'reduce_lr_on_plateau',
        monitor: str = 'val_loss',
        **scheduler_params
    ):
        """
        Initialize learning rate scheduler.
        
        Args:
            scheduler_type: Type of scheduler to use
            monitor: Metric to monitor for scheduling
            scheduler_params: Additional scheduler parameters
        """
        super().__init__()
        self.scheduler_type = scheduler_type
        self.monitor = monitor
        self.scheduler_params = scheduler_params
        self.scheduler = None
        
    def on_training_start(self, trainer: Any):
        """Initialize scheduler based on type."""
        if self.scheduler_type == 'reduce_lr_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                trainer.optimizer,
                mode='min',
                **self.scheduler_params
            )
        elif self.scheduler_type == 'cosine_annealing':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer,
                **self.scheduler_params
            )
            
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]):
        """
        Update learning rate based on scheduler type.
        
        Args:
            trainer: Training instance
            epoch: Current epoch number
            logs: Dictionary containing metrics
        """
        if self.scheduler_type == 'reduce_lr_on_plateau':
            self.scheduler.step(logs[self.monitor])
        else:
            self.scheduler.step()

def create_default_callbacks(
    experiment_name: str,
    output_dir: str,
    **kwargs
) -> List[BaseCallback]:
    """
    Create default set of callbacks for training.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Output directory for logs and checkpoints
        **kwargs: Additional callback configuration options
        
    Returns:
        List of configured callbacks
    """
    callbacks = [
        ModelCheckpoint(
            filepath=f"{output_dir}/checkpoints",
            monitor='val_loss',
            save_best_only=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_epochs=20
        ),
        PerformanceMonitor(log_interval=10),
        TensorBoardCallback(
            log_dir=f"{output_dir}/tensorboard/{experiment_name}"
        ),
        MetricsHistory(
            filepath=f"{output_dir}/metrics_history.json"
        ),
        LRSchedulerCallback(
            scheduler_type='reduce_lr_on_plateau',
            patience=5,
            factor=0.1
        )
    ]
    
    # Add W&B callback if configured
    if kwargs.get('wandb_project'):
        callbacks.append(
            WandBCallback(
                project=kwargs['wandb_project'],
                entity=kwargs.get('wandb_entity'),
                config=kwargs.get('wandb_config')
            )
        )
        
    return callbacks