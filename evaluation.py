"""
MultiPriNTF Evaluation System
=======================

This module implements a comprehensive evaluation framework for the MultiPriNTF
architecture, providing detailed analysis of multimedia processing capabilities,
system performance, and market analysis accuracy.

Key Features:
1. Multi-modal evaluation metrics
2. Real-time performance monitoring
3. Advanced visualization capabilities
4. System resource analysis
5. Cross-platform compatibility testing

Technical Architecture:
The evaluation system uses a modular approach with separate components for
different aspects of system evaluation, allowing for detailed analysis of
multimedia processing capabilities, market prediction accuracy, and
system performance metrics.
"""

import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import h5py
import psutil
import GPUtil
from concurrent.futures import ThreadPoolExecutor

class ModelEvaluator:
    """
    Comprehensive evaluation system for NFT market analysis model with
    focus on multimedia processing capabilities and system performance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        num_gpus: int = 3,
        output_dir: str = "evaluation_results"
    ):
        """
        Initialize the evaluation system.

        Args:
            model: The model to evaluate
            val_loader: Validation data loader
            num_gpus: Number of available GPUs
            output_dir: Directory for saving evaluation results
        """
        self.model = model
        self.val_loader = val_loader
        self.num_gpus = num_gpus
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize GPU streams for parallel processing
        self.streams = [torch.cuda.Stream() for _ in range(num_gpus)]
        
        # Initialize metrics modules
        self.metrics = {
            'technical': TechnicalMetrics(),
            'multimedia': MultimediaMetrics(),
            'economic': EconomicMetrics(),
            'system': SystemPerformanceMetrics()
        }
        
        # Initialize results storage
        self.results_file = self.output_dir / 'evaluation_results.h5'
        self.results_storage = h5py.File(self.results_file, 'a')
        
        # Setup visualization parameters
        self._setup_visualization()
        
    def _setup_logging(self):
        """Configure comprehensive logging system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_visualization(self):
        """Configure visualization settings."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 300

    @torch.cuda.amp.autocast()
    def evaluate_model(self) -> Dict[str, float]:
        """
        Perform comprehensive model evaluation with detailed metrics.
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        self.logger.info("Starting comprehensive model evaluation")
        all_metrics = {}
        
        try:
            with torch.no_grad():
                for batch_idx, (images, types, attributes, prices) in enumerate(
                    tqdm(self.val_loader, desc="Evaluating")
                ):
                    # Distribute batch across GPUs
                    gpu_id = batch_idx % self.num_gpus
                    with torch.cuda.stream(self.streams[gpu_id]):
                        # Process batch
                        batch_metrics = self._process_batch(
                            images, types, attributes, prices, gpu_id
                        )
                        
                        # Update metrics
                        self._update_metrics(batch_metrics)
                
                # Calculate final metrics
                all_metrics = self._calculate_final_metrics()
                
                # Store and visualize results
                self._store_evaluation_results(all_metrics)
                self._generate_evaluation_plots(all_metrics)
                
                self.logger.info("Evaluation completed successfully")
                
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise
            
        return all_metrics

    def _process_batch(
        self,
        images: torch.Tensor,
        types: torch.Tensor,
        attributes: torch.Tensor,
        prices: torch.Tensor,
        gpu_id: int
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single batch with comprehensive metrics collection.
        
        Args:
            images: Batch of images
            types: NFT type labels
            attributes: NFT attributes
            prices: NFT prices
            gpu_id: GPU to use for processing
            
        Returns:
            Dictionary containing batch metrics
        """
        # Move data to appropriate GPU
        images = images.cuda(gpu_id)
        types = types.cuda(gpu_id)
        attributes = attributes.cuda(gpu_id)
        prices = prices.cuda(gpu_id)
        
        # Get model predictions
        outputs = self.model(images, types, attributes)
        
        # Calculate batch metrics
        batch_metrics = {
            'reconstruction_loss': F.mse_loss(
                outputs['reconstructed'],
                images
            ),
            'attribute_accuracy': self._calculate_attribute_accuracy(
                outputs['attribute_prediction'],
                attributes
            ),
            'price_prediction_error': self._calculate_price_error(
                outputs['price_prediction'],
                prices
            ),
            'processing_time': self._measure_processing_time()
        }
        
        return batch_metrics

    def _calculate_attribute_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate attribute prediction accuracy with detailed metrics.
        
        Args:
            predictions: Predicted attributes
            targets: Target attributes
            
        Returns:
            Attribute accuracy metrics
        """
        predictions = (predictions > 0.5).float()
        correct = (predictions == targets).float()
        accuracy = correct.mean()
        
        return accuracy

    def _calculate_price_error(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate price prediction error with multiple metrics.
        
        Args:
            predictions: Predicted prices
            targets: Target prices
            
        Returns:
            Price prediction error metrics
        """
        mse = F.mse_loss(predictions, targets)
        mae = F.l1_loss(predictions, targets)
        mape = (torch.abs(predictions - targets) / (targets + 1e-8)).mean()
        
        return {
            'mse': mse,
            'mae': mae,
            'mape': mape
        }

    def _measure_processing_time(self) -> float:
        """
        Measure batch processing time with CPU and GPU metrics.
        
        Returns:
            Processing time metrics
        """
        return {
            'gpu_time': torch.cuda.elapsed_time(
                self.streams[0].record_event()
            ),
            'cpu_time': time.time()
        }

def _update_metrics(self, batch_metrics: Dict[str, torch.Tensor]):
        """
        Update all evaluation metrics with batch results.
        
        Args:
            batch_metrics: Dictionary containing batch metrics
        """
        # Update technical metrics
        self.metrics['technical'].update(
            batch_metrics['reconstruction_loss'],
            batch_metrics['attribute_accuracy']
        )
        
        # Update multimedia metrics
        self.metrics['multimedia'].update(
            batch_metrics['processing_time'],
            batch_metrics.get('visual_quality', None)
        )
        
        # Update economic metrics
        self.metrics['economic'].update(
            batch_metrics['price_prediction_error']
        )
        
        # Update system metrics
        self.metrics['system'].update(
            batch_metrics['processing_time']
        )

    def _calculate_final_metrics(self) -> Dict[str, float]:
        """
        Calculate final comprehensive evaluation metrics.
        Combines results from all metric categories into a detailed report.
        
        Returns:
            Dictionary containing all final metrics
        """
        final_metrics = {}
        
        # Calculate metrics for each category
        for metric_type, metric_calculator in self.metrics.items():
            final_metrics[metric_type] = metric_calculator.compute()
            
        # Add system-wide metrics
        final_metrics['system_wide'] = {
            'total_processing_time': self.metrics['system'].total_time,
            'average_batch_time': self.metrics['system'].avg_batch_time,
            'gpu_utilization': self._calculate_gpu_utilization(),
            'memory_efficiency': self._calculate_memory_efficiency()
        }
        
        return final_metrics

    def _calculate_gpu_utilization(self) -> Dict[str, float]:
        """
        Calculate detailed GPU utilization metrics across all available GPUs.
        
        Returns:
            Dictionary containing GPU utilization metrics
        """
        gpu_metrics = {}
        
        if torch.cuda.is_available():
            for gpu_id in range(self.num_gpus):
                gpu = GPUtil.getGPUs()[gpu_id]
                gpu_metrics[f'gpu_{gpu_id}'] = {
                    'utilization': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                }
                
        return gpu_metrics

    def _calculate_memory_efficiency(self) -> Dict[str, float]:
        """
        Calculate memory usage efficiency metrics for both CPU and GPU.
        
        Returns:
            Dictionary containing memory efficiency metrics
        """
        memory_metrics = {
            'cpu_memory': {
                'used': psutil.Process().memory_info().rss / 1024**3,
                'total': psutil.virtual_memory().total / 1024**3,
                'percent': psutil.virtual_memory().percent
            }
        }
        
        if torch.cuda.is_available():
            for gpu_id in range(self.num_gpus):
                memory_metrics[f'gpu_{gpu_id}_memory'] = {
                    'allocated': torch.cuda.memory_allocated(gpu_id) / 1024**3,
                    'cached': torch.cuda.memory_reserved(gpu_id) / 1024**3
                }
                
        return memory_metrics

    def _store_evaluation_results(self, metrics: Dict[str, float]):
        """
        Store evaluation results in HDF5 format with comprehensive metadata.
        
        Args:
            metrics: Dictionary containing evaluation metrics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with self.results_storage.create_group(f'evaluation_{timestamp}') as group:
            # Store metrics
            for metric_type, metric_values in metrics.items():
                if isinstance(metric_values, dict):
                    metric_subgroup = group.create_group(metric_type)
                    for k, v in metric_values.items():
                        metric_subgroup.create_dataset(k, data=v)
                else:
                    group.create_dataset(metric_type, data=metric_values)
            
            # Store metadata
            group.attrs['timestamp'] = timestamp
            group.attrs['num_gpus'] = self.num_gpus
            group.attrs['dataset_size'] = len(self.val_loader.dataset)

    def _generate_evaluation_plots(self, metrics: Dict[str, float]):
        """
        Generate comprehensive visualization plots for evaluation results.
        Creates detailed visualizations for each metric category.
        
        Args:
            metrics: Dictionary containing evaluation metrics
        """
        # Create plots directory
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Generate specific plots
        self._plot_technical_metrics(metrics['technical'], plots_dir)
        self._plot_multimedia_metrics(metrics['multimedia'], plots_dir)
        self._plot_economic_metrics(metrics['economic'], plots_dir)
        self._plot_system_metrics(metrics['system'], plots_dir)
        
        # Generate summary plot
        self._generate_summary_plot(metrics, plots_dir)

    def _plot_technical_metrics(
        self,
        metrics: Dict[str, float],
        output_dir: Path
    ):
        """
        Create visualization for technical performance metrics.
        
        Args:
            metrics: Technical metrics dictionary
            output_dir: Output directory for plots
        """
        plt.figure(figsize=(12, 8))
        
        # Create technical metrics visualization
        sns.barplot(
            x=list(metrics.keys()),
            y=list(metrics.values())
        )
        
        plt.title('Technical Performance Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'technical_metrics.png')
        plt.close()

    def _plot_multimedia_metrics(
        self,
        metrics: Dict[str, float],
        output_dir: Path
    ):
        """
        Create visualization for multimedia processing metrics.
        
        Args:
            metrics: Multimedia metrics dictionary
            output_dir: Output directory for plots
        """
        plt.figure(figsize=(12, 8))
        
        # Create multimedia metrics visualization
        performance_data = pd.DataFrame(metrics)
        sns.lineplot(data=performance_data)
        
        plt.title('Multimedia Processing Performance')
        plt.xlabel('Batch')
        plt.ylabel('Processing Time (ms)')
        plt.tight_layout()
        plt.savefig(output_dir / 'multimedia_metrics.png')
        plt.close()

    def _plot_economic_metrics(
        self,
        metrics: Dict[str, float],
        output_dir: Path
    ):
        """
        Create visualization for economic prediction metrics.
        
        Args:
            metrics: Economic metrics dictionary
            output_dir: Output directory for plots
        """
        plt.figure(figsize=(12, 8))
        
        # Create economic metrics visualization
        predictions = metrics.get('predictions', [])
        actual = metrics.get('actual', [])
        
        if predictions and actual:
            plt.scatter(actual, predictions, alpha=0.5)
            plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
        
        plt.title('Price Prediction Performance')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.tight_layout()
        plt.savefig(output_dir / 'economic_metrics.png')
        plt.close()

    def _plot_system_metrics(
        self,
        metrics: Dict[str, float],
        output_dir: Path
    ):
        """
        Create visualization for system performance metrics.
        
        Args:
            metrics: System metrics dictionary
            output_dir: Output directory for plots
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # GPU utilization plot
        gpu_util = [metrics[f'gpu_{i}_utilization'] for i in range(self.num_gpus)]
        ax1.bar(range(self.num_gpus), gpu_util)
        ax1.set_title('GPU Utilization')
        ax1.set_xlabel('GPU ID')
        ax1.set_ylabel('Utilization (%)')
        
        # Memory usage plot
        memory_usage = [
            metrics[f'gpu_{i}_memory']['used'] for i in range(self.num_gpus)
        ]
        ax2.bar(range(self.num_gpus), memory_usage)
        ax2.set_title('GPU Memory Usage')
        ax2.set_xlabel('GPU ID')
        ax2.set_ylabel('Memory Used (GB)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'system_metrics.png')
        plt.close()

    def _generate_summary_plot(
        self,
        metrics: Dict[str, float],
        output_dir: Path
    ):
        """
        Create comprehensive summary visualization of all metrics.
        
        Args:
            metrics: Complete metrics dictionary
            output_dir: Output directory for plots
        """
        plt.figure(figsize=(15, 10))
        
        # Create summary subplots
        gs = plt.GridSpec(2, 2)
        
        # Technical metrics
        ax1 = plt.subplot(gs[0, 0])
        self._plot_technical_summary(metrics['technical'], ax1)
        
        # Multimedia metrics
        ax2 = plt.subplot(gs[0, 1])
        self._plot_multimedia_summary(metrics['multimedia'], ax2)
        
        # Economic metrics
        ax3 = plt.subplot(gs[1, 0])
        self._plot_economic_summary(metrics['economic'], ax3)
        
        # System metrics
        ax4 = plt.subplot(gs[1, 1])
        self._plot_system_summary(metrics['system'], ax4)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'evaluation_summary.png')
        plt.close()

class TechnicalMetrics:
    """
    Technical metrics calculation for model evaluation.
    Focuses on core model performance metrics including reconstruction quality,
    latent space characteristics, and feature extraction efficiency.
    """
    
    def __init__(self):
        """Initialize technical metrics tracking system."""
        self.reset()
        
    def reset(self):
        """Reset all metrics counters and storage."""
        self.feature_distances = []
        self.reconstruction_errors = []
        self.latent_statistics = []
        self.batch_processing_times = []
        
    def update(
        self,
        fused_features: torch.Tensor,
        visual_features: torch.Tensor,
        transaction_features: torch.Tensor
    ):
        """
        Update technical metrics with new batch results.
        
        Args:
            fused_features: Features from fusion layer
            visual_features: Features from visual encoder
            transaction_features: Features from transaction encoder
        """
        # Calculate feature space metrics
        distance = torch.norm(visual_features - transaction_features, dim=1)
        self.feature_distances.extend(distance.cpu().numpy())
        
        # Calculate reconstruction quality
        recon_error = torch.mean((fused_features - visual_features) ** 2, dim=1)
        self.reconstruction_errors.extend(recon_error.cpu().numpy())
        
        # Calculate latent space statistics
        self.latent_statistics.append({
            'mean': torch.mean(fused_features).item(),
            'std': torch.std(fused_features).item(),
            'dimensionality': self._estimate_effective_dim(fused_features)
        })
        
    def _estimate_effective_dim(self, features: torch.Tensor) -> float:
        """
        Estimate effective dimensionality of feature space using PCA-based method.
        
        Args:
            features: Feature tensor to analyze
            
        Returns:
            Estimated effective dimensionality
        """
        # Calculate singular values
        _, s, _ = torch.svd(features)
        
        # Calculate normalized cumulative energy
        cumulative_energy = torch.cumsum(s, dim=0) / torch.sum(s)
        
        # Find effective dimension (95% energy threshold)
        return torch.sum(cumulative_energy < 0.95).item()
        
    def compute(self) -> Dict[str, float]:
        """
        Compute final technical metrics.
        
        Returns:
            Dictionary containing computed metrics
        """
        return {
            'avg_feature_distance': np.mean(self.feature_distances),
            'std_feature_distance': np.std(self.feature_distances),
            'avg_reconstruction_error': np.mean(self.reconstruction_errors),
            'latent_mean': np.mean([s['mean'] for s in self.latent_statistics]),
            'latent_std': np.mean([s['std'] for s in self.latent_statistics]),
            'effective_dim': np.mean([s['dimensionality'] for s in self.latent_statistics])
        }

class MultimediaMetrics:
    """
    Specialized metrics for multimedia processing evaluation.
    Focuses on visual quality assessment, processing efficiency,
    and content preservation metrics.
    """
    
    def __init__(self):
        """Initialize multimedia metrics tracking system."""
        self.reset()
        
    def reset(self):
        """Reset multimedia metrics storage."""
        self.visual_quality_scores = []
        self.processing_times = []
        self.content_preservation = []
        self.memory_usage = []
        
    def update(
        self,
        reconstructed_image: torch.Tensor,
        original_image: torch.Tensor,
        processing_time: float
    ):
        """
        Update multimedia metrics with new batch results.
        
        Args:
            reconstructed_image: Reconstructed image tensor
            original_image: Original image tensor
            processing_time: Batch processing time
        """
        # Calculate visual quality metrics
        quality_score = self._calculate_visual_quality(
            reconstructed_image,
            original_image
        )
        self.visual_quality_scores.append(quality_score)
        
        # Update processing metrics
        self.processing_times.append(processing_time)
        
        # Calculate content preservation score
        preservation_score = self._calculate_content_preservation(
            reconstructed_image,
            original_image
        )
        self.content_preservation.append(preservation_score)
        
        # Track memory usage
        self.memory_usage.append(
            torch.cuda.max_memory_allocated() / 1024**3
        )
        
    def _calculate_visual_quality(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor
    ) -> float:
        """
        Calculate comprehensive visual quality score.
        
        Args:
            reconstructed: Reconstructed image tensor
            original: Original image tensor
            
        Returns:
            Visual quality score
        """
        # Calculate PSNR (Peak Signal-to-Noise Ratio)
        mse = F.mse_loss(reconstructed, original)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        # Calculate SSIM (Structural Similarity Index)
        ssim = self._calculate_ssim(reconstructed, original)
        
        # Combined quality score
        return 0.5 * (psnr.item() / 50.0 + ssim.item())
        
    def _calculate_ssim(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Structural Similarity Index (SSIM).
        
        Args:
            reconstructed: Reconstructed image tensor
            original: Original image tensor
            
        Returns:
            SSIM score
        """
        # Constants for stability
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        
        # Calculate means
        mu1 = F.avg_pool2d(reconstructed, kernel_size=11, stride=1, padding=5)
        mu2 = F.avg_pool2d(original, kernel_size=11, stride=1, padding=5)
        
        # Calculate variances and covariance
        sigma1_sq = F.avg_pool2d(reconstructed**2, kernel_size=11, stride=1, padding=5) - mu1**2
        sigma2_sq = F.avg_pool2d(original**2, kernel_size=11, stride=1, padding=5) - mu2**2
        sigma12 = F.avg_pool2d(reconstructed * original, kernel_size=11, stride=1, padding=5) - mu1 * mu2
        
        # Calculate SSIM
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
               
        return ssim.mean()
        
    def _calculate_content_preservation(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor
    ) -> float:
        """
        Calculate content preservation score using feature similarity.
        
        Args:
            reconstructed: Reconstructed image tensor
            original: Original image tensor
            
        Returns:
            Content preservation score
        """
        # Extract features using pretrained model
        with torch.no_grad():
            original_features = self.feature_extractor(original)
            reconstructed_features = self.feature_extractor(reconstructed)
            
        # Calculate feature similarity
        similarity = F.cosine_similarity(
            original_features.view(original_features.size(0), -1),
            reconstructed_features.view(reconstructed_features.size(0), -1)
        )
        
        return similarity.mean().item()
        
    def compute(self) -> Dict[str, float]:
        """
        Compute final multimedia metrics.
        
        Returns:
            Dictionary containing computed metrics
        """
        return {
            'avg_visual_quality': np.mean(self.visual_quality_scores),
            'avg_processing_time': np.mean(self.processing_times),
            'content_preservation': np.mean(self.content_preservation),
            'peak_memory_usage': max(self.memory_usage),
            'throughput': len(self.processing_times) / sum(self.processing_times)
        }

class EconomicMetrics:
    """
    Economic metrics evaluation system for NFT market analysis.
    This class implements sophisticated metrics for assessing the model's
    ability to predict market behavior and price movements.
    """
    
    def __init__(self):
        """
        Initialize economic metrics tracking system with comprehensive
        market analysis capabilities.
        """
        self.reset()
        
    def reset(self):
        """
        Reset all economic metrics storage containers. This ensures clean
        state for new evaluation runs.
        """
        # Initialize price prediction metrics storage
        self.predicted_prices = []
        self.true_prices = []
        self.prediction_errors = []
        
        # Initialize market efficiency metrics
        self.market_efficiency_scores = []
        self.liquidity_metrics = []
        self.volatility_measures = []
        
    def update(
        self,
        predicted_prices: torch.Tensor,
        true_prices: torch.Tensor,
        market_data: Optional[Dict] = None
    ):
        """
        Update economic metrics with new batch results. This method processes
        both price predictions and broader market metrics.
        
        Args:
            predicted_prices: Model's price predictions
            true_prices: Actual prices from the market
            market_data: Additional market metrics (optional)
        """
        # Calculate basic price prediction metrics
        self.predicted_prices.extend(predicted_prices.cpu().numpy())
        self.true_prices.extend(true_prices.cpu().numpy())
        
        # Calculate prediction errors
        batch_errors = self._calculate_prediction_errors(
            predicted_prices,
            true_prices
        )
        self.prediction_errors.extend(batch_errors)
        
        # Calculate market efficiency if market data is provided
        if market_data is not None:
            efficiency_score = self._calculate_market_efficiency(market_data)
            self.market_efficiency_scores.append(efficiency_score)
            
            # Calculate liquidity metrics
            liquidity = self._calculate_liquidity_metrics(market_data)
            self.liquidity_metrics.append(liquidity)
            
            # Calculate volatility measures
            volatility = self._calculate_volatility(market_data)
            self.volatility_measures.append(volatility)
            
    def _calculate_prediction_errors(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> np.ndarray:
        """
        Calculate comprehensive prediction error metrics using multiple
        error measures for robust evaluation.
        
        Args:
            predictions: Predicted prices
            targets: Actual prices
            
        Returns:
            Array of prediction errors using different metrics
        """
        # Calculate percentage errors
        percentage_errors = torch.abs(predictions - targets) / (targets + 1e-8)
        
        # Calculate log price errors
        log_price_errors = torch.abs(
            torch.log1p(predictions) - torch.log1p(targets)
        )
        
        return np.array([
            percentage_errors.mean().item(),
            log_price_errors.mean().item()
        ])
        
    def _calculate_market_efficiency(self, market_data: Dict) -> float:
        """
        Calculate market efficiency score based on price prediction accuracy
        and market response speed.
        
        Args:
            market_data: Dictionary containing market metrics
            
        Returns:
            Market efficiency score
        """
        # Calculate price discovery efficiency
        price_efficiency = 1 - np.mean(np.abs(
            np.array(self.predicted_prices) - np.array(self.true_prices)
        ) / (np.array(self.true_prices) + 1e-8))
        
        # Calculate market response time if available
        response_efficiency = 1.0
        if 'response_time' in market_data:
            response_time = market_data['response_time']
            response_efficiency = 1 / (1 + np.exp(response_time - 5))
            
        # Combined efficiency score
        return 0.7 * price_efficiency + 0.3 * response_efficiency
        
    def _calculate_liquidity_metrics(self, market_data: Dict) -> Dict[str, float]:
        """
        Calculate comprehensive liquidity metrics including depth,
        spread, and volume-based measures.
        
        Args:
            market_data: Dictionary containing market metrics
            
        Returns:
            Dictionary of liquidity metrics
        """
        return {
            'depth': self._calculate_market_depth(market_data),
            'spread': self._calculate_bid_ask_spread(market_data),
            'volume': market_data.get('volume', 0.0),
            'turnover': self._calculate_turnover(market_data)
        }
        
    def _calculate_volatility(self, market_data: Dict) -> float:
        """
        Calculate price volatility using multiple time windows for
        comprehensive analysis.
        
        Args:
            market_data: Dictionary containing market metrics
            
        Returns:
            Volatility measure
        """
        # Calculate returns
        prices = np.array(self.true_prices)
        returns = np.diff(np.log(prices + 1e-8))
        
        # Calculate volatility measures
        daily_vol = np.std(returns) * np.sqrt(24)  # Assuming hourly data
        
        return daily_vol

    def compute(self) -> Dict[str, float]:
        """
        Compute final economic metrics combining all measures into a
        comprehensive market analysis report.
        
        Returns:
            Dictionary containing all computed economic metrics
        """
        return {
            'price_prediction': {
                'mse': np.mean((np.array(self.predicted_prices) - 
                              np.array(self.true_prices))**2),
                'mae': np.mean(np.abs(np.array(self.predicted_prices) - 
                                    np.array(self.true_prices))),
                'mape': np.mean(np.array(self.prediction_errors)[:, 0])
            },
            'market_efficiency': {
                'score': np.mean(self.market_efficiency_scores),
                'liquidity': np.mean([m['depth'] for m in self.liquidity_metrics]),
                'volatility': np.mean(self.volatility_measures)
            }
        }

class SystemPerformanceMetrics:
    """
    System performance metrics tracking for comprehensive evaluation of
    computational efficiency and resource utilization.
    """
    
    def __init__(self):
        """
        Initialize system performance monitoring with comprehensive
        resource tracking capabilities.
        """
        self.reset()
        self._setup_monitoring()
        
    def reset(self):
        """Reset all system performance metrics storage."""
        self.processing_times = []
        self.memory_usage = []
        self.gpu_utilization = []
        self.batch_sizes = []
        self.throughput_history = []
        
    def _setup_monitoring(self):
        """Configure system monitoring tools and intervals."""
        self.start_time = time.time()
        self.total_samples = 0
        self.peak_memory = 0
        
        # Initialize GPU monitoring if available
        if torch.cuda.is_available():
            self.gpu_handles = [
                torch.cuda.device(i) 
                for i in range(torch.cuda.device_count())
            ]
            
    def update(self, batch_metrics: Dict[str, float]):
        """
        Update system performance metrics with new batch results.
        
        Args:
            batch_metrics: Dictionary containing batch performance metrics
        """
        # Update processing time metrics
        self.processing_times.append(batch_metrics['processing_time'])
        
        # Update memory usage metrics
        current_memory = self._get_memory_usage()
        self.memory_usage.append(current_memory)
        self.peak_memory = max(self.peak_memory, current_memory)
        
        # Update GPU utilization metrics
        if torch.cuda.is_available():
            self.gpu_utilization.append(self._get_gpu_utilization())
            
        # Update throughput metrics
        batch_size = batch_metrics.get('batch_size', 0)
        self.batch_sizes.append(batch_size)
        self.total_samples += batch_size
        
        # Calculate current throughput
        elapsed_time = time.time() - self.start_time
        self.throughput_history.append(self.total_samples / elapsed_time)
        
    def _get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics for both CPU and GPU.
        
        Returns:
            Dictionary containing memory usage metrics
        """
        memory_stats = {
            'cpu': {
                'used': psutil.Process().memory_info().rss / 1024**3,
                'total': psutil.virtual_memory().total / 1024**3,
                'percent': psutil.virtual_memory().percent
            }
        }
        
        if torch.cuda.is_available():
            for i, handle in enumerate(self.gpu_handles):
                memory_stats[f'gpu_{i}'] = {
                    'used': torch.cuda.memory_allocated(i) / 1024**3,
                    'cached': torch.cuda.memory_reserved(i) / 1024**3
                }
                
        return memory_stats
        
    def _get_gpu_utilization(self) -> Dict[str, float]:
        """
        Get current GPU utilization metrics.
        
        Returns:
            Dictionary containing GPU utilization metrics
        """
        gpu_stats = {}
        for i, gpu in enumerate(GPUtil.getGPUs()):
            gpu_stats[f'gpu_{i}'] = {
                'utilization': gpu.load * 100,
                'temperature': gpu.temperature,
                'power_usage': gpu.powerUsage if hasattr(gpu, 'powerUsage') else None
            }
            
        return gpu_stats
        
    def compute(self) -> Dict[str, float]:
        """
        Compute final system performance metrics.
        
        Returns:
            Dictionary containing all computed system metrics
        """
        return {
            'processing_efficiency': {
                'avg_time': np.mean(self.processing_times),
                'std_time': np.std(self.processing_times),
                'throughput': self.throughput_history[-1]
            },
            'memory_efficiency': {
                'peak_memory': self.peak_memory,
                'avg_memory': np.mean([m['cpu']['used'] for m in self.memory_usage])
            },
            'gpu_efficiency': {
                'avg_utilization': np.mean([
                    u['gpu_0']['utilization'] 
                    for u in self.gpu_utilization
                ]),
                'peak_temperature': max([
                    u['gpu_0']['temperature'] 
                    for u in self.gpu_utilization
                ])
            }
        }

def create_evaluator(
    model: nn.Module,
    val_loader: DataLoader,
    num_gpus: int = 3,
    output_dir: str = "evaluation_results"
) -> ModelEvaluator:
    """
    Create and configure a ModelEvaluator instance with optimal settings.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        num_gpus: Number of available GPUs
        output_dir: Directory for saving evaluation results
        
    Returns:
        Configured ModelEvaluator instance
    """
    return ModelEvaluator(
        model=model,
        val_loader=val_loader,
        num_gpus=num_gpus,
        output_dir=output_dir
    )