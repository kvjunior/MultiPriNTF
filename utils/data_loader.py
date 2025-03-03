"""
MultiPriNTF Data Loading System
=========================

This module implements an optimized data loading system for the MultiPriNTF architecture,
specifically designed for efficient multimedia processing of NFT images and
transaction data on multi-GPU systems.

Key Features:
1. Memory-mapped data handling for large datasets
2. Parallel data loading with GPU streams
3. Advanced caching mechanisms
4. Robust error handling and recovery
5. Real-time data augmentation
6. Efficient multi-modal data integration

Technical Implementation:
The system uses a hybrid approach combining memory mapping for large files with
GPU-specific optimizations. Data is processed in parallel streams and cached
efficiently to minimize I/O overhead.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import h5py
import threading
from queue import Queue
import mmap
import cv2
from pathlib import Path
import psutil
from concurrent.futures import ThreadPoolExecutor

class OptimizedNFTDataset(Dataset):
    """
    Optimized dataset implementation for handling NFT data with efficient memory
    usage and multi-GPU support. Implements memory mapping and parallel processing
    for large-scale data handling.
    """
    
    def __init__(
        self,
        image_dir: str,
        transaction_file: str,
        transform: Optional[transforms.Compose] = None,
        cache_size: int = 1000,
        num_gpus: int = 3,
        prefetch_factor: int = 2
    ):
        """
        Initialize the dataset with optimized memory handling.

        Args:
            image_dir: Directory containing NFT images
            transaction_file: Path to transaction data file
            transform: Optional data transformations
            cache_size: Size of memory cache
            num_gpus: Number of available GPUs
            prefetch_factor: Number of batches to prefetch
        """
        super().__init__()
        self.image_dir = Path(image_dir)
        self.transform = transform or self._get_default_transform()
        self.cache_size = cache_size
        self.num_gpus = num_gpus
        self.prefetch_factor = prefetch_factor
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize memory-mapped cache
        self.image_cache = self._initialize_cache()
        
        # Load and preprocess transaction data
        self.transactions = self._load_transactions(transaction_file)
        
        # Initialize mappings
        self.type_to_idx = self._create_type_mapping()
        self.attribute_mappings = self._create_attribute_mappings()
        
        # Setup multi-GPU processing
        self._setup_gpu_processing()
        
        # Initialize monitoring
        self.stats = self._initialize_statistics()
        
    def _setup_logging(self):
        """Configure logging system for data loading operations."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler('data_loading.log')
        console_handler = logging.StreamHandler()
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _get_default_transform(self) -> transforms.Compose:
        """
        Create default transformation pipeline optimized for NFT images.
        
        Returns:
            Composed transformation pipeline
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
        ])

    def _initialize_cache(self) -> h5py.File:
        """
        Initialize memory-mapped cache for image data.
        
        Returns:
            HDF5 file object for caching
        """
        cache_path = self.image_dir / 'image_cache.h5'
        
        try:
            return h5py.File(cache_path, 'a')
        except Exception as e:
            self.logger.error(f"Failed to initialize cache: {e}")
            raise

    def _load_transactions(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess transaction data with memory optimization.
        
        Args:
            file_path: Path to transaction data file
            
        Returns:
            Preprocessed transaction DataFrame
        """
        try:
            # Use memory-efficient JSON loading
            transactions = []
            with open(file_path, 'r') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                for line in iter(mm.readline, b''):
                    try:
                        transaction = json.loads(line.decode())
                        transactions.append(transaction)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Skipping malformed JSON line: {e}")
                        continue

            df = pd.DataFrame(transactions)
            
            # Optimize DataFrame memory usage
            df = self._optimize_dataframe(df)
            
            self.logger.info(
                f"Loaded {len(df)} transactions with {df.memory_usage().sum() / 1e6:.2f}MB memory usage"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load transactions: {e}")
            raise

    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage through type conversion.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Memory-optimized DataFrame
        """
        # Optimize numeric columns
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')
                
        # Optimize categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Low cardinality
                df[col] = df[col].astype('category')
                
        return df

    def _create_type_mapping(self) -> Dict[str, int]:
        """
        Create mapping for NFT types.
        
        Returns:
            Dictionary mapping NFT types to indices
        """
        unique_types = self.transactions['type'].unique()
        return {t: i for i, t in enumerate(sorted(unique_types))}

    def _create_attribute_mappings(self) -> Dict[str, int]:
        """
        Create mapping for NFT attributes.
        
        Returns:
            Dictionary mapping attributes to indices
        """
        unique_attributes = set()
        for attrs in self.transactions['accessories']:
            if isinstance(attrs, list):
                unique_attributes.update(attrs)
                
        return {attr: idx for idx, attr in enumerate(sorted(unique_attributes))}

    def _setup_gpu_processing(self):
        """Configure multi-GPU processing environment."""
        if torch.cuda.is_available():
            # Initialize GPU queues
            self.gpu_queues = [
                Queue(maxsize=self.cache_size) for _ in range(self.num_gpus)
            ]
            
            # Create CUDA streams
            self.streams = [
                torch.cuda.Stream() for _ in range(self.num_gpus)
            ]
            
            # Start prefetch thread
            self.prefetch_thread = threading.Thread(
                target=self._prefetch_data,
                daemon=True
            )
            self.prefetch_thread.start()

    def _initialize_statistics(self) -> Dict:
        """
        Initialize statistics tracking.
        
        Returns:
            Dictionary for tracking statistics
        """
        return {
            'loads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': [],
            'memory_usage': []
        }

def _prefetch_data(self):
        """
        Prefetch data in background thread for efficient GPU processing.
        This method implements advanced prefetching with error handling
        and memory optimization.
        """
        while True:
            try:
                # Generate randomized indices for better training distribution
                indices = np.random.permutation(len(self))
                
                for idx in indices:
                    # Load and preprocess image
                    image_path = self.image_dir / f"{idx}.png"
                    try:
                        image = self._load_image(image_path)
                        if image is not None:
                            # Distribute across GPU queues with load balancing
                            gpu_id = self._get_optimal_gpu()
                            self.gpu_queues[gpu_id].put(
                                (idx, image),
                                timeout=1.0
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Error prefetching image {idx}: {e}"
                        )
                        continue
                        
            except Exception as e:
                self.logger.error(f"Prefetch thread error: {e}")
                time.sleep(1.0)  # Prevent tight error loops

    def _get_optimal_gpu(self) -> int:
        """
        Determine optimal GPU for next batch based on current load.
        Uses memory utilization and queue sizes to balance load.
        
        Returns:
            Index of optimal GPU for next batch
        """
        if not torch.cuda.is_available():
            return 0
            
        # Calculate load scores for each GPU
        scores = []
        for gpu_id in range(self.num_gpus):
            memory_used = torch.cuda.memory_allocated(gpu_id) / \
                         torch.cuda.get_device_properties(gpu_id).total_memory
            queue_size = self.gpu_queues[gpu_id].qsize()
            
            # Combined score (lower is better)
            score = memory_used + queue_size / self.cache_size
            scores.append(score)
            
        return np.argmin(scores)

    def _load_image(self, image_path: Path) -> Optional[torch.Tensor]:
        """
        Load and preprocess image with caching and error handling.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor or None if loading fails
        """
        try:
            # Check cache first
            cache_key = str(image_path)
            if cache_key in self.image_cache:
                self.stats['cache_hits'] += 1
                image_data = self.image_cache[cache_key][:]
                image = Image.fromarray(image_data)
            else:
                self.stats['cache_misses'] += 1
                # Load image with OpenCV for better performance
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                
                # Cache the image
                self.image_cache.create_dataset(
                    cache_key,
                    data=np.array(image),
                    compression="lzf"
                )

            # Apply transformations
            if self.transform:
                image = self.transform(image)
                
            self.stats['loads'] += 1
            return image

        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return None

    def __len__(self) -> int:
        """
        Get dataset length.
        
        Returns:
            Number of items in dataset
        """
        return len(self.transactions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get dataset item with optimized processing.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (image, type_idx, attributes, price) tensors
        """
        start_time = time.time()
        
        try:
            # Get image data
            image_path = self.image_dir / f"{idx}.png"
            image = self._load_image(image_path)
            
            if image is None:
                raise ValueError(f"Failed to load image {idx}")
            
            # Get transaction data
            transaction = self.transactions.iloc[idx]
            
            # Process NFT type
            punk_type = transaction['type']
            if isinstance(punk_type, list):
                punk_type = punk_type[0]
            type_idx = torch.tensor(self.type_to_idx[punk_type])
            
            # Process attributes with optimized memory usage
            attributes = torch.zeros(len(self.attribute_mappings))
            if isinstance(transaction['accessories'], list):
                for acc in transaction['accessories']:
                    if acc in self.attribute_mappings:
                        attr_idx = self.attribute_mappings[acc]
                        attributes[attr_idx] = 1
            
            # Process price with proper scaling
            price = torch.tensor(
                float(transaction['eth']) if 'eth' in transaction else 0.0,
                dtype=torch.float32
            )
            
            # Update processing statistics
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            self.stats['memory_usage'].append(
                psutil.Process().memory_info().rss / 1024**2
            )
            
            return image, type_idx, attributes, price
            
        except Exception as e:
            self.logger.error(f"Error getting item {idx}: {e}")
            # Return zero tensors as fallback
            return (
                torch.zeros(3, 224, 224),
                torch.tensor(0),
                torch.zeros(len(self.attribute_mappings)),
                torch.tensor(0.0)
            )

class OptimizedDataLoader:
    """
    Optimized data loader with multi-GPU support and efficient batching.
    Implements advanced features for distributed training and memory
    optimization.
    """
    
    def __init__(
        self,
        dataset: OptimizedNFTDataset,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        num_gpus: int = 3,
        prefetch_factor: int = 2
    ):
        """
        Initialize optimized data loader.
        
        Args:
            dataset: Dataset instance
            batch_size: Batch size
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            num_gpus: Number of GPUs
            prefetch_factor: Number of batches to prefetch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_gpus = num_gpus
        self.prefetch_factor = prefetch_factor
        
        # Calculate optimal batch size per GPU
        self.batch_size_per_gpu = batch_size // num_gpus
        
        # Initialize samplers for multi-GPU training
        self.train_sampler = None
        self.val_sampler = None
        
        # Initialize worker pool
        self.worker_pool = ThreadPoolExecutor(
            max_workers=self.num_workers
        )
        
    def setup_distributed(self, world_size: int, rank: int):
        """
        Setup distributed training configuration.
        
        Args:
            world_size: Total number of processes
            rank: Process rank
        """
        self.train_sampler = DistributedSampler(
            self.dataset,
            num_replicas=world_size,
            rank=rank
        )
        
    def get_train_loader(self) -> DataLoader:
        """
        Get training data loader with optimized configuration.
        
        Returns:
            Configured DataLoader instance
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            shuffle=(self.train_sampler is None),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=self.train_sampler,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            drop_last=False
        )
        
    def get_val_loader(self, val_dataset: OptimizedNFTDataset) -> DataLoader:
        """
        Get validation data loader.
        
        Args:
            val_dataset: Validation dataset
            
        Returns:
            Configured DataLoader instance
        """
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_gpu,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=self.val_sampler,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True
        )

def create_data_loaders(
    image_dir: str,
    transaction_file: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_gpus: int = 3,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders with optimal configuration.
    
    Args:
        image_dir: Directory containing images
        transaction_file: Path to transaction data
        batch_size: Batch size
        val_split: Validation split ratio
        num_gpus: Number of GPUs
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset with optimized transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    dataset = OptimizedNFTDataset(
        image_dir=image_dir,
        transaction_file=transaction_file,
        transform=transform,
        num_gpus=num_gpus
    )
    
    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )
    
    # Create data loaders
    loader = OptimizedDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        num_gpus=num_gpus
    )
    
    train_loader = loader.get_train_loader()
    val_loader = loader.get_val_loader(val_dataset)
    
    return train_loader, val_loader