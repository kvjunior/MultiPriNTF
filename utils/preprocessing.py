"""
MultiPriNTF Preprocessing System
=========================

This module implements advanced preprocessing for NFT data, handling both
image and transaction data with optimized memory usage and parallel processing.

Key Features:
1. Memory-mapped data processing
2. Parallel processing with GPU acceleration
3. Efficient caching mechanisms
4. Robust error handling
5. Transaction data normalization
6. Image preprocessing optimization
7. Real-time data augmentation

Technical Architecture:
The system uses a hybrid approach combining memory mapping for large datasets
with parallel processing capabilities, optimized for the MultiPriNTF architecture.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import h5py
import json
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import mmap
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Optimized data preprocessor for NFT images and transaction data.
    Implements efficient parallel processing and memory management.
    """
    
    def __init__(
        self,
        image_dir: str,
        transaction_file: str,
        output_dir: str,
        num_gpus: int = 3,
        batch_size: int = 128,
        image_size: Tuple[int, int] = (224, 224),
        cache_size: int = 1000
    ):
        """
        Initialize the preprocessor with optimized configuration.

        Args:
            image_dir: Directory containing NFT images
            transaction_file: Path to transaction data file
            output_dir: Output directory for processed data
            num_gpus: Number of available GPUs
            batch_size: Processing batch size
            image_size: Target image size
            cache_size: Size of memory cache
        """
        self.image_dir = Path(image_dir)
        self.transaction_file = Path(transaction_file)
        self.output_dir = Path(output_dir)
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.image_size = image_size
        self.cache_size = cache_size
        
        # Initialize processing queues
        self.image_queue = Queue(maxsize=cache_size)
        self.transaction_queue = Queue(maxsize=cache_size)
        
        # Initialize GPU streams if available
        if torch.cuda.is_available():
            self.streams = [
                torch.cuda.Stream() for _ in range(num_gpus)
            ]
        
        # Initialize data scalers
        self.price_scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
        # Initialize storage
        self.storage = self._initialize_storage()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize processing pools
        self._initialize_processing_pools()
        
    def _setup_logging(self):
        """Configure comprehensive logging system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'preprocessing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_storage(self) -> h5py.File:
        """
        Initialize HDF5 storage for processed data.

        Returns:
            HDF5 file object
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return h5py.File(self.output_dir / 'processed_data.h5', 'a')

    def _initialize_processing_pools(self):
        """Initialize parallel processing pools."""
        num_cpus = mp.cpu_count()
        self.thread_pool = ThreadPoolExecutor(
            max_workers=num_cpus * 2
        )
        self.process_pool = ProcessPoolExecutor(
            max_workers=num_cpus
        )

    def preprocess_images(self):
        """Process images with GPU acceleration and parallel processing."""
        self.logger.info("Starting image preprocessing...")
        
        try:
            # List all image files
            image_files = list(self.image_dir.glob('*.png'))
            total_images = len(image_files)
            
            # Process images in parallel
            with tqdm(total=total_images, desc="Processing images") as pbar:
                futures = []
                
                for i in range(0, total_images, self.batch_size):
                    batch_files = image_files[i:i + self.batch_size]
                    future = self.process_pool.submit(
                        self._process_image_batch,
                        batch_files
                    )
                    futures.append(future)
                    
                # Process results
                for future in futures:
                    try:
                        processed_batch = future.result()
                        self._store_processed_images(processed_batch)
                        pbar.update(len(processed_batch))
                    except Exception as e:
                        self.logger.error(f"Error processing image batch: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error in image preprocessing: {e}")
            raise

    def _process_image_batch(
        self,
        image_files: List[Path]
    ) -> Dict[str, np.ndarray]:
        """
        Process a batch of images with optimized memory usage.

        Args:
            image_files: List of image file paths

        Returns:
            Dictionary of processed images
        """
        processed_batch = {}
        
        for img_path in image_files:
            try:
                # Load and preprocess image
                img = self._load_and_preprocess_image(img_path)
                if img is not None:
                    processed_batch[img_path.stem] = img
                    
            except Exception as e:
                self.logger.error(f"Error processing image {img_path}: {e}")
                continue
                
        return processed_batch

    def _load_and_preprocess_image(
        self,
        image_path: Path
    ) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image with optimizations.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image array or None if processing fails
        """
        try:
            # Load image with OpenCV for better performance
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
                
            # Convert to RGB and resize
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size)
            
            # Normalize and convert to float32
            img = img.astype(np.float32) / 255.0
            
            # Apply data augmentation if needed
            img = self._apply_augmentation(img)
            
            return img
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return None

def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply real-time data augmentation to images.

        Args:
            image: Input image array

        Returns:
            Augmented image array
        """
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = np.fliplr(image)
            
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness_factor, 0, 1)
        
        # Random contrast adjustment
        contrast_factor = np.random.uniform(0.8, 1.2)
        mean = image.mean(axis=(0, 1), keepdims=True)
        image = np.clip((image - mean) * contrast_factor + mean, 0, 1)
        
        return image

    def _store_processed_images(self, processed_batch: Dict[str, np.ndarray]):
        """
        Store processed images in HDF5 format with compression.

        Args:
            processed_batch: Dictionary of processed images
        """
        with threading.Lock():
            for img_id, img_data in processed_batch.items():
                self.storage.create_dataset(
                    f'images/{img_id}',
                    data=img_data,
                    compression='lzf'
                )

    def preprocess_transactions(self):
        """Process transaction data with optimized memory usage."""
        self.logger.info("Starting transaction preprocessing...")
        
        try:
            # Load transactions efficiently
            transactions = self._load_transactions()
            
            # Convert to DataFrame with optimized memory usage
            df = pd.DataFrame(transactions)
            df = self._optimize_dataframe(df)
            
            # Process features
            df = self._process_numeric_features(df)
            df = self._process_categorical_features(df)
            df = self._add_temporal_features(df)
            
            # Store processed transactions
            self._store_processed_transactions(df)
            
        except Exception as e:
            self.logger.error(f"Error processing transactions: {e}")
            raise

    def _load_transactions(self) -> List[Dict]:
        """
        Load transaction data with memory-efficient streaming.

        Returns:
            List of transaction dictionaries
        """
        transactions = []
        
        try:
            with open(self.transaction_file, 'r') as f:
                # Memory-map the file for efficient reading
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                
                for line in tqdm(
                    iter(mm.readline, b''),
                    desc="Loading transactions"
                ):
                    try:
                        transaction = json.loads(line.decode())
                        transactions.append(transaction)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Skipping malformed JSON line: {e}")
                        continue
                        
            return transactions
            
        except Exception as e:
            self.logger.error(f"Error loading transactions: {e}")
            raise

    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage through type optimization.

        Args:
            df: Input DataFrame

        Returns:
            Optimized DataFrame
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
                
        # Report memory usage
        memory_usage = df.memory_usage().sum() / 1024**2
        self.logger.info(f"DataFrame memory usage: {memory_usage:.2f} MB")
        
        return df

    def _process_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process numeric features with scaling and normalization.

        Args:
            df: Input DataFrame

        Returns:
            Processed DataFrame
        """
        numeric_columns = df.select_dtypes(
            include=['float32', 'int32']
        ).columns
        
        for col in numeric_columns:
            # Handle missing values
            df[col] = df[col].fillna(df[col].median())
            
            # Scale numeric features
            if col == 'eth':
                df[col] = self.price_scaler.fit_transform(
                    df[col].values.reshape(-1, 1)
                )
            else:
                df[col] = self.feature_scaler.fit_transform(
                    df[col].values.reshape(-1, 1)
                )
                
        return df

    def _process_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process categorical features with efficient encoding.

        Args:
            df: Input DataFrame

        Returns:
            Processed DataFrame
        """
        categorical_columns = df.select_dtypes(
            include=['category', 'object']
        ).columns
        
        for col in categorical_columns:
            if col == 'type':
                # One-hot encoding for NFT type
                type_dummies = pd.get_dummies(df[col], prefix='type')
                df = pd.concat([df, type_dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
                
            elif col == 'accessories':
                # Multi-label encoding for accessories
                unique_accessories = set()
                for accessories in df[col]:
                    if isinstance(accessories, list):
                        unique_accessories.update(accessories)
                        
                for accessory in unique_accessories:
                    df[f'accessory_{accessory}'] = df[col].apply(
                        lambda x: 1 if isinstance(x, list) and accessory in x else 0
                    )
                    
                df.drop(col, axis=1, inplace=True)
                
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features for time-series analysis.

        Args:
            df: Input DataFrame

        Returns:
            Enhanced DataFrame with temporal features
        """
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Calculate time-based statistics
        df['daily_volume'] = df.groupby(
            df['timestamp'].dt.date
        )['eth'].transform('sum')
        
        df['hourly_transactions'] = df.groupby([
            df['timestamp'].dt.date,
            df['timestamp'].dt.hour
        ]).transform('count')
        
        # Calculate moving averages
        df['price_ma_7d'] = df.groupby('type')['eth'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        return df

    def _store_processed_transactions(self, df: pd.DataFrame):
        """
        Store processed transactions with efficient compression.

        Args:
            df: Processed DataFrame
        """
        # Store processed DataFrame
        with threading.Lock():
            for column in df.columns:
                self.storage.create_dataset(
                    f'transactions/{column}',
                    data=df[column].values,
                    compression='lzf'
                )
                
            # Store feature metadata
            metadata = {
                'numeric_features': list(
                    df.select_dtypes(include=['float32', 'int32']).columns
                ),
                'categorical_features': list(
                    df.select_dtypes(include=['category']).columns
                ),
                'temporal_features': [
                    'hour', 'day', 'day_of_week', 'month'
                ],
                'price_scaler_params': {
                    'mean': self.price_scaler.mean_.tolist(),
                    'scale': self.price_scaler.scale_.tolist()
                }
            }
            
            self.storage.create_dataset(
                'metadata',
                data=json.dumps(metadata).encode('utf-8')
            )

    def process_all(self):
        """Process all data with parallel execution."""
        self.logger.info("Starting complete data preprocessing...")
        
        try:
            # Process images and transactions in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                image_future = executor.submit(self.preprocess_images)
                transaction_future = executor.submit(self.preprocess_transactions)
                
                # Wait for completion
                image_future.result()
                transaction_future.result()
                
            self.logger.info("Data preprocessing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            raise
        finally:
            self.storage.close()
            self._cleanup()
            
    def _cleanup(self):
        """Clean up resources and temporary files."""
        try:
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Close processing pools
            self.thread_pool.shutdown()
            self.process_pool.shutdown()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

def create_preprocessor(
    image_dir: str,
    transaction_file: str,
    output_dir: str,
    num_gpus: int = 3,
    batch_size: int = 128
) -> DataPreprocessor:
    """
    Create and configure a data preprocessor instance.
    
    Args:
        image_dir: Directory containing images
        transaction_file: Path to transaction file
        output_dir: Output directory
        num_gpus: Number of GPUs to use
        batch_size: Processing batch size
        
    Returns:
        Configured DataPreprocessor instance
    """
    return DataPreprocessor(
        image_dir=image_dir,
        transaction_file=transaction_file,
        output_dir=output_dir,
        num_gpus=num_gpus,
        batch_size=batch_size
    )

if __name__ == "__main__":
    # Example usage
    preprocessor = create_preprocessor(
        image_dir="data/raw/images",
        transaction_file="data/raw/transactions.jsonl",
        output_dir="data/processed",
        num_gpus=3
    )
    preprocessor.process_all()