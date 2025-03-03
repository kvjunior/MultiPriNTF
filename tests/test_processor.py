"""
MultiPriNTF Processor Test Suite
=========================

This module implements comprehensive testing for the MultiPriNTF multimedia
processing system, including:
1. Batch processing validation
2. Multi-GPU optimization testing
3. Memory efficiency verification
4. Stream processing validation
5. Error handling testing
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Dict, List, Optional

from src.models.multimedia_processor import (
    OptimizedMultimediaProcessor,
    OptimizedStreamBuffer,
    OptimizedRealTimeProcessor
)

class TestMultimediaProcessor(unittest.TestCase):
    """
    Comprehensive test suite for the multimedia processing system.
    """
    
    @classmethod
    def setUpClass(cls):
        """Initialize test environment with GPU support."""
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.num_gpus = torch.cuda.device_count()
        cls.batch_size = 32
        cls.feature_dim = 1024
        cls.buffer_size = 2000
        
        # Create temporary directory for test data
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls._setup_test_data()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.temp_dir)
        torch.cuda.empty_cache()
        
    @classmethod
    def _setup_test_data(cls):
        """Generate synthetic test data."""
        cls.test_data = {
            'visual': torch.randn(cls.batch_size, 3, 224, 224),
            'transaction': torch.randn(cls.batch_size, cls.feature_dim)
        }
        
        # Save test data
        torch.save(cls.test_data, cls.temp_dir / 'test_data.pt')
        
    def setUp(self):
        """Prepare for each test."""
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        
        # Initialize processor
        self.processor = OptimizedMultimediaProcessor(
            batch_size=self.batch_size,
            buffer_size=self.buffer_size,
            num_gpus=self.num_gpus,
            feature_dim=self.feature_dim
        )
        
    def test_processor_initialization(self):
        """Test processor initialization and configuration."""
        self.assertEqual(self.processor.batch_size, self.batch_size)
        self.assertEqual(self.processor.num_gpus, self.num_gpus)
        self.assertEqual(len(self.processor.streams), self.num_gpus)
        self.assertEqual(len(self.processor.processing_queues), self.num_gpus)
        
    def test_batch_processing(self):
        """Test batch processing capabilities."""
        self.processor.start()
        
        try:
            # Submit test batch
            batch = self._create_test_batch()
            gpu_id = 0
            
            processed_batch = self.processor._batch_process(batch, gpu_id)
            
            self.assertEqual(len(processed_batch), len(batch))
            self.assertIn('visual_features', processed_batch[0])
            self.assertIn('transaction_features', processed_batch[0])
            
        finally:
            self.processor.stop()
            
    def test_multi_gpu_processing(self):
        """Test processing distribution across multiple GPUs."""
        if self.num_gpus < 2:
            self.skipTest("Need at least 2 GPUs for this test")
            
        self.processor.start()
        
        try:
            # Submit batches to different GPUs
            results = []
            for gpu_id in range(self.num_gpus):
                batch = self._create_test_batch()
                result = self.processor._batch_process(batch, gpu_id)
                results.append(result)
                
            # Verify results from each GPU
            for result in results:
                self.assertTrue(all('visual_features' in item for item in result))
                self.assertTrue(
                    all('transaction_features' in item for item in result)
                )
                
        finally:
            self.processor.stop()
            
    def test_stream_buffer(self):
        """Test stream buffer functionality."""
        buffer = OptimizedStreamBuffer(max_size=self.buffer_size)
        
        # Test addition and retrieval
        test_item = {'id': 1, 'data': torch.randn(224, 224)}
        self.assertTrue(buffer.add_item(test_item))
        
        retrieved_item = buffer.get_item()
        self.assertIsNotNone(retrieved_item)
        self.assertEqual(retrieved_item['id'], test_item['id'])
        
        # Test buffer capacity
        for i in range(self.buffer_size + 1):
            success = buffer.add_item({'id': i})
            if i < self.buffer_size:
                self.assertTrue(success)
            else:
                self.assertFalse(success)
                
    def test_real_time_processor(self):
        """Test real-time processing capabilities."""
        # Create dummy model
        model = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU()
        ).to(self.device)
        
        processor = OptimizedRealTimeProcessor(
            model=model,
            buffer_size=self.buffer_size,
            num_gpus=self.num_gpus
        )
        
        # Test stream processing
        test_stream = self._create_test_stream()
        processor.process_stream(test_stream)
        
        # Check results
        results = processor.get_performance_metrics()
        self.assertIn('monitor_metrics', results)
        self.assertIn('input_buffer_metrics', results)
        self.assertIn('output_buffer_metrics', results)
        
    def test_error_handling(self):
        """Test error handling and recovery."""
        self.processor.start()
        
        try:
            # Test invalid batch
            invalid_batch = [{'invalid_data': None}]
            with self.assertRaises(Exception):
                self.processor._batch_process(invalid_batch, 0)
                
            # Test recovery after error
            valid_batch = self._create_test_batch()
            processed_batch = self.processor._batch_process(valid_batch, 0)
            self.assertIsNotNone(processed_batch)
            
        finally:
            self.processor.stop()
            
    def test_memory_management(self):
        """Test memory management efficiency."""
        initial_memory = torch.cuda.memory_allocated()
        
        self.processor.start()
        
        try:
            # Process multiple batches
            for _ in range(5):
                batch = self._create_test_batch()
                self.processor._batch_process(batch, 0)
                
            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory
            
            # Check memory efficiency
            memory_threshold = 2 * 1024 * 1024 * 1024  # 2GB
            self.assertLess(memory_increase, memory_threshold)
            
        finally:
            self.processor.stop()
            
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities."""
        self.processor.start()
        
        try:
            # Process batch and check metrics
            batch = self._create_test_batch()
            start_time = time.time()
            
            processed_batch = self.processor._batch_process(batch, 0)
            processing_time = time.time() - start_time
            
            # Verify monitoring metrics
            metrics = self.processor.get_performance_metrics()
            self.assertIn('throughput', metrics)
            self.assertIn('processing_time', metrics)
            self.assertGreater(metrics['throughput'], 0)
            self.assertLess(metrics['processing_time'], 1.0)
            
        finally:
            self.processor.stop()
            
    def _create_test_batch(self) -> List[Dict]:
        """Create synthetic test batch."""
        return [
            {
                'id': i,
                'visual': torch.randn(3, 224, 224),
                'transaction': torch.randn(self.feature_dim)
            }
            for i in range(self.batch_size)
        ]
        
    def _create_test_stream(self) -> List[Dict]:
        """Create synthetic test stream."""
        return [
            {
                'id': i,
                'data': torch.randn(self.feature_dim)
            }
            for i in range(100)
        ]

class TestProcessorIntegration(unittest.TestCase):
    """
    Integration tests for the multimedia processor.
    """
    
    def setUp(self):
        """Set up integration test environment."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count()
        self.batch_size = 32
        self.feature_dim = 1024
        
        # Create test model
        self.model = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU()
        ).to(self.device)
        
        # Initialize processor
        self.processor = OptimizedRealTimeProcessor(
            model=self.model,
            buffer_size=2000,
            num_gpus=self.num_gpus
        )
        
    def test_end_to_end_processing(self):
        """Test end-to-end processing pipeline."""
        # Create test data
        test_stream = [
            {
                'id': i,
                'data': torch.randn(self.feature_dim)
            }
            for i in range(100)
        ]
        
        # Process stream
        self.processor.process_stream(test_stream)
        
        # Verify results
        metrics = self.processor.get_performance_metrics()
        self.assertGreater(metrics['monitor_metrics']['total_processed'], 0)
        self.assertLess(metrics['monitor_metrics']['drop_rate'], 0.1)
        
    def test_concurrent_processing(self):
        """Test concurrent processing capabilities."""
        def process_stream(stream_id: int):
            stream = [
                {
                    'id': f'{stream_id}_{i}',
                    'data': torch.randn(self.feature_dim)
                }
                for i in range(50)
            ]
            self.processor.process_stream(stream)
            
        # Create multiple processing threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_stream, i)
                for i in range(4)
            ]
            
        # Wait for completion and check results
        for future in futures:
            future.result()
            
        metrics = self.processor.get_performance_metrics()
        self.assertGreater(metrics['monitor_metrics']['total_processed'], 150)

def run_tests():
    """Run test suite."""
    unittest.main()

if __name__ == '__main__':
    run_tests()