"""
MultiPriNTF Multimodal Architecture Test Suite
======================================

This module implements comprehensive testing for the MultiPriNTF multimodal
architecture components, including:
1. Cross-modal attention testing
2. Multimodal fusion validation
3. Memory efficiency verification
4. GPU optimization testing
5. Error handling validation
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil
import logging
from typing import Dict, Tuple
import h5py
import warnings

from src.models.multimodal_architecture import (
    CrossModalAttention,
    MultimodalFusionEncoder,
    OptimizedResNetBlock
)
from src.utils.data_loader import OptimizedNFTDataset
from src.models.security_module import SecureMultimodalFusion

class TestMultimodalArchitecture(unittest.TestCase):
    """
    Test suite for multimodal architecture components.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with GPU support."""
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.batch_size = 32
        cls.input_dim = 512
        cls.num_heads = 8
        
        # Create temporary directory for test data
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls._setup_test_data()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore')
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.temp_dir)
        torch.cuda.empty_cache()
        
    @classmethod
    def _setup_test_data(cls):
        """Create synthetic test data."""
        # Create test images
        cls.test_images = torch.randn(
            cls.batch_size, 3, 224, 224,
            device=cls.device
        )
        
        # Create test transaction data
        cls.test_transactions = torch.randn(
            cls.batch_size,
            cls.input_dim,
            device=cls.device
        )
        
        # Save test data
        with h5py.File(cls.temp_dir / 'test_data.h5', 'w') as f:
            f.create_dataset('images', data=cls.test_images.cpu().numpy())
            f.create_dataset(
                'transactions',
                data=cls.test_transactions.cpu().numpy()
            )
    
    def setUp(self):
        """Set up each test."""
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        
    def test_cross_modal_attention(self):
        """Test CrossModalAttention module."""
        attention = CrossModalAttention(
            dim=self.input_dim,
            num_heads=self.num_heads
        ).to(self.device)
        
        # Test forward pass
        output = attention(self.test_images, self.test_transactions)
        
        self.assertEqual(output.shape, self.test_images.shape)
        self.assertFalse(torch.isnan(output).any())
        
        # Test attention weights
        with torch.no_grad():
            _, weights = attention.get_attention_weights(
                self.test_images,
                self.test_transactions
            )
            self.assertEqual(
                weights.shape,
                (self.batch_size, self.num_heads, 224, 224)
            )
        
    def test_multimodal_fusion_encoder(self):
        """Test MultimodalFusionEncoder module."""
        encoder = MultimodalFusionEncoder(
            visual_dim=512,
            transaction_dim=self.input_dim,
            fusion_dim=512
        ).to(self.device)
        
        # Test forward pass
        fused_output, visual_features, transaction_features = encoder(
            self.test_images,
            self.test_transactions
        )
        
        self.assertEqual(fused_output.shape, (self.batch_size, 512))
        self.assertFalse(torch.isnan(fused_output).any())
        
        # Test feature extraction
        self.assertEqual(visual_features.shape, (self.batch_size, 512))
        self.assertEqual(
            transaction_features.shape,
            (self.batch_size, 512)
        )
        
    def test_resnet_block(self):
        """Test OptimizedResNetBlock module."""
        block = OptimizedResNetBlock(64, 128, stride=2).to(self.device)
        input_tensor = torch.randn(
            self.batch_size, 64, 56, 56,
            device=self.device
        )
        
        # Test forward pass
        output = block(input_tensor)
        
        self.assertEqual(output.shape, (self.batch_size, 128, 28, 28))
        self.assertFalse(torch.isnan(output).any())
        
    def test_secure_multimodal_fusion(self):
        """Test SecureMultimodalFusion module."""
        secure_fusion = SecureMultimodalFusion(
            visual_dim=512,
            transaction_dim=self.input_dim,
            output_dim=512
        ).to(self.device)
        
        # Test forward pass with encryption
        outputs = secure_fusion(self.test_images, self.test_transactions)
        
        self.assertIn('fused_features', outputs)
        self.assertIn('encrypted_visual', outputs)
        self.assertIn('encrypted_transaction', outputs)
        
        # Test feature reconstruction
        decrypted_features = secure_fusion.decrypt_features(
            outputs['encrypted_visual'],
            outputs['visual_features'].device
        )
        self.assertTrue(
            torch.allclose(
                outputs['visual_features'],
                decrypted_features,
                atol=1e-5
            )
        )
        
    def test_gpu_memory_efficiency(self):
        """Test GPU memory efficiency."""
        initial_memory = torch.cuda.memory_allocated()
        
        # Create model
        model = MultimodalFusionEncoder(
            visual_dim=512,
            transaction_dim=self.input_dim,
            fusion_dim=512
        ).to(self.device)
        
        # Process batch
        with torch.no_grad():
            _ = model(self.test_images, self.test_transactions)
            
        final_memory = torch.cuda.memory_allocated()
        memory_increase = final_memory - initial_memory
        
        # Check memory efficiency
        memory_threshold = 2 * 1024 * 1024 * 1024  # 2GB
        self.assertLess(memory_increase, memory_threshold)
        
    def test_model_parallelization(self):
        """Test model parallelization across GPUs."""
        if torch.cuda.device_count() < 2:
            self.skipTest("Need at least 2 GPUs for this test")
            
        model = nn.DataParallel(
            MultimodalFusionEncoder(
                visual_dim=512,
                transaction_dim=self.input_dim,
                fusion_dim=512
            )
        ).to(self.device)
        
        # Test parallel forward pass
        outputs = model(self.test_images, self.test_transactions)
        self.assertEqual(outputs[0].shape, (self.batch_size, 512))
        
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        model = MultimodalFusionEncoder(
            visual_dim=512,
            transaction_dim=self.input_dim,
            fusion_dim=512
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        # Forward pass
        outputs = model(self.test_images, self.test_transactions)
        target = torch.randn_like(outputs[0])
        loss = criterion(outputs[0], target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param.grad).any())
            
    def test_model_save_load(self):
        """Test model saving and loading."""
        model = MultimodalFusionEncoder(
            visual_dim=512,
            transaction_dim=self.input_dim,
            fusion_dim=512
        ).to(self.device)
        
        # Save model
        save_path = self.temp_dir / 'model.pt'
        torch.save(model.state_dict(), save_path)
        
        # Load model
        loaded_model = MultimodalFusionEncoder(
            visual_dim=512,
            transaction_dim=self.input_dim,
            fusion_dim=512
        ).to(self.device)
        loaded_model.load_state_dict(torch.load(save_path))
        
        # Compare outputs
        with torch.no_grad():
            original_output = model(self.test_images, self.test_transactions)
            loaded_output = loaded_model(
                self.test_images,
                self.test_transactions
            )
            
        for orig, loaded in zip(original_output, loaded_output):
            self.assertTrue(torch.allclose(orig, loaded))
            
    def test_batch_processing(self):
        """Test batch processing capabilities."""
        model = MultimodalFusionEncoder(
            visual_dim=512,
            transaction_dim=self.input_dim,
            fusion_dim=512
        ).to(self.device)
        
        # Test different batch sizes
        batch_sizes = [1, 16, 32, 64]
        for size in batch_sizes:
            images = torch.randn(
                size, 3, 224, 224,
                device=self.device
            )
            transactions = torch.randn(
                size,
                self.input_dim,
                device=self.device
            )
            
            outputs = model(images, transactions)
            self.assertEqual(outputs[0].shape[0], size)
            
    def test_error_handling(self):
        """Test error handling capabilities."""
        model = MultimodalFusionEncoder(
            visual_dim=512,
            transaction_dim=self.input_dim,
            fusion_dim=512
        ).to(self.device)
        
        # Test invalid input dimensions
        with self.assertRaises(RuntimeError):
            invalid_images = torch.randn(
                self.batch_size, 4, 224, 224,
                device=self.device
            )
            model(invalid_images, self.test_transactions)
            
        # Test mismatched batch sizes
        with self.assertRaises(RuntimeError):
            mismatched_transactions = torch.randn(
                self.batch_size + 1,
                self.input_dim,
                device=self.device
            )
            model(self.test_images, mismatched_transactions)

def run_tests():
    """Run test suite."""
    unittest.main()

if __name__ == '__main__':
    run_tests()