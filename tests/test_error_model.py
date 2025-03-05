"""
Unit tests for error model components.
"""

import unittest
import torch
import numpy as np
import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthlongread.error_model import SequenceErrorModel, QualityScoreModel

class TestErrorModels(unittest.TestCase):
    """Test error model components"""
    
    def setUp(self):
        """Set up test environment"""
        self.context_size = 5
        self.device = "cpu"
        
        # Create models
        self.seq_error_model = SequenceErrorModel(context_size=self.context_size)
        self.quality_model = QualityScoreModel(context_size=self.context_size)
    
    def test_sequence_error_model_dimensions(self):
        """Test if sequence error model outputs have correct dimensions"""
        # Create a batch of random inputs
        batch_size = 4
        context = torch.rand(batch_size, 4 * self.context_size)
        position = torch.rand(batch_size, 1)
        
        # Forward pass
        output = self.seq_error_model(context, position)
        
        # Check output dimensions
        self.assertEqual(output.shape, (batch_size, 4, 3), 
                        "Output shape should be [batch_size, 4, 3]")
    
    def test_sequence_error_model_probabilities(self):
        """Test if sequence error model outputs are valid probabilities"""
        # Create a batch of random inputs
        batch_size = 4
        context = torch.rand(batch_size, 4 * self.context_size)
        position = torch.rand(batch_size, 1)
        
        # Forward pass
        output = self.seq_error_model(context, position)
        
        # Check if probabilities sum to 1 for each base
        prob_sums = output.sum(dim=2)
        for sum_val in prob_sums.view(-1):
            self.assertAlmostEqual(sum_val.item(), 1.0, places=5, 
                                 msg="Probabilities should sum to 1")
        
        # Check if probabilities are in [0, 1]
        self.assertTrue(torch.all(output >= 0), "Probabilities should be >= 0")
        self.assertTrue(torch.all(output <= 1), "Probabilities should be <= 1")
    
    def test_sequence_error_model_deterministic(self):
        """Test if sequence error model is deterministic"""
        # Create two identical inputs
        context = torch.rand(1, 4 * self.context_size)
        position = torch.rand(1, 1)
        
        # Two forward passes
        output1 = self.seq_error_model(context, position)
        output2 = self.seq_error_model(context, position)
        
        # Check if outputs are identical
        self.assertTrue(torch.allclose(output1, output2), 
                       "Model should be deterministic for the same input")
    
    def test_quality_model_dimensions(self):
        """Test if quality model outputs have correct dimensions"""
        # Create a batch of random inputs
        batch_size = 4
        context = torch.rand(batch_size, 4 * self.context_size)
        error_probs = torch.rand(batch_size, 3)
        position = torch.rand(batch_size, 1)
        
        # Forward pass
        output = self.quality_model(context, error_probs, position)
        
        # Check output dimensions
        self.assertEqual(output.shape, (batch_size, 1), 
                        "Output shape should be [batch_size, 1]")
    
    def test_quality_model_range(self):
        """Test if quality model outputs are in valid range"""
        # Create a batch of random inputs
        batch_size = 4
        context = torch.rand(batch_size, 4 * self.context_size)
        error_probs = torch.rand(batch_size, 3)
        position = torch.rand(batch_size, 1)
        
        # Forward pass
        output = self.quality_model(context, error_probs, position)
        
        # Check if quality scores are in valid range [0, 41]
        self.assertTrue(torch.all(output >= 0), "Quality scores should be >= 0")
        self.assertTrue(torch.all(output <= 41), "Quality scores should be <= 41")
    
    def test_model_save_load(self):
        """Test saving and loading models"""
        # Create temporary directory for saving models
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save models
            torch.save(self.seq_error_model.state_dict(), 
                      os.path.join(tmpdirname, "seq_error_model.pt"))
            torch.save(self.quality_model.state_dict(),
                      os.path.join(tmpdirname, "quality_model.pt"))
            
            # Create new models for loading
            seq_model_loaded = SequenceErrorModel(context_size=self.context_size)
            quality_model_loaded = QualityScoreModel(context_size=self.context_size)
            
            # Load saved models
            seq_model_loaded.load_state_dict(
                torch.load(os.path.join(tmpdirname, "seq_error_model.pt"))
            )
            quality_model_loaded.load_state_dict(
                torch.load(os.path.join(tmpdirname, "quality_model.pt"))
            )
            
            # Check if models have same parameters
            for p1, p2 in zip(self.seq_error_model.parameters(), seq_model_loaded.parameters()):
                self.assertTrue(torch.allclose(p1, p2), 
                               "Sequence error model parameters should be identical after loading")
            
            for p1, p2 in zip(self.quality_model.parameters(), quality_model_loaded.parameters()):
                self.assertTrue(torch.allclose(p1, p2), 
                               "Quality model parameters should be identical after loading")

if __name__ == '__main__':
    unittest.main()
