"""
Unit tests for FASTQGenerator component.
"""

import unittest
import os
import sys
import tempfile
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthlongread.fastq_generator import FASTQGenerator

class TestFASTQGenerator(unittest.TestCase):
    """Test FASTQGenerator component"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create mock dependencies
        self.error_model_trainer = MagicMock()
        self.isoform_synth = MagicMock()
        self.cell_barcode_synth = MagicMock()
        
        # Configure mock error models
        self.seq_error_model = MagicMock()
        self.quality_model = MagicMock()
        
        self.seq_error_model.eval = MagicMock()
        self.quality_model.eval = MagicMock()
        
        self.error_model_trainer.seq_error_model = self.seq_error_model
        self.error_model_trainer.quality_model = self.quality_model
        
        # Configure mock read length model
        self.error_model_trainer.read_length_model = MagicMock()
        self.error_model_trainer.read_length_model.sample.return_value = np.array([[1000]])
        
        # Configure mock isoform sequences
        self.isoform_synth.isoform_sequences = {
            "transcript1": "A" * 1000,
            "transcript2": "G" * 1000,
            "transcript3": "C" * 1000
        }
        
        # Configure mock barcode generator
        self.cell_barcode_synth.generate_barcodes.return_value = ["ACGTACGTACGTACGT"]
        self.cell_barcode_synth.introduce_barcode_errors.return_value = "ACGTACGTACGTACGT"
        
        # Configure seq_error_model to return error probabilities
        def mock_seq_error(context, position):
            batch_size = context.shape[0]
            # Return probabilities for match, sub, del (heavily biased towards matches)
            probs = torch.zeros(batch_size, 4, 3)
            probs[:, :, 0] = 0.95  # 95% probability of match
            probs[:, :, 1] = 0.03  # 3% probability of substitution
            probs[:, :, 2] = 0.02  # 2% probability of deletion
            return probs
        
        self.seq_error_model.side_effect = mock_seq_error
        
        # Configure quality_model to return quality scores
        def mock_quality(context, error_probs, position):
            batch_size = context.shape[0]
            return torch.full((batch_size, 1), 30.0)  # Q30 quality
        
        self.quality_model.side_effect = mock_quality
        
        # Create FASTQGenerator instance
        self.generator = FASTQGenerator(
            self.error_model_trainer,
            self.isoform_synth,
            self.cell_barcode_synth,
            device="cpu"
        )
    
    def tearDown(self):
        """Clean up after tests"""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test if FASTQGenerator initializes correctly"""
        self.assertEqual(self.generator.error_model_trainer, self.error_model_trainer)
        self.assertEqual(self.generator.isoform_synth, self.isoform_synth)
        self.assertEqual(self.generator.cell_barcode_synth, self.cell_barcode_synth)
        self.assertEqual(self.generator.device, "cpu")
        
        # Check if models were put in eval mode
        self.seq_error_model.eval.assert_called_once()
        self.quality_model.eval.assert_called_once()
    
    def test_one_hot_encode(self):
        """Test one-hot encoding of sequences"""
        # Test with a simple sequence
        seq = "ACGT"
        encoded = self.generator.one_hot_encode(seq)
        
        # Expected encoding
        expected = [
            1, 0, 0, 0,  # A
            0, 1, 0, 0,  # C
            0, 0, 1, 0,  # G
            0, 0, 0, 1   # T
        ]
        
        self.assertEqual(encoded, expected)
        
        # Test with unknown base
        seq = "ACNGT"
        encoded = self.generator.one_hot_encode(seq)
        
        # Expected encoding
        expected = [
            1, 0, 0, 0,          # A
            0, 1, 0, 0,          # C
            0.25, 0.25, 0.25, 0.25,  # N
            0, 0, 1, 0,          # G
            0, 0, 0, 1           # T
        ]
        
        self.assertEqual(encoded, expected)
    
    @patch('torch.tensor')
    @patch('torch.no_grad')
    @patch('numpy.random.choice')
    def test_generate_read_with_errors(self, mock_choice, mock_no_grad, mock_tensor):
        """Test generating a read with errors"""
        # Configure mocks
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock()
        
        # Configure numpy.random.choice to always return match (0)
        mock_choice.return_value = 0
        
        # Generate a read
        sequence = "ACGTACGTACGT"
        read_seq, quality = self.generator.generate_read_with_errors(sequence)
        
        # Check if result has correct type and length
        self.assertIsInstance(read_seq, str)
        self.assertIsInstance(quality, list)
        
        # Length might be shorter due to deletions
        # or longer due to insertions, so we can't check exact length
        self.assertGreater(len(read_seq), 0)
        self.assertEqual(len(read_seq), len(quality))
    
    def test_generate_read_length(self):
        """Test generating read lengths"""
        # Test with mock read length model
        length = self.generator._generate_read_length()
        self.assertEqual(length, 1000)
        
        # Test with no model (fallback to normal distribution)
        self.error_model_trainer.read_length_model = None
        
        # Patch numpy.random.normal to return a specific value
        with patch('numpy.random.normal', return_value=500):
            length = self.generator._generate_read_length()
            self.assertEqual(length, 500)
    
    def test_generate_reads_from_transcript(self):
        """Test generating reads from a transcript"""
        # Configure patches for random choices
        with patch('numpy.random.randint', return_value=0), \
             patch('numpy.random.random', return_value=0.5), \
             patch('torch.tensor', return_value=torch.zeros(1, 1)), \
             patch('torch.no_grad', MagicMock()):
            
            # Generate reads
            reads = self.generator.generate_reads_from_transcript(
                "transcript1", 2, "ACGTACGTACGTACGT"
            )
            
            # Check results
            self.assertEqual(len(reads), 2)
            for read_seq, quality, read_id in reads:
                self.assertIsInstance(read_seq, str)
                self.assertIsInstance(quality, list)
                self.assertIsInstance(read_id, str)
                self.assertIn("transcript1", read_id)
                self.assertIn("cell", read_id)
    
    def test_write_fastq(self):
        """Test writing reads to FASTQ file"""
        # Create some test reads
        reads = [
            ("ACGT", [30, 30, 30, 30], "read1"),
            ("TGCA", [20, 20, 20, 20], "read2")
        ]
        
        # Write to a temporary file
        output_file = os.path.join(self.temp_dir.name, "test.fastq")
        self.generator.write_fastq(reads, output_file)
        
        # Check if file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check file content
        with open(output_file, 'r') as f:
            content = f.read()
            
            # Check if both reads are in the file
            self.assertIn("@read1_0", content)
            self.assertIn("@read2_1", content)
            
            # Check if sequences are in the file
            self.assertIn("ACGT", content)
            self.assertIn("TGCA", content)
            
            # Check if quality scores are in the file
            # Q30 is ASCII '?', Q20 is ASCII '5'
            self.assertIn("????", content)  # Q30
            self.assertIn("5555", content)  # Q20
    
    def test_generate_dataset(self):
        """Test generating a complete dataset"""
        # Create a mock cell matrix
        cell_matrix = {
            "cell_1": {"transcript1": 1, "transcript2": 2},
            "cell_2": {"transcript2": 1, "transcript3": 1}
        }
        
        # Configure patches
        with patch.object(self.generator, 'generate_cell_reads') as mock_generate_cell_reads, \
             patch.object(self.generator, 'write_fastq') as mock_write_fastq, \
             patch('numpy.random.shuffle'):
            
            # Configure mock to return some reads
            mock_generate_cell_reads.side_effect = [
                [("ACGT", [30, 30, 30, 30], "read1"), ("TGCA", [20, 20, 20, 20], "read2")],
                [("GGCC", [25, 25, 25, 25], "read3")]
            ]
            
            # Generate dataset
            output_file = os.path.join(self.temp_dir.name, "test.fastq")
            ground_truth = self.generator.generate_dataset(cell_matrix, output_file)
            
            # Check if ground_truth matches cell_matrix
            self.assertEqual(ground_truth, cell_matrix)
            
            # Check if generate_cell_reads was called for each cell
            self.assertEqual(mock_generate_cell_reads.call_count, 2)
            
            # Check if write_fastq was called with all reads
            mock_write_fastq.assert_called_once()
            self.assertEqual(len(mock_write_fastq.call_args[0][0]), 3)  # 3 reads total

if __name__ == '__main__':
    unittest.main()
