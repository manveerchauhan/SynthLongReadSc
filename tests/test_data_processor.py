"""
Unit tests for DataProcessor component.
"""

import unittest
import os
import sys
import tempfile
import numpy as np
import pickle
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthlongread.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """Test DataProcessor component"""
    
    @patch('synthlongread.data_processor.SeqIO')
    def setUp(self, mock_seqio):
        """Set up test environment"""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create mock files
        self.fastq_file = os.path.join(self.temp_dir.name, "test.fastq")
        self.reference_file = os.path.join(self.temp_dir.name, "test.fa")
        
        # Create empty files
        with open(self.fastq_file, 'w') as f:
            f.write("")
        with open(self.reference_file, 'w') as f:
            f.write("")
        
        # Mock BioPython SeqIO for parse method
        mock_record1 = MagicMock()
        mock_record1.seq = "ACGTACGTACGT"
        mock_record1.letter_annotations = {"phred_quality": [30]*12}
        
        mock_record2 = MagicMock()
        mock_record2.seq = "GCTGCTGCTGCT"
        mock_record2.letter_annotations = {"phred_quality": [20]*12}
        
        # Configure mock to return our mock records
        mock_seqio.parse.return_value = [mock_record1, mock_record2]
        
        # Create DataProcessor instance
        self.processor = DataProcessor(
            self.fastq_file, self.reference_file, platform="ONT"
        )
    
    def tearDown(self):
        """Clean up after tests"""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test if DataProcessor initializes correctly"""
        self.assertEqual(self.processor.fastq_file, self.fastq_file)
        self.assertEqual(self.processor.reference_file, self.reference_file)
        self.assertEqual(self.processor.platform, "ONT")
        self.assertIsNone(self.processor.alignment_file)
        self.assertEqual(self.processor.threads, 4)
        
        # Check if error profiles are initialized
        self.assertIn('substitutions', self.processor.error_profiles)
        self.assertIn('insertions', self.processor.error_profiles)
        self.assertIn('deletions', self.processor.error_profiles)
        self.assertIn('homopolymer_errors', self.processor.error_profiles)
        self.assertIn('position_effect', self.processor.error_profiles)
        self.assertIn('quality_by_error', self.processor.error_profiles)
    
    def test_parse_fastq(self):
        """Test if parse_fastq works correctly"""
        # Run parse_fastq
        stats = self.processor.parse_fastq()
        
        # Check if stats are correctly computed
        self.assertIn('length_dist', stats)
        self.assertIn('quality_dist', stats)
        self.assertIn('gc_content', stats)
        self.assertIn('mean_length', stats)
        self.assertIn('std_length', stats)
        self.assertIn('mean_quality', stats)
        
        # Check if stats values are correct
        self.assertEqual(len(stats['length_dist']), 2)  # 2 mock records
        self.assertEqual(len(stats['quality_dist']), 24)  # 2*12 quality scores
        self.assertEqual(len(stats['gc_content']), 2)  # 2 records
    
    @patch('synthlongread.data_processor.os')
    def test_align_to_reference(self, mock_os):
        """Test if align_to_reference works correctly"""
        # Configure mock os to return success
        mock_os.system.return_value = 0
        mock_os.path.exists.return_value = True
        
        # Run align_to_reference
        output_bam = os.path.join(self.temp_dir.name, "test.bam")
        result = self.processor.align_to_reference(output_bam=output_bam)
        
        # Check if result is correct
        self.assertEqual(result, output_bam)
        self.assertEqual(self.processor.alignment_file, output_bam)
        
        # Check if commands were run
        mock_os.system.assert_any_call(
            f"minimap2 -ax map-ont -t 4 {self.reference_file} {self.fastq_file} | samtools sort -@ 4 > {output_bam}"
        )
        mock_os.system.assert_any_call(f"samtools index {output_bam}")
    
    @patch('synthlongread.data_processor.pysam')
    def test_extract_error_profiles(self, mock_pysam):
        """Test if extract_error_profiles works correctly"""
        # Set up alignment file
        self.processor.alignment_file = os.path.join(self.temp_dir.name, "test.bam")
        
        # Configure mock pysam
        mock_alignment = MagicMock()
        mock_alignment.is_unmapped = False
        mock_alignment.is_secondary = False
        mock_alignment.is_supplementary = False
        mock_alignment.get_reference_sequence.return_value = "ACGTACGTACGT"
        mock_alignment.query_sequence = "ACGTACATACGT"  # One mismatch
        mock_alignment.query_qualities = [30] * 12
        mock_alignment.cigartuples = [(0, 12)]  # 12M (12 matches/mismatches)
        
        # Configure mock AlignmentFile to return our mock alignment
        mock_bam = MagicMock()
        mock_bam.__enter__.return_value = mock_bam
        mock_bam.fetch.return_value = [mock_alignment]
        mock_pysam.AlignmentFile.return_value = mock_bam
        
        # Run extract_error_profiles
        profiles = self.processor.extract_error_profiles()
        
        # Check if profiles are correctly extracted
        self.assertIn('substitutions', profiles)
        self.assertIn('insertions', profiles)
        self.assertIn('deletions', profiles)
        self.assertIn('homopolymer_errors', profiles)
        self.assertIn('position_effect', profiles)
        self.assertIn('quality_by_error', profiles)
        
        # Should have non-zero substitution rate
        self.assertTrue(np.any(profiles['substitutions'] > 0))
    
    def test_save_load_profiles(self):
        """Test saving and loading profiles"""
        # Set up some sample data
        self.processor.read_stats = {
            'length_dist': np.array([100, 200]),
            'quality_dist': np.array([30, 30, 20, 20]),
            'gc_content': np.array([0.5, 0.5]),
            'mean_length': 150.0,
            'std_length': 50.0,
            'mean_quality': 25.0
        }
        
        # Save profiles
        profiles_file = os.path.join(self.temp_dir.name, "profiles.pkl")
        self.processor.save_profiles(profiles_file)
        
        # Check if file was created
        self.assertTrue(os.path.exists(profiles_file))
        
        # Load profiles into a new processor
        new_processor = DataProcessor.load_profiles(profiles_file)
        
        # Check if loaded data matches
        self.assertEqual(new_processor.platform, self.processor.platform)
        self.assertEqual(new_processor.read_stats['mean_length'], self.processor.read_stats['mean_length'])
        self.assertEqual(new_processor.read_stats['mean_quality'], self.processor.read_stats['mean_quality'])
        
        # Check arrays
        np.testing.assert_array_equal(
            new_processor.read_stats['length_dist'], 
            self.processor.read_stats['length_dist']
        )
        np.testing.assert_array_equal(
            new_processor.read_stats['quality_dist'], 
            self.processor.read_stats['quality_dist']
        )

if __name__ == '__main__':
    unittest.main()
