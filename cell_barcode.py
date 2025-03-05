"""
Generate realistic cell barcodes with errors for scRNA-seq data.
"""

import numpy as np
import logging
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CellBarcodeSynthesizer')

# Constants
NUCLEOTIDES = ['A', 'C', 'G', 'T']

class CellBarcodeSynthesizer:
    """Generate realistic cell barcodes with errors"""
    
    def __init__(self, real_barcodes: Optional[List[str]] = None, 
                 error_rate: float = 0.01,
                 barcode_length: int = 16):
        """
        Initialize the CellBarcodeSynthesizer.
        
        Args:
            real_barcodes: List of real cell barcodes to sample from (optional)
            error_rate: Error rate for barcode errors
            barcode_length: Length of synthetic barcodes if generated
        """
        self.real_barcodes = real_barcodes
        self.error_rate = error_rate
        self.barcode_length = barcode_length
    
    def generate_barcodes(self, n_cells: int) -> List[str]:
        """
        Generate cell barcodes.
        
        Args:
            n_cells: Number of cell barcodes to generate
            
        Returns:
            List[str]: List of generated cell barcodes
        """
        if self.real_barcodes and len(self.real_barcodes) >= n_cells:
            # Sample from real barcodes
            return np.random.choice(self.real_barcodes, n_cells, replace=False)
        else:
            # Generate synthetic barcodes
            barcodes = []
            for _ in range(n_cells):
                barcode = ''.join(np.random.choice(NUCLEOTIDES, self.barcode_length))
                barcodes.append(barcode)
            return barcodes
    
    def introduce_barcode_errors(self, barcode: str) -> str:
        """
        Introduce realistic errors into barcodes.
        
        Args:
            barcode: Input barcode sequence
            
        Returns:
            str: Barcode with introduced errors
        """
        barcode_list = list(barcode)
        for i in range(len(barcode)):
            if np.random.random() < self.error_rate:
                # Randomly choose error type: substitution, insertion, deletion
                error_type = np.random.choice(['sub', 'ins', 'del'])
                if error_type == 'sub':
                    # Substitution
                    options = list(set(NUCLEOTIDES) - {barcode[i]})
                    barcode_list[i] = np.random.choice(options)
                elif error_type == 'ins':
                    # Insertion
                    barcode_list.insert(i, np.random.choice(NUCLEOTIDES))
                elif error_type == 'del' and len(barcode) > 1:
                    # Deletion
                    barcode_list.pop(i)
                    break  # Need to break to avoid index issues
        return ''.join(barcode_list)
    
    def load_real_barcodes(self, barcode_file: str):
        """
        Load real barcodes from a file.
        
        Args:
            barcode_file: Path to file containing real barcodes (one per line)
        """
        try:
            with open(barcode_file, 'r') as f:
                self.real_barcodes = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(self.real_barcodes)} real barcodes from {barcode_file}")
        except Exception as e:
            logger.error(f"Error loading real barcodes: {str(e)}")
    
    def generate_10x_pattern_barcodes(self, n_cells: int) -> List[str]:
        """
        Generate barcodes with 10x Genomics-like patterns.
        
        Args:
            n_cells: Number of cell barcodes to generate
            
        Returns:
            List[str]: List of generated cell barcodes
        """
        # 10x barcodes use a whitelist of sequences
        # Here we just simulate the pattern but don't use the actual whitelist
        barcodes = []
        
        # Start with the default barcode length
        length = 16
        
        # Generate barcodes with reasonable hamming distance
        while len(barcodes) < n_cells:
            candidate = ''.join(np.random.choice(NUCLEOTIDES, length))
            
            # Check if it's sufficiently different from existing barcodes
            # (simplified approach)
            if all(self._hamming_distance(candidate, bc) >= 3 for bc in barcodes):
                barcodes.append(candidate)
        
        return barcodes
    
    def _hamming_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Hamming distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            int: Hamming distance
        """
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
