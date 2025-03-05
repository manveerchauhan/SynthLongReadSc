"""
Main SynthLongRead framework class for generating synthetic long-read scRNA-seq data.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import random
from typing import Dict, List, Tuple, Optional, Union
from .data_processor import DataProcessor
from .error_model import ErrorModelTrainer
from .isoform_synth import IsoformSynthesizer
from .cell_barcode import CellBarcodeSynthesizer
from .fastq_generator import FASTQGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SynthLongRead')

class SynthLongRead:
    """Main class for generating synthetic long-read scRNA-seq data"""
    
    def __init__(self, 
                 reference_transcriptome: str, 
                 reference_gtf: str,
                 platform: str = "ONT",
                 is_single_nucleus: bool = False,
                 context_size: int = 5,
                 output_dir: str = "./output",
                 threads: int = 4):
        """
        Initialize the SynthLongRead framework.
        
        Args:
            reference_transcriptome: Path to reference transcriptome FASTA file
            reference_gtf: Path to reference GTF annotation file
            platform: Sequencing platform ("ONT" or "PacBio")
            is_single_nucleus: Whether data is single-nucleus (vs single-cell)
            context_size: Context size for error model (default: 5)
            output_dir: Directory to save output files
            threads: Number of threads to use for processing
        """
        self.reference_transcriptome = reference_transcriptome
        self.reference_gtf = reference_gtf
        self.platform = platform
        self.is_single_nucleus = is_single_nucleus
        self.context_size = context_size
        self.output_dir = output_dir
        self.threads = threads
        
        # Check that files exist
        self._check_files_exist()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up components
        self.processor = None
        self.error_model_trainer = None
        self.isoform_synth = None
        self.cell_barcode_synth = None
        self.fastq_generator = None
    
    def _check_files_exist(self):
        """Check that required files exist"""
        if not os.path.exists(self.reference_transcriptome):
            raise FileNotFoundError(f"Reference transcriptome not found: {self.reference_transcriptome}")
        if not os.path.exists(self.reference_gtf):
            raise FileNotFoundError(f"Reference GTF not found: {self.reference_gtf}")
    
    def learn_from_real_data(self, 
                            real_fastq: str, 
                            alignment_file: Optional[str] = None,
                            model_dir: Optional[str] = None):
        """
        Learn error profiles and train models from real data.
        
        Args:
            real_fastq: Path to real FASTQ file
            alignment_file: Path to existing alignment BAM file (optional)
            model_dir: Directory with pre-trained models (optional)
            
        Returns:
            self: For method chaining
        """
        logger.info("Learning from real data...")
        
        if not os.path.exists(real_fastq):
            raise FileNotFoundError(f"Real FASTQ file not found: {real_fastq}")
        
        # Process real data
        self.processor = DataProcessor(
            real_fastq, 
            self.reference_transcriptome,
            platform=self.platform,
            alignment_file=alignment_file,
            threads=self.threads
        )
        
        self.processor.parse_fastq()
        
        if alignment_file and os.path.exists(alignment_file):
            self.processor.alignment_file = alignment_file
        else:
            self.processor.align_to_reference()
        
        self.processor.extract_error_profiles()
        
        # Save profiles
        profile_file = os.path.join(self.output_dir, "error_profiles.pkl")
        self.processor.save_profiles(profile_file)
        
        # Train models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.error_model_trainer = ErrorModelTrainer(
            self.processor, 
            context_size=self.context_size,
            device=device
        )
        
        # If model_dir is provided, load existing models
        if model_dir and os.path.exists(model_dir):
            self.error_model_trainer.load_models(model_dir)
        else:
            self.error_model_trainer.train_models()
            
            # Save trained models
            model_dir = os.path.join(self.output_dir, "models")
            self.error_model_trainer.save_models(model_dir)
        
        return self
    
    def initialize_components(self):
        """Initialize remaining components after learning from data"""
        if self.processor is None or self.error_model_trainer is None:
            raise ValueError("Must call learn_from_real_data first")
        
        # Set up isoform synthesizer
        self.isoform_synth = IsoformSynthesizer(
            self.reference_gtf
        )
        
        # Load transcript sequences
        self.isoform_synth.load_transcript_sequences(self.reference_transcriptome)
        
        # Set up cell barcode synthesizer
        self.cell_barcode_synth = CellBarcodeSynthesizer()
        
        # Set up FASTQ generator
        self.fastq_generator = FASTQGenerator(
            self.error_model_trainer,
            self.isoform_synth,
            self.cell_barcode_synth
        )
    
    def generate_synthetic_dataset(self, 
                                  n_cells: int = 100, 
                                  sparsity: float = 0.8,
                                  max_reads: Optional[int] = None):
        """
        Generate complete synthetic dataset.
        
        Args:
            n_cells: Number of cells to simulate
            sparsity: Gene expression sparsity (0-1, higher means more zeros)
            max_reads: Maximum number of reads to generate (optional)
            
        Returns:
            Tuple[str, str]: Paths to output FASTQ file and ground truth CSV
        """
        # Initialize components if needed
        if self.fastq_generator is None:
            self.initialize_components()
        
        # Generate expression matrix
        logger.info(f"Generating expression matrix for {n_cells} cells")
        cell_matrix = self.isoform_synth.generate_cell_matrix(n_cells, sparsity)
        
        # Generate FASTQ
        output_fastq = os.path.join(self.output_dir, "synthetic_data.fastq")
        ground_truth = self.fastq_generator.generate_dataset(
            cell_matrix, output_fastq, max_reads
        )
        
        # Save ground truth
        ground_truth_file = os.path.join(self.output_dir, "ground_truth.csv")
        self._save_ground_truth(ground_truth, ground_truth_file)
        
        logger.info(f"Generated synthetic dataset: {output_fastq}")
        logger.info(f"Ground truth saved to: {ground_truth_file}")
        
        return output_fastq, ground_truth_file
    
    def _save_ground_truth(self, ground_truth: Dict[str, Dict[str, int]], output_file: str):
        """Save ground truth information to CSV"""
        rows = []
        
        for cell_id, isoform_counts in ground_truth.items():
            for isoform_id, count in isoform_counts.items():
                # Map isoform to gene
                gene_id = "unknown"
                for gene, isoforms in self.isoform_synth.gene_isoforms.items():
                    if isoform_id in isoforms:
                        gene_id = gene
                        break
                
                rows.append({
                    'cell_id': cell_id,
                    'gene_id': gene_id,
                    'transcript_id': isoform_id,
                    'count': count
                })
        
        # Write to CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
