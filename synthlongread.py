"""
Main SynthLongRead framework class for generating synthetic long-read scRNA-seq data.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import pickle
import random
from typing import Dict, List, Tuple, Optional, Union, Any

from .data_processor import DataProcessor
from .error_model import ErrorModelTrainer
from .isoform_synth import IsoformSynthesizer
from .cell_barcode import CellBarcodeSynthesizer
from .fastq_generator import FASTQGenerator
from .internal_priming import InternalPrimingSimulator
from .config import load_config, save_config

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SynthLongRead')

class SynthLongRead:
    """Main class for generating synthetic long-read scRNA-seq data"""
    
    def __init__(self, 
                 reference_transcriptome: str = None, 
                 reference_gtf: str = None,
                 platform: str = "ONT",
                 is_single_nucleus: bool = False,
                 context_size: int = 5,
                 output_dir: str = "./output",
                 threads: int = 4,
                 adapter_5p: str = "CCCATGTACTCTGCGTTGATACCACTGCTT",
                 adapter_3p: str = "AAAAAAAAAAAAAAAAAA",
                 internal_priming: bool = False,
                 internal_priming_rate: Optional[float] = None,
                 config_file: Optional[str] = None,
                 seed: int = 42):
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
            adapter_5p: 5' adapter sequence
            adapter_3p: 3' adapter sequence (poly-A tail)
            internal_priming: Whether to simulate internal priming
            internal_priming_rate: Rate of internal priming (None=auto-infer)
            config_file: Path to configuration YAML file (overrides other arguments)
            seed: Random seed for reproducibility
        """
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # If config file is provided, load configuration
        if config_file:
            self._load_from_config(config_file)
        else:
            # Use provided arguments
            self.reference_transcriptome = reference_transcriptome
            self.reference_gtf = reference_gtf
            self.platform = platform
            self.is_single_nucleus = is_single_nucleus
            self.context_size = context_size
            self.output_dir = output_dir
            self.threads = threads
            self.adapter_5p = adapter_5p
            self.adapter_3p = adapter_3p
            self.internal_priming = internal_priming
            self.internal_priming_rate = internal_priming_rate
            self.seed = seed
            
            # Initialize other configuration with defaults
            self.config = {
                'input': {
                    'reference_transcriptome': reference_transcriptome,
                    'reference_gtf': reference_gtf,
                    'alignment_file': None,
                    'real_fastq': None
                },
                'output': {
                    'directory': output_dir,
                    'overwrite': True
                },
                'platform': {
                    'type': platform,
                    'is_single_nucleus': is_single_nucleus,
                    'adapter_5p': adapter_5p,
                    'adapter_3p': adapter_3p
                },
                'error_model': {
                    'context_size': context_size,
                    'load_existing': False,
                    'model_dir': None
                },
                'internal_priming': {
                    'enabled': internal_priming,
                    'rate': internal_priming_rate,
                    'min_a_content': 0.65,
                    'window_size': 10,
                    'infer_from_data': True
                },
                'performance': {
                    'threads': threads,
                    'seed': seed,
                    'device': 'auto'
                }
            }
        
        # Check that required files are specified
        if self.reference_transcriptome is None or self.reference_gtf is None:
            raise ValueError("Reference transcriptome and GTF must be specified")
        
        # Check that files exist
        self._check_files_exist()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save the configuration
        self._save_config()
        
        # Set up components
        self.processor = None
        self.error_model_trainer = None
        self.isoform_synth = None
        self.cell_barcode_synth = None
        self.fastq_generator = None
        self.internal_priming_simulator = None
    
    def _load_from_config(self, config_file: str):
        """Load settings from configuration file"""
        self.config = load_config(config_file)
        
        # Extract key settings from config
        self.reference_transcriptome = self.config['input']['reference_transcriptome']
        self.reference_gtf = self.config['input']['reference_gtf']
        self.platform = self.config['platform']['type']
        self.is_single_nucleus = self.config['platform']['is_single_nucleus']
        self.context_size = self.config['error_model']['context_size']
        self.output_dir = self.config['output']['directory']
        self.threads = self.config['performance']['threads']
        self.adapter_5p = self.config['platform']['adapter_5p']
        self.adapter_3p = self.config['platform']['adapter_3p']
        self.internal_priming = self.config['internal_priming']['enabled']
        self.internal_priming_rate = self.config['internal_priming']['rate']
        self.seed = self.config['performance']['seed']
        
        # Update random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        logger.info(f"Configuration loaded from {config_file}")
    
    def _save_config(self):
        """Save current configuration to file"""
        config_file = os.path.join(self.output_dir, "synthlongread_config.yaml")
        save_config(self.config, config_file)
    
    def _check_files_exist(self):
        """Check that required files exist"""
        if not os.path.exists(self.reference_transcriptome):
            raise FileNotFoundError(f"Reference transcriptome not found: {self.reference_transcriptome}")
        if not os.path.exists(self.reference_gtf):
            raise FileNotFoundError(f"Reference GTF not found: {self.reference_gtf}")
    
    def learn_from_real_data(self, 
                            real_fastq: str = None, 
                            alignment_file: Optional[str] = None,
                            model_dir: Optional[str] = None,
                            infer_internal_priming: Optional[bool] = None):
        """
        Learn error profiles and train models from real data.
        
        Args:
            real_fastq: Path to real FASTQ file
            alignment_file: Path to existing alignment BAM file (optional)
            model_dir: Directory with pre-trained models (optional)
            infer_internal_priming: Whether to infer internal priming from real data
            
        Returns:
            self: For method chaining
        """
        logger.info("Learning from real data...")
        
        # Update config and parameters from args if provided
        if real_fastq is not None:
            self.config['input']['real_fastq'] = real_fastq
        else:
            real_fastq = self.config['input']['real_fastq']
        
        if alignment_file is not None:
            self.config['input']['alignment_file'] = alignment_file
        else:
            alignment_file = self.config['input']['alignment_file']
        
        if model_dir is not None:
            self.config['error_model']['model_dir'] = model_dir
            self.config['error_model']['load_existing'] = True
        else:
            model_dir = self.config['error_model']['model_dir']
        
        if infer_internal_priming is not None:
            self.config['internal_priming']['infer_from_data'] = infer_internal_priming
        else:
            infer_internal_priming = self.config['internal_priming']['infer_from_data']
        
        # Check that real FASTQ exists
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
        device = "cuda" if torch.cuda.is_available() and self.config['performance']['device'] != 'cpu' else "cpu"
        self.error_model_trainer = ErrorModelTrainer(
            self.processor, 
            context_size=self.context_size,
            device=device
        )
        
        # If model_dir is provided or load_existing is True, load existing models
        if self.config['error_model']['load_existing'] and model_dir and os.path.exists(model_dir):
            self.error_model_trainer.load_models(model_dir)
        else:
            self.error_model_trainer.train_models()
            
            # Save trained models
            model_dir = os.path.join(self.output_dir, "models")
            self.error_model_trainer.save_models(model_dir)
            self.config['error_model']['model_dir'] = model_dir
        
        # Initialize internal priming simulator if enabled
        if self.internal_priming:
            logger.info("Initializing internal priming simulation")
            self.internal_priming_simulator = InternalPrimingSimulator(
                priming_rate=self.internal_priming_rate,
                min_a_content=self.config['internal_priming']['min_a_content'],
                window_size=self.config['internal_priming']['window_size']
            )
            
            # Infer internal priming model from real data if requested
            if infer_internal_priming:
                logger.info("Inferring internal priming model from real data")
                self.internal_priming_simulator.infer_internal_priming_model(
                    self.processor.alignment_file,
                    self.reference_gtf,
                    self.reference_transcriptome
                )
                
                # Update internal priming rate if it was auto-inferred
                if self.internal_priming_rate is None:
                    self.internal_priming_rate = self.internal_priming_simulator.priming_rate
                    self.config['internal_priming']['rate'] = self.internal_priming_rate
                
                # Save internal priming model
                internal_priming_file = os.path.join(self.output_dir, "internal_priming_model.pkl")
                self.internal_priming_simulator.save(internal_priming_file)
            else:
                # Just scan transcripts for potential priming sites
                logger.info("Scanning transcripts for potential internal priming sites")
                self.internal_priming_simulator.scan_transcripts(self.reference_transcriptome)
        
        # Save updated configuration
        self._save_config()
        
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
        
        # Set up FASTQ generator with internal priming simulator if enabled
        self.fastq_generator = FASTQGenerator(
            self.error_model_trainer,
            self.isoform_synth,
            self.cell_barcode_synth,
            adapter_5p=self.adapter_5p,
            adapter_3p=self.adapter_3p,
            internal_priming=self.internal_priming,
            internal_priming_simulator=self.internal_priming_simulator
        )
    
    def generate_synthetic_dataset(self, 
                                  n_cells: Optional[int] = None, 
                                  sparsity: Optional[float] = None,
                                  max_reads: Optional[int] = None):
        """
        Generate complete synthetic dataset.
        
        Args:
            n_cells: Number of cells to generate (overrides config)
            sparsity: Gene expression sparsity (0-1, overrides config)
            max_reads: Maximum number of reads to generate (overrides config)
            
        Returns:
            Tuple[str, str]: Paths to output FASTQ file and ground truth CSV
        """
        # Initialize components if needed
        if self.fastq_generator is None:
            self.initialize_components()
        
        # Use config values if not specified in arguments
        if n_cells is None:
            n_cells = self.config['dataset'].get('n_cells', 100)
        
        if sparsity is None:
            sparsity = self.config['dataset'].get('sparsity', 0.8)
        
        if max_reads is None:
            max_reads = self.config['dataset'].get('max_reads', None)
        
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
                    'count': count,
                    'has_internal_priming': "_internal_" in isoform_id  # Track internal priming
                })
        
        # Write to CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        
        # Log some statistics about internal priming
        if self.internal_priming and self.internal_priming_simulator:
            internal_priming_count = sum(1 for row in rows if row.get('has_internal_priming', False))
            total_reads = len(rows)
            internal_priming_pct = internal_priming_count / total_reads * 100 if total_reads > 0 else 0
            logger.info(f"Internal priming statistics:")
            logger.info(f"  - Target rate: {self.internal_priming_simulator.priming_rate:.2f}")
            logger.info(f"  - Actual rate: {internal_priming_pct:.2f}% ({internal_priming_count}/{total_reads} reads)")