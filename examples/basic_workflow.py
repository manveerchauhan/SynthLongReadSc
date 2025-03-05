"""
Basic workflow example for SynthLongRead.
"""

import os
import argparse
import logging
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthlongread import SynthLongRead
from synthlongread.evaluation.benchmarking import benchmark_FLAMES

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Example')

def main():
    parser = argparse.ArgumentParser(description='SynthLongRead basic workflow example')
    
    # Required arguments
    parser.add_argument('--reference_transcriptome', required=True, 
                        help='Path to reference transcriptome FASTA file')
    parser.add_argument('--reference_gtf', required=True, 
                        help='Path to reference GTF annotation file')
    parser.add_argument('--real_fastq', required=True, 
                        help='Path to real FASTQ file for training')
    
    # Optional arguments
    parser.add_argument('--platform', choices=['ONT', 'PacBio'], default='ONT',
                        help='Sequencing platform (ONT or PacBio)')
    parser.add_argument('--output_dir', default='./output',
                        help='Directory to save output files')
    parser.add_argument('--n_cells', type=int, default=100,
                        help='Number of cells to simulate')
    parser.add_argument('--sparsity', type=float, default=0.8,
                        help='Gene expression sparsity (0-1)')
    parser.add_argument('--max_reads', type=int, default=100000,
                        help='Maximum number of reads to generate')
    parser.add_argument('--alignment_file', 
                        help='Path to existing alignment BAM file (optional)')
    parser.add_argument('--model_dir', 
                        help='Directory with pre-trained models (optional)')
    parser.add_argument('--run_benchmark', action='store_true',
                        help='Run FLAMES benchmark after generation')
    parser.add_argument('--reference_genome',
                        help='Path to reference genome FASTA file (required for benchmarking)')
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of threads to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    import numpy as np
    import torch
    import random
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Initialize SynthLongRead
    logger.info("Initializing SynthLongRead")
    synth = SynthLongRead(
        reference_transcriptome=args.reference_transcriptome,
        reference_gtf=args.reference_gtf,
        platform=args.platform,
        output_dir=args.output_dir,
        threads=args.threads
    )
    
    # Learn from real data
    logger.info("Learning from real data")
    synth.learn_from_real_data(
        real_fastq=args.real_fastq,
        alignment_file=args.alignment_file,
        model_dir=args.model_dir
    )
    
    # Generate synthetic dataset
    logger.info(f"Generating synthetic dataset with {args.n_cells} cells")
    synthetic_fastq, ground_truth = synth.generate_synthetic_dataset(
        n_cells=args.n_cells,
        sparsity=args.sparsity,
        max_reads=args.max_reads
    )
    
    logger.info(f"Generated synthetic data: {synthetic_fastq}")
    logger.info(f"Ground truth: {ground_truth}")
    
    # Run benchmark if requested
    if args.run_benchmark:
        if not args.reference_genome:
            logger.error("Reference genome is required for benchmarking")
        else:
            logger.info("Running FLAMES benchmark")
            benchmark_dir = os.path.join(args.output_dir, "benchmark")
            
            metrics_file = benchmark_FLAMES(
                synthetic_fastq=synthetic_fastq,
                ground_truth=ground_truth,
                reference_genome=args.reference_genome,
                reference_gtf=args.reference_gtf,
                output_dir=benchmark_dir
            )
            
            if metrics_file:
                logger.info(f"Benchmark results: {metrics_file}")
    
    logger.info("Workflow completed successfully")

if __name__ == "__main__":
    main()
