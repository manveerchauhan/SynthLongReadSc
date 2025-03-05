"""
Example script for benchmarking FLAMES on synthetic data.
"""

import os
import argparse
import logging
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthlongread.evaluation.benchmarking import benchmark_FLAMES
from synthlongread.evaluation.read_metrics import ReadLevelMetrics

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BenchmarkFLAMES')

def main():
    parser = argparse.ArgumentParser(description='Benchmark FLAMES on synthetic data')
    
    # Required arguments
    parser.add_argument('--synthetic_fastq', required=True, 
                        help='Path to synthetic FASTQ file')
    parser.add_argument('--ground_truth', required=True, 
                        help='Path to ground truth CSV file')
    parser.add_argument('--reference_genome', required=True,
                        help='Path to reference genome FASTA file')
    parser.add_argument('--reference_gtf', required=True,
                        help='Path to reference GTF annotation file')
    
    # Optional arguments
    parser.add_argument('--real_fastq', 
                        help='Path to real FASTQ file for comparison')
    parser.add_argument('--output_dir', default='./benchmark_results',
                        help='Directory to save benchmark results')
    parser.add_argument('--flames_path', default='flames',
                        help='Path to FLAMES executable')
    parser.add_argument('--config_file',
                        help='Path to custom FLAMES config file')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run FLAMES benchmark
    logger.info("Running FLAMES benchmark")
    benchmark_dir = os.path.join(args.output_dir, "flames")
    
    metrics_file = benchmark_FLAMES(
        synthetic_fastq=args.synthetic_fastq,
        ground_truth=args.ground_truth,
        reference_genome=args.reference_genome,
        reference_gtf=args.reference_gtf,
        output_dir=benchmark_dir,
        config_file=args.config_file,
        flames_path=args.flames_path
    )
    
    if metrics_file:
        logger.info(f"FLAMES benchmark results: {metrics_file}")
    else:
        logger.error("FLAMES benchmark failed")
        return
    
    # Compare read-level characteristics with real data if provided
    if args.real_fastq:
        logger.info("Comparing read-level characteristics with real data")
        read_metrics_dir = os.path.join(args.output_dir, "read_metrics")
        
        read_metrics = ReadLevelMetrics(
            synthetic_fastq=args.synthetic_fastq,
            real_fastq=args.real_fastq,
            output_dir=read_metrics_dir
        )
        
        stats = read_metrics.compare_stats()
        logger.info("Read-level comparison completed")
    
    logger.info("Benchmarking completed successfully")
    
    # Print a summary of the benchmark results
    print_benchmark_summary(metrics_file)

def print_benchmark_summary(metrics_file):
    """Print a summary of the benchmark results"""
    import pandas as pd
    
    try:
        metrics = pd.read_csv(metrics_file)
        
        print("\n=== FLAMES BENCHMARK SUMMARY ===\n")
        
        # Transcript-level metrics
        print("Transcript-level metrics:")
        print(f"  Detection F1 Score:       {metrics['avg_f1_score'].values[0]:.4f}")
        print(f"  Quantification Pearson r: {metrics['avg_pearson_r'].values[0]:.4f}")
        print(f"  Quantification Spearman r: {metrics['avg_spearman_r'].values[0]:.4f}")
        
        # Gene-level metrics
        print("\nGene-level metrics:")
        print(f"  Detection F1 Score:       {metrics['avg_gene_f1_score'].values[0]:.4f}")
        print(f"  Quantification Pearson r: {metrics['avg_gene_pearson_r'].values[0]:.4f}")
        
        # Isoform ratio metrics
        print("\nIsoform ratio metrics:")
        print(f"  Jensen-Shannon Divergence: {metrics['avg_js_divergence'].values[0]:.4f}")
        print(f"  Proportion MAE:            {metrics['avg_proportion_mae'].values[0]:.4f}")
        
        print("\nFor detailed results, see the metrics directory in the output folder.")
        
    except Exception as e:
        logger.error(f"Error reading metrics file: {str(e)}")

if __name__ == "__main__":
    main()
