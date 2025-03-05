"""
Example of simulating internal priming in synthetic long-read scRNA-seq data.
"""

import os
import argparse
import logging
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthlongread import SynthLongRead
from synthlongread.evaluation.benchmarking import benchmark_FLAMES
from synthlongread.evaluation.read_metrics import ReadLevelMetrics

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('InternalPrimingExample')

def main():
    parser = argparse.ArgumentParser(description='SynthLongRead internal priming example')
    
    # Required arguments
    parser.add_argument('--reference_transcriptome', required=True, 
                        help='Path to reference transcriptome FASTA file')
    parser.add_argument('--reference_gtf', required=True, 
                        help='Path to reference GTF annotation file')
    parser.add_argument('--real_fastq', required=True, 
                        help='Path to real FASTQ file for training')
    
    # Optional arguments
    parser.add_argument('--output_dir', default='./internal_priming_output',
                        help='Directory to save output files')
    parser.add_argument('--n_cells', type=int, default=50,
                        help='Number of cells to simulate')
    parser.add_argument('--max_reads', type=int, default=50000,
                        help='Maximum number of reads to generate')
    parser.add_argument('--internal_priming_rate', type=float,
                        help='Rate of internal priming (default: auto-infer)')
    parser.add_argument('--reference_genome', 
                        help='Path to reference genome FASTA (for FLAMES)')
    parser.add_argument('--run_benchmark', action='store_true',
                        help='Run FLAMES benchmark after generation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize SynthLongRead with internal priming enabled
    logger.info("Initializing SynthLongRead with internal priming")
    synth = SynthLongRead(
        reference_transcriptome=args.reference_transcriptome,
        reference_gtf=args.reference_gtf,
        output_dir=args.output_dir,
        internal_priming=True,  # Enable internal priming
        internal_priming_rate=args.internal_priming_rate
    )
    
    # Learn from real data and infer internal priming model
    logger.info("Learning from real data and inferring internal priming model")
    synth.learn_from_real_data(
        real_fastq=args.real_fastq,
        infer_internal_priming=True  # Automatically infer internal priming from real data
    )
    
    # Generate synthetic dataset
    logger.info("Generating synthetic dataset with internal priming")
    synthetic_fastq, ground_truth = synth.generate_synthetic_dataset(
        n_cells=args.n_cells,
        max_reads=args.max_reads
    )
    
    logger.info(f"Generated synthetic data with internal priming: {synthetic_fastq}")
    logger.info(f"Ground truth: {ground_truth}")
    
    # Compare reads to real data to validate internal priming simulation
    logger.info("Validating synthetic data against real data")
    read_metrics_dir = os.path.join(args.output_dir, "read_metrics")
    os.makedirs(read_metrics_dir, exist_ok=True)
    
    read_metrics = ReadLevelMetrics(
        synthetic_fastq=synthetic_fastq,
        real_fastq=args.real_fastq,
        output_dir=read_metrics_dir
    )
    
    read_metrics.compare_stats()
    
    # Run FLAMES benchmark if requested
    if args.run_benchmark and args.reference_genome:
        logger.info("Running FLAMES benchmark with internal priming evaluation")
        benchmark_dir = os.path.join(args.output_dir, "benchmark")
        
        metrics_file = benchmark_FLAMES(
            synthetic_fastq=synthetic_fastq,
            ground_truth=ground_truth,
            reference_genome=args.reference_genome,
            reference_gtf=args.reference_gtf,
            output_dir=benchmark_dir
        )
        
        if metrics_file:
            logger.info(f"Benchmark results saved to: {metrics_file}")
            
            # Run internal priming-specific evaluation
            from synthlongread.evaluation import IsoformBenchmark
            
            # The results_file path depends on FLAMES output format
            results_file = os.path.join(benchmark_dir, "flames_results/transcript_count.csv.gz")
            
            if os.path.exists(results_file):
                logger.info("Running internal priming-specific evaluation")
                ip_eval_dir = os.path.join(benchmark_dir, "internal_priming_eval")
                os.makedirs(ip_eval_dir, exist_ok=True)
                
                ip_benchmark = IsoformBenchmark(
                    ground_truth_file=ground_truth,
                    results_file=results_file,
                    output_dir=ip_eval_dir,
                    evaluate_internal_priming=True  # Enable internal priming evaluation
                )
                
                ip_metrics = ip_benchmark.run_all_evaluations()
                
                logger.info("Internal priming evaluation completed")
    
    logger.info("Internal priming example completed successfully")
    
    # Print a summary of the key findings
    print("\n=====================================================")
    print("INTERNAL PRIMING SIMULATION SUMMARY")
    print("=====================================================")
    print(f"- Generated {args.max_reads} reads for {args.n_cells} cells")
    
    try:
        # Try to read internal priming rate from the simulator
        if hasattr(synth, 'internal_priming_simulator') and synth.internal_priming_simulator:
            inferred_rate = synth.internal_priming_simulator.priming_rate
            print(f"- Inferred internal priming rate: {inferred_rate:.2f}")
            
            # Count internal priming events in ground truth
            import pandas as pd
            gt_df = pd.read_csv(ground_truth)
            if 'has_internal_priming' in gt_df.columns:
                internal_count = gt_df['has_internal_priming'].sum()
                total_count = len(gt_df)
                actual_rate = internal_count / total_count
                print(f"- Actual internal priming events: {internal_count}/{total_count} ({actual_rate:.2%})")
    except Exception as e:
        pass
    
    print("\nFor detailed results, see the evaluation metrics in the output directory.")
    print("=====================================================")

if __name__ == "__main__":
    main()
