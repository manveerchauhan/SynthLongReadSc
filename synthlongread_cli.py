#!/usr/bin/env python
"""
Command-line interface for SynthLongRead.
"""

import os
import argparse
import logging
import sys
from synthlongread import SynthLongRead
from synthlongread.config import create_template_config, load_config
from synthlongread.evaluation.benchmarking import benchmark_FLAMES
from synthlongread.evaluation.read_metrics import ReadLevelMetrics

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SynthLongRead-CLI')

def create_template_command(args):
    """Create a template configuration file"""
    create_template_config(args.output)
    logger.info(f"Created template configuration at {args.output}")

def run_command(args):
    """Run SynthLongRead with configuration file"""
    # Load configuration
    config = load_config(args.config)
    
    # Override settings if specified in command line
    if args.output_dir:
        config['output']['directory'] = args.output_dir
    
    if args.max_reads:
        config['dataset']['max_reads'] = args.max_reads
    
    # Initialize SynthLongRead
    synth = SynthLongRead(config_file=args.config)
    
    # Run the workflow
    synth.learn_from_real_data()
    
    # Generate synthetic dataset
    output_fastq, ground_truth = synth.generate_synthetic_dataset()
    
    # Run validation against real data if requested
    if args.validate:
        logger.info("Validating synthetic data against real data")
        read_metrics_dir = os.path.join(config['output']['directory'], "read_metrics")
        os.makedirs(read_metrics_dir, exist_ok=True)
        
        read_metrics = ReadLevelMetrics(
            synthetic_fastq=output_fastq,
            real_fastq=config['input']['real_fastq'],
            output_dir=read_metrics_dir
        )
        
        read_metrics.compare_stats()
    
    # Run FLAMES benchmark if requested
    if config['benchmark']['run_flames'] and config['input'].get('reference_genome'):
        logger.info("Running FLAMES benchmark")
        benchmark_dir = os.path.join(config['output']['directory'], "benchmark")
        
        metrics_file = benchmark_FLAMES(
            synthetic_fastq=output_fastq,
            ground_truth=ground_truth,
            reference_genome=config['input']['reference_genome'],
            reference_gtf=config['input']['reference_gtf'],
            output_dir=benchmark_dir,
            flames_path=config['benchmark'].get('flames_path', 'flames')
        )
        
        if metrics_file:
            logger.info(f"Benchmark results saved to: {metrics_file}")
    
    logger.info("SynthLongRead run completed successfully")

def benchmark_command(args):
    """Run benchmark on synthetic data"""
    if not os.path.exists(args.synthetic_fastq):
        logger.error(f"Synthetic FASTQ file not found: {args.synthetic_fastq}")
        return
    
    if not os.path.exists(args.ground_truth):
        logger.error(f"Ground truth file not found: {args.ground_truth}")
        return
    
    if not os.path.exists(args.reference_genome):
        logger.error(f"Reference genome file not found: {args.reference_genome}")
        return
    
    if not os.path.exists(args.reference_gtf):
        logger.error(f"Reference GTF file not found: {args.reference_gtf}")
        return
    
    logger.info("Running FLAMES benchmark")
    
    metrics_file = benchmark_FLAMES(
        synthetic_fastq=args.synthetic_fastq,
        ground_truth=args.ground_truth,
        reference_genome=args.reference_genome,
        reference_gtf=args.reference_gtf,
        output_dir=args.output_dir,
        flames_path=args.flames_path
    )
    
    if metrics_file:
        logger.info(f"Benchmark results saved to: {metrics_file}")

def validate_command(args):
    """Validate synthetic data against real data"""
    if not os.path.exists(args.synthetic_fastq):
        logger.error(f"Synthetic FASTQ file not found: {args.synthetic_fastq}")
        return
    
    if not os.path.exists(args.real_fastq):
        logger.error(f"Real FASTQ file not found: {args.real_fastq}")
        return
    
    logger.info("Validating synthetic data against real data")
    
    read_metrics = ReadLevelMetrics(
        synthetic_fastq=args.synthetic_fastq,
        real_fastq=args.real_fastq,
        output_dir=args.output_dir
    )
    
    read_metrics.compare_stats()
    
    logger.info(f"Validation results saved to {args.output_dir}")

def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(description='SynthLongRead Command Line Interface')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Template creation command
    template_parser = subparsers.add_parser('create-template', help='Create template configuration file')
    template_parser.add_argument('--output', default='synthlongread_config.yaml', help='Output template file path')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run SynthLongRead with configuration')
    run_parser.add_argument('--config', required=True, help='Path to configuration YAML file')
    run_parser.add_argument('--output-dir', help='Override output directory from config')
    run_parser.add_argument('--max-reads', type=int, help='Override maximum number of reads')
    run_parser.add_argument('--validate', action='store_true', help='Validate synthetic data against real data')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark on synthetic data')
    benchmark_parser.add_argument('--synthetic-fastq', required=True, help='Path to synthetic FASTQ file')
    benchmark_parser.add_argument('--ground-truth', required=True, help='Path to ground truth CSV file')
    benchmark_parser.add_argument('--reference-genome', required=True, help='Path to reference genome FASTA file')
    benchmark_parser.add_argument('--reference-gtf', required=True, help='Path to reference GTF file')
    benchmark_parser.add_argument('--output-dir', default='./benchmark_results', help='Output directory for benchmark results')
    benchmark_parser.add_argument('--flames-path', default='flames', help='Path to FLAMES executable')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate synthetic data against real data')
    validate_parser.add_argument('--synthetic-fastq', required=True, help='Path to synthetic FASTQ file')
    validate_parser.add_argument('--real-fastq', required=True, help='Path to real FASTQ file')
    validate_parser.add_argument('--output-dir', default='./validation_results', help='Output directory for validation results')
    
    args = parser.parse_args()
    
    if args.command == 'create-template':
        create_template_command(args)
    elif args.command == 'run':
        run_command(args)
    elif args.command == 'benchmark':
        benchmark_command(args)
    elif args.command == 'validate':
        validate_command(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
