"""
Benchmark FLAMES on synthetic data and evaluate performance.
"""

import os
import logging
import json
import pandas as pd
from typing import Optional, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Benchmarking')

def benchmark_FLAMES(synthetic_fastq: str, 
                    ground_truth: str,
                    reference_genome: str,
                    reference_gtf: str,
                    output_dir: str,
                    config_file: Optional[str] = None,
                    flames_path: str = "flames") -> str:
    """
    Run FLAMES on synthetic data and evaluate performance.
    
    Args:
        synthetic_fastq: Path to synthetic FASTQ file
        ground_truth: Path to ground truth CSV file
        reference_genome: Path to reference genome FASTA file
        reference_gtf: Path to reference GTF annotation file
        output_dir: Directory to save results
        config_file: Path to FLAMES config file (optional)
        flames_path: Path to FLAMES executable
        
    Returns:
        str: Path to metrics file
    """
    logger.info("Running FLAMES benchmark")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create config file if not provided
    if not config_file:
        config_file = os.path.join(output_dir, "flames_config.json")
        with open(config_file, 'w') as f:
            config = {
                "genome_annotation": reference_gtf,
                "genome_fasta": reference_genome,
                "outdir": os.path.join(output_dir, "flames_results"),
                "reads": synthetic_fastq,
                "do_genome_alignment": True,
                "do_isoform_identification": True,
                "minimap2_path": "minimap2"
            }
            json.dump(config, f, indent=2)
    
    # Run FLAMES
    # Assuming FLAMES is installed and available in PATH
    logger.info(f"Running FLAMES with config: {config_file}")
    cmd = f"{flames_path} --config {config_file}"
    logger.info(f"Command: {cmd}")
    ret = os.system(cmd)
    
    if ret != 0:
        logger.error("FLAMES execution failed")
        return None
    
    # Load FLAMES results
    flames_count_file = os.path.join(output_dir, "flames_results/transcript_count.csv.gz")
    if not os.path.exists(flames_count_file):
        logger.error(f"FLAMES output not found: {flames_count_file}")
        return None
    
    # Run evaluation
    from .isoform_metrics import IsoformBenchmark
    
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    benchmark = IsoformBenchmark(
        ground_truth_file=ground_truth,
        results_file=flames_count_file,
        output_dir=metrics_dir
    )
    
    metrics = benchmark.run_all_evaluations()
    
    # Save metrics summary
    metrics_file = os.path.join(output_dir, "benchmark_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    
    logger.info(f"Benchmark results saved to {metrics_file}")
    
    return metrics_file


def compare_tools(synthetic_fastq: str,
                 ground_truth: str,
                 reference_genome: str,
                 reference_gtf: str,
                 output_dir: str,
                 tools: Dict[str, Dict]) -> Dict[str, str]:
    """
    Compare multiple scRNA-seq analysis tools on the same synthetic data.
    
    Args:
        synthetic_fastq: Path to synthetic FASTQ file
        ground_truth: Path to ground truth CSV file
        reference_genome: Path to reference genome FASTA file
        reference_gtf: Path to reference GTF annotation file
        output_dir: Directory to save results
        tools: Dictionary of tools to compare, with tool-specific parameters
        
    Returns:
        Dict[str, str]: Dictionary of tool names and paths to metrics files
    """
    logger.info(f"Comparing {len(tools)} tools on synthetic data")
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_files = {}
    
    # Run FLAMES if included
    if 'flames' in tools:
        flames_dir = os.path.join(output_dir, "flames")
        metrics_files['flames'] = benchmark_FLAMES(
            synthetic_fastq=synthetic_fastq,
            ground_truth=ground_truth,
            reference_genome=reference_genome,
            reference_gtf=reference_gtf,
            output_dir=flames_dir,
            **tools['flames']
        )
    
    # Run other tools
    # Add implementations for other tools as needed
    
    # Create comparison report
    if len(metrics_files) > 1:
        _create_comparison_report(metrics_files, output_dir)
    
    return metrics_files


def _create_comparison_report(metrics_files: Dict[str, str], output_dir: str):
    """
    Create a comparison report of multiple tools.
    
    Args:
        metrics_files: Dictionary of tool names and paths to metrics files
        output_dir: Directory to save report
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load metrics for each tool
    all_metrics = {}
    for tool, file_path in metrics_files.items():
        if file_path and os.path.exists(file_path):
            metrics = pd.read_csv(file_path)
            all_metrics[tool] = metrics.iloc[0].to_dict()
    
    if not all_metrics:
        logger.warning("No metrics available for comparison")
        return
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_metrics).T
    comparison_df.index.name = 'Tool'
    
    # Save comparison table
    comparison_file = os.path.join(output_dir, "tools_comparison.csv")
    comparison_df.to_csv(comparison_file)
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Select key metrics to visualize
    key_metrics = [
        'avg_f1_score', 'avg_pearson_r', 'avg_js_divergence',
        'avg_gene_f1_score', 'avg_gene_pearson_r'
    ]
    
    # Plot comparison
    for i, metric in enumerate(key_metrics):
        if metric in comparison_df.columns:
            plt.subplot(len(key_metrics), 1, i+1)
            sns.barplot(x=comparison_df.index, y=comparison_df[metric])
            plt.title(metric)
            plt.ylabel('Score')
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tools_comparison.png"), dpi=300)
    
    logger.info(f"Comparison report saved to {comparison_file}")
