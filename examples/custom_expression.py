"""
Example of creating custom expression patterns with SynthLongRead.
"""

import os
import argparse
import logging
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthlongread import SynthLongRead
from synthlongread.isoform_synth import IsoformSynthesizer

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CustomExpressionExample')

def main():
    parser = argparse.ArgumentParser(description='SynthLongRead custom expression example')
    
    # Required arguments
    parser.add_argument('--reference_transcriptome', required=True, 
                        help='Path to reference transcriptome FASTA file')
    parser.add_argument('--reference_gtf', required=True, 
                        help='Path to reference GTF annotation file')
    parser.add_argument('--real_fastq', required=True, 
                        help='Path to real FASTQ file for training')
    
    # Optional arguments
    parser.add_argument('--output_dir', default='./custom_output',
                        help='Directory to save output files')
    parser.add_argument('--n_cells', type=int, default=50,
                        help='Number of cells to simulate (total)')
    parser.add_argument('--max_reads', type=int, default=50000,
                        help='Maximum number of reads to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize SynthLongRead
    logger.info("Initializing SynthLongRead")
    synth = SynthLongRead(
        reference_transcriptome=args.reference_transcriptome,
        reference_gtf=args.reference_gtf,
        output_dir=args.output_dir
    )
    
    # Learn from real data
    logger.info("Learning from real data")
    synth.learn_from_real_data(
        real_fastq=args.real_fastq
    )
    
    # Initialize components
    synth.initialize_components()
    
    # First, let's explore the available genes and isoforms
    isoform_synth = synth.isoform_synth
    
    # Get a list of genes with multiple isoforms
    genes = list(isoform_synth.gene_isoforms.keys())
    logger.info(f"Found {len(genes)} genes with multiple isoforms")
    
    # Select a few genes for our custom expression pattern
    selected_genes = genes[:10]  # Use first 10 genes for simplicity
    
    logger.info("Selected genes for custom expression:")
    for gene in selected_genes:
        isoforms = isoform_synth.gene_isoforms[gene]
        logger.info(f"Gene {gene}: {len(isoforms)} isoforms - {', '.join(isoforms[:3])}...")
    
    # Define cell types with different expression patterns
    cell_configs = [
        {
            'cell_type': 'type_A',
            'genes': {gene: 0.8 for gene in selected_genes[:5]},  # High expression of first 5 genes
            'isoform_bias': {}  # No specific isoform bias
        },
        {
            'cell_type': 'type_B',
            'genes': {gene: 0.7 for gene in selected_genes[5:]},  # High expression of next 5 genes
            'isoform_bias': {}  # No specific isoform bias
        },
        {
            'cell_type': 'type_C',
            'genes': {gene: 0.5 for gene in selected_genes},      # Moderate expression of all genes
            'isoform_bias': {}  # No specific isoform bias
        }
    ]
    
    # Add isoform-specific bias to some cells
    for i, gene in enumerate(selected_genes[:3]):
        isoforms = isoform_synth.gene_isoforms[gene]
        if len(isoforms) >= 2:
            # Add bias for first isoform in type A, second isoform in type B
            cell_configs[0]['isoform_bias'][isoforms[0]] = 3.0
            cell_configs[1]['isoform_bias'][isoforms[1]] = 3.0
    
    # Generate custom expression matrix
    logger.info("Generating custom expression matrix")
    cell_matrix = isoform_synth.generate_custom_matrix(
        cell_configs, 
        n_cells=args.n_cells
    )
    
    # Generate synthetic dataset
    logger.info("Generating synthetic dataset with custom expression patterns")
    output_fastq = os.path.join(args.output_dir, "custom_synthetic_data.fastq")
    ground_truth = synth.fastq_generator.generate_dataset(
        cell_matrix, output_fastq, args.max_reads
    )
    
    # Save ground truth to CSV
    ground_truth_file = os.path.join(args.output_dir, "custom_ground_truth.csv")
    synth._save_ground_truth(ground_truth, ground_truth_file)
    
    logger.info(f"Generated synthetic data: {output_fastq}")
    logger.info(f"Ground truth: {ground_truth_file}")
    
    # Generate a report on the expression patterns
    logger.info("Generating expression pattern report")
    generate_expression_report(cell_matrix, args.output_dir)
    
    logger.info("Custom expression example completed successfully")

def generate_expression_report(cell_matrix, output_dir):
    """Generate a report on the expression patterns"""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Convert cell matrix to DataFrame
    rows = []
    for cell_id, isoform_counts in cell_matrix.items():
        for isoform_id, count in isoform_counts.items():
            rows.append({
                'cell_id': cell_id,
                'isoform_id': isoform_id,
                'count': count
            })
    
    df = pd.DataFrame(rows)
    
    # Extract cell type from cell_id
    cell_types = {
        'type_A': [],
        'type_B': [],
        'type_C': []
    }
    
    # Assign first third to type A, second third to type B, rest to type C
    n_cells = len(set(df['cell_id']))
    third = n_cells // 3
    
    for i, cell_id in enumerate(sorted(set(df['cell_id']))):
        if i < third:
            cell_types['type_A'].append(cell_id)
        elif i < 2 * third:
            cell_types['type_B'].append(cell_id)
        else:
            cell_types['type_C'].append(cell_id)
    
    # Create cell type column
    def get_cell_type(cell_id):
        for cell_type, cells in cell_types.items():
            if cell_id in cells:
                return cell_type
        return 'unknown'
    
    df['cell_type'] = df['cell_id'].apply(get_cell_type)
    
    # Create a pivot table to visualize the expression patterns
    pivot = df.pivot_table(
        index='isoform_id', 
        columns='cell_type', 
        values='count', 
        aggfunc='mean'
    )
    
    # Save to CSV
    pivot.to_csv(os.path.join(output_dir, 'expression_patterns.csv'))
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    
    # Select top genes by expression
    top_isoforms = pivot.mean(axis=1).sort_values(ascending=False).head(30).index
    
    # Plot heatmap for top isoforms
    im = plt.imshow(np.log1p(pivot.loc[top_isoforms]), aspect='auto', cmap='viridis')
    
    plt.colorbar(im, label='Log(count + 1)')
    plt.xlabel('Cell Type')
    plt.ylabel('Transcript ID')
    plt.title('Custom Expression Patterns')
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(top_isoforms)), top_isoforms)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expression_patterns.png'), dpi=300)

if __name__ == "__main__":
    main()
