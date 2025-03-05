"""
Generate realistic isoform expression patterns across cells.
"""

import os
import numpy as np
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Optional, Union
from Bio import SeqIO

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('IsoformSynthesizer')

class IsoformSynthesizer:
    """Generate realistic isoform expression patterns across cells"""
    
    def __init__(self, reference_gtf: str, cell_types: Optional[List[str]] = None):
        """
        Initialize the IsoformSynthesizer.
        
        Args:
            reference_gtf: Path to reference GTF annotation file
            cell_types: List of cell types to simulate (optional)
        """
        self.reference_gtf = reference_gtf
        self.cell_types = cell_types
        self.gene_isoforms = self._parse_gtf()
        self.gene_lengths = {}
        self.isoform_sequences = {}
    
    def _parse_gtf(self) -> Dict[str, List[str]]:
        """
        Parse GTF file to extract gene-isoform relationships.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping gene IDs to lists of transcript IDs
        """
        logger.info(f"Parsing GTF file: {self.reference_gtf}")
        
        gene_isoforms = defaultdict(list)
        
        with open(self.reference_gtf, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                    
                if fields[2] == 'transcript':
                    # Extract gene and transcript IDs from attributes
                    attributes = fields[8]
                    gene_id = None
                    transcript_id = None
                    
                    for attr in attributes.split(';'):
                        attr = attr.strip()
                        if attr.startswith('gene_id'):
                            gene_id = attr.split('"')[1] if '"' in attr else attr.split(' ')[1]
                        elif attr.startswith('transcript_id'):
                            transcript_id = attr.split('"')[1] if '"' in attr else attr.split(' ')[1]
                    
                    if gene_id and transcript_id:
                        gene_isoforms[gene_id].append(transcript_id)
        
        # Remove genes with only one isoform (not interesting for benchmarking)
        gene_isoforms = {g: isoforms for g, isoforms in gene_isoforms.items() if len(isoforms) > 1}
        
        logger.info(f"Found {len(gene_isoforms)} genes with multiple isoforms")
        return gene_isoforms
    
    def load_transcript_sequences(self, transcriptome_file: str):
        """
        Load transcript sequences from a FASTA file.
        
        Args:
            transcriptome_file: Path to transcriptome FASTA file
        """
        logger.info(f"Loading transcript sequences from {transcriptome_file}")
        
        self.isoform_sequences = {}
        
        with open(transcriptome_file, 'r') as f:
            for record in SeqIO.parse(f, "fasta"):
                transcript_id = record.id.split('|')[0].split('.')[0]
                self.isoform_sequences[transcript_id] = str(record.seq)
                self.gene_lengths[transcript_id] = len(record.seq)
        
        logger.info(f"Loaded {len(self.isoform_sequences)} transcript sequences")
    
    def generate_cell_matrix(self, n_cells: int, sparsity: float = 0.8) -> Dict[str, Dict[str, int]]:
        """
        Generate gene-isoform-cell expression matrix.
        
        Args:
            n_cells: Number of cells to simulate
            sparsity: Gene expression sparsity (0-1, higher means more zeros)
            
        Returns:
            Dict[str, Dict[str, int]]: Dictionary mapping cell IDs to dictionaries of transcript counts
        """
        logger.info(f"Generating expression matrix for {n_cells} cells")
        
        genes = list(self.gene_isoforms.keys())
        n_genes = len(genes)
        
        # Optionally assign cell types
        cell_types = None
        if self.cell_types:
            cell_types = np.random.choice(self.cell_types, n_cells)
        
        # Decide which genes are expressed in each cell
        gene_expression = np.random.binomial(1, 1-sparsity, (n_cells, n_genes))
        
        # For expressed genes, generate isoform distributions
        cell_matrix = {}
        
        for cell_idx in range(n_cells):
            cell_isoforms = {}
            cell_type = cell_types[cell_idx] if cell_types is not None else None
            
            for gene_idx, is_expressed in enumerate(gene_expression[cell_idx]):
                if is_expressed:
                    gene = genes[gene_idx]
                    isoforms = self.gene_isoforms[gene]
                    
                    # Generate relative isoform expression
                    # Use Dirichlet distribution with alpha < 1 to create sparsity
                    alpha = np.ones(len(isoforms)) * 0.3
                    isoform_weights = np.random.dirichlet(alpha)
                    
                    # Generate absolute counts (using Negative Binomial)
                    # Typical scRNA-seq has mean counts of ~1-10 per gene
                    r = 0.5  # dispersion parameter (lower = more dispersion)
                    p = 0.1  # success probability
                    total_count = np.random.negative_binomial(r, p)
                    
                    # Scale count (depends on cell type, sequencing depth, etc.)
                    scale_factor = np.random.lognormal(0, 0.5)  # biological variability
                    if cell_type:
                        # Add cell-type specific expression bias
                        # This is just a placeholder - in reality would be more complex
                        cell_type_factor = hash(f"{gene}_{cell_type}") % 10 / 10.0 + 0.5
                        scale_factor *= cell_type_factor
                    
                    total_count = max(1, int(total_count * scale_factor))
                    
                    # Ensure at least one count if gene is "expressed"
                    if total_count == 0:
                        total_count = 1
                    
                    # Generate counts for each isoform
                    if total_count > 0:
                        isoform_counts = np.random.multinomial(total_count, isoform_weights)
                    else:
                        isoform_counts = np.zeros(len(isoforms), dtype=int)
                    
                    # Store in cell matrix
                    for isoform_idx, count in enumerate(isoform_counts):
                        if count > 0:
                            isoform = isoforms[isoform_idx]
                            cell_isoforms[isoform] = count
            
            cell_matrix[f"cell_{cell_idx}"] = cell_isoforms
        
        return cell_matrix
    
    def generate_custom_matrix(self, 
                              cell_config: List[Dict],
                              n_cells: int = None) -> Dict[str, Dict[str, int]]:
        """
        Generate a custom expression matrix based on specific configuration.
        
        Args:
            cell_config: List of cell configuration dictionaries with keys:
                - 'cell_type': Optional cell type name
                - 'genes': Dict of gene IDs and their expression levels (0-1)
                - 'isoform_bias': Optional dict of transcript IDs and their bias factors
            n_cells: Total number of cells to generate (if None, uses len(cell_config))
            
        Returns:
            Dict[str, Dict[str, int]]: Dictionary mapping cell IDs to dictionaries of transcript counts
        """
        if n_cells is None:
            n_cells = len(cell_config)
        
        logger.info(f"Generating custom expression matrix for {n_cells} cells")
        
        # Repeat configurations if n_cells > len(cell_config)
        if n_cells > len(cell_config):
            # Repeat configurations with slight variation
            extended_config = []
            for i in range(n_cells):
                base_config = cell_config[i % len(cell_config)]
                
                # Create a slightly varied version
                varied_config = base_config.copy()
                
                # Add slight variation to gene expression levels
                if 'genes' in varied_config:
                    varied_genes = {}
                    for gene, level in varied_config['genes'].items():
                        # Add noise to expression level
                        noise = np.random.normal(0, 0.1)
                        varied_genes[gene] = max(0, min(1, level + noise))
                    varied_config['genes'] = varied_genes
                
                extended_config.append(varied_config)
            
            cell_config = extended_config
        
        # Generate matrix
        cell_matrix = {}
        
        for cell_idx, config in enumerate(cell_config[:n_cells]):
            cell_type = config.get('cell_type', None)
            gene_expr = config.get('genes', {})
            isoform_bias = config.get('isoform_bias', {})
            
            cell_isoforms = {}
            
            # Process each specified gene
            for gene, expr_level in gene_expr.items():
                if gene in self.gene_isoforms and expr_level > 0:
                    isoforms = self.gene_isoforms[gene]
                    
                    # Generate relative isoform expression with optional bias
                    alpha = np.ones(len(isoforms)) * 0.3
                    
                    # Apply isoform bias if specified
                    for i, isoform in enumerate(isoforms):
                        if isoform in isoform_bias:
                            alpha[i] *= isoform_bias[isoform]
                    
                    isoform_weights = np.random.dirichlet(alpha)
                    
                    # Scale total counts by expression level
                    # Higher expr_level means more counts
                    base_count = np.random.negative_binomial(0.5, 0.1)
                    total_count = max(1, int(base_count * expr_level * 10))
                    
                    # Generate counts for each isoform
                    isoform_counts = np.random.multinomial(total_count, isoform_weights)
                    
                    # Store in cell matrix
                    for isoform_idx, count in enumerate(isoform_counts):
                        if count > 0:
                            isoform = isoforms[isoform_idx]
                            cell_isoforms[isoform] = count
            
            cell_matrix[f"cell_{cell_idx}"] = cell_isoforms
        
        return cell_matrix
    
    def get_transcript_length(self, transcript_id: str) -> int:
        """
        Get the length of a transcript.
        
        Args:
            transcript_id: Transcript ID
            
        Returns:
            int: Transcript length
        """
        if transcript_id in self.gene_lengths:
            return self.gene_lengths[transcript_id]
        elif transcript_id in self.isoform_sequences:
            return len(self.isoform_sequences[transcript_id])
        else:
            logger.warning(f"Transcript length not found for {transcript_id}")
            return 1000  # Default length
