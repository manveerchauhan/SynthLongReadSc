"""
Evaluate read-level characteristics of synthetic data compared to real data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import gzip
from Bio import SeqIO
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ReadLevelMetrics')

class ReadLevelMetrics:
    """Evaluate read-level characteristics of synthetic data compared to real data"""
    
    def __init__(self, synthetic_fastq: str, real_fastq: str, output_dir: str):
        """
        Initialize ReadLevelMetrics.
        
        Args:
            synthetic_fastq: Path to synthetic FASTQ file
            real_fastq: Path to real FASTQ file
            output_dir: Directory to save results
        """
        self.synthetic_fastq = synthetic_fastq
        self.real_fastq = real_fastq
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_read_stats(self, fastq_file: str, max_reads: int = 100000) -> dict:
        """
        Extract read statistics from a FASTQ file.
        
        Args:
            fastq_file: Path to FASTQ file
            max_reads: Maximum number of reads to process
            
        Returns:
            dict: Dictionary of read statistics
        """
        logger.info(f"Extracting read stats from: {fastq_file}")
        
        # Statistics to collect
        stats = {
            'read_lengths': [],
            'quality_scores': [],
            'gc_content': [],
            'homopolymer_counts': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, '10+': 0},
            'base_composition': {'A': 0, 'C': 0, 'G': 0, 'T': 0, 'N': 0}
        }
        
        # Determine if file is gzipped
        is_gzipped = fastq_file.endswith('.gz')
        open_func = gzip.open if is_gzipped else open
        mode = 'rt' if is_gzipped else 'r'
        
        # Process FASTQ file
        read_count = 0
        with open_func(fastq_file, mode) as f:
            for record in SeqIO.parse(f, "fastq"):
                read_count += 1
                
                if read_count % 10000 == 0:
                    logger.info(f"Processed {read_count} reads")
                
                # Read length
                seq = str(record.seq)
                stats['read_lengths'].append(len(seq))
                
                # Quality scores
                stats['quality_scores'].extend(record.letter_annotations["phred_quality"])
                
                # GC content
                gc = (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else 0
                stats['gc_content'].append(gc)
                
                # Base composition
                for base in 'ACGTN':
                    stats['base_composition'][base] += seq.count(base)
                
                # Homopolymer runs
                for base in 'ACGT':
                    i = 0
                    while i < len(seq):
                        if seq[i] != base:
                            i += 1
                            continue
                        
                        # Count length of homopolymer
                        j = i
                        while j < len(seq) and seq[j] == base:
                            j += 1
                        
                        run_length = j - i
                        if run_length >= 10:
                            stats['homopolymer_counts']['10+'] += 1
                        elif run_length > 1:  # Only count runs of 2 or more
                            stats['homopolymer_counts'][run_length] += 1
                        
                        i = j
                
                # Limit the number of reads to process
                if read_count >= max_reads:
                    break
        
        # Calculate percentages for base composition
        total_bases = sum(stats['base_composition'].values())
        for base in stats['base_composition']:
            stats['base_composition'][base] /= total_bases
        
        logger.info(f"Processed total of {read_count} reads")
        return stats
    
    def compare_stats(self, max_reads: int = 100000) -> Dict:
        """
        Compare statistics between synthetic and real data.
        
        Args:
            max_reads: Maximum number of reads to process
            
        Returns:
            Dict: Dictionary of statistics for synthetic and real data
        """
        # Extract stats
        synthetic_stats = self.extract_read_stats(self.synthetic_fastq, max_reads)
        real_stats = self.extract_read_stats(self.real_fastq, max_reads)
        
        # Compare and visualize
        self._compare_read_lengths(synthetic_stats, real_stats)
        self._compare_quality_scores(synthetic_stats, real_stats)
        self._compare_gc_content(synthetic_stats, real_stats)
        self._compare_homopolymers(synthetic_stats, real_stats)
        self._compare_base_composition(synthetic_stats, real_stats)
        
        return {
            'synthetic_stats': synthetic_stats,
            'real_stats': real_stats
        }
    
    def _compare_read_lengths(self, synthetic_stats, real_stats):
        """
        Compare read length distributions.
        
        Args:
            synthetic_stats: Dictionary of synthetic read statistics
            real_stats: Dictionary of real read statistics
        """
        plt.figure(figsize=(12, 6))
        
        # Convert to numpy arrays
        syn_lengths = np.array(synthetic_stats['read_lengths'])
        real_lengths = np.array(real_stats['read_lengths'])
        
        # Plot histograms
        bins = np.linspace(0, max(syn_lengths.max(), real_lengths.max()), 50)
        plt.hist(real_lengths, bins=bins, alpha=0.5, label='Real Data', density=True)
        plt.hist(syn_lengths, bins=bins, alpha=0.5, label='Synthetic Data', density=True)
        
        plt.xlabel('Read Length')
        plt.ylabel('Density')
        plt.title('Read Length Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics as text
        stats_text = (
            f"Real Data: Mean={real_lengths.mean():.1f}, Median={np.median(real_lengths):.1f}, SD={real_lengths.std():.1f}\n"
            f"Synthetic: Mean={syn_lengths.mean():.1f}, Median={np.median(syn_lengths):.1f}, SD={syn_lengths.std():.1f}"
        )
        plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'read_length_comparison.png'), dpi=300)
        plt.close()
    
    def _compare_quality_scores(self, synthetic_stats, real_stats):
        """
        Compare quality score distributions.
        
        Args:
            synthetic_stats: Dictionary of synthetic read statistics
            real_stats: Dictionary of real read statistics
        """
        plt.figure(figsize=(12, 6))
        
        # Convert to numpy arrays
        syn_quality = np.array(synthetic_stats['quality_scores'])
        real_quality = np.array(real_stats['quality_scores'])
        
        # Plot histograms
        bins = np.arange(0, 42)
        plt.hist(real_quality, bins=bins, alpha=0.5, label='Real Data', density=True)
        plt.hist(syn_quality, bins=bins, alpha=0.5, label='Synthetic Data', density=True)
        
        plt.xlabel('Quality Score (Phred)')
        plt.ylabel('Density')
        plt.title('Quality Score Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics as text
        stats_text = (
            f"Real Data: Mean={real_quality.mean():.1f}, Median={np.median(real_quality):.1f}\n"
            f"Synthetic: Mean={syn_quality.mean():.1f}, Median={np.median(syn_quality):.1f}"
        )
        plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'quality_score_comparison.png'), dpi=300)
        plt.close()
    
    def _compare_gc_content(self, synthetic_stats, real_stats):
        """
        Compare GC content distributions.
        
        Args:
            synthetic_stats: Dictionary of synthetic read statistics
            real_stats: Dictionary of real read statistics
        """
        plt.figure(figsize=(12, 6))
        
        # Convert to numpy arrays
        syn_gc = np.array(synthetic_stats['gc_content'])
        real_gc = np.array(real_stats['gc_content'])
        
        # Plot histograms
        bins = np.linspace(0, 1, 20)
        plt.hist(real_gc, bins=bins, alpha=0.5, label='Real Data', density=True)
        plt.hist(syn_gc, bins=bins, alpha=0.5, label='Synthetic Data', density=True)
        
        plt.xlabel('GC Content')
        plt.ylabel('Density')
        plt.title('GC Content Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics as text
        stats_text = (
            f"Real Data: Mean={real_gc.mean():.3f}, Median={np.median(real_gc):.3f}\n"
            f"Synthetic: Mean={syn_gc.mean():.3f}, Median={np.median(syn_gc):.3f}"
        )
        plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'gc_content_comparison.png'), dpi=300)
        plt.close()
    
    def _compare_homopolymers(self, synthetic_stats, real_stats):
        """
        Compare homopolymer distributions.
        
        Args:
            synthetic_stats: Dictionary of synthetic read statistics
            real_stats: Dictionary of real read statistics
        """
        plt.figure(figsize=(12, 6))
        
        # Extract homopolymer counts
        syn_homo = synthetic_stats['homopolymer_counts']
        real_homo = real_stats['homopolymer_counts']
        
        # Normalize to percentages
        syn_total = sum(syn_homo.values())
        real_total = sum(real_homo.values())
        
        syn_pct = {k: v / syn_total * 100 for k, v in syn_homo.items()}
        real_pct = {k: v / real_total * 100 for k, v in real_homo.items()}
        
        # Prepare data for plotting
        x = list(syn_homo.keys())
        x_pos = np.arange(len(x))
        
        syn_values = [syn_pct[k] for k in x]
        real_values = [real_pct[k] for k in x]
        
        # Plot as grouped bar chart
        width = 0.35
        plt.bar(x_pos - width/2, real_values, width, label='Real Data')
        plt.bar(x_pos + width/2, syn_values, width, label='Synthetic Data')
        
        plt.xlabel('Homopolymer Length')
        plt.ylabel('Percentage')
        plt.title('Homopolymer Distribution Comparison')
        plt.xticks(x_pos, x)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'homopolymer_comparison.png'), dpi=300)
        plt.close()
    
    def _compare_base_composition(self, synthetic_stats, real_stats):
        """
        Compare base composition.
        
        Args:
            synthetic_stats: Dictionary of synthetic read statistics
            real_stats: Dictionary of real read statistics
        """
        plt.figure(figsize=(10, 6))
        
        # Extract base compositions
        syn_bases = synthetic_stats['base_composition']
        real_bases = real_stats['base_composition']
        
        # Prepare data for plotting
        bases = ['A', 'C', 'G', 'T', 'N']
        x_pos = np.arange(len(bases))
        
        syn_values = [syn_bases[b] * 100 for b in bases]
        real_values = [real_bases[b] * 100 for b in bases]
        
        # Plot as grouped bar chart
        width = 0.35
        plt.bar(x_pos - width/2, real_values, width, label='Real Data')
        plt.bar(x_pos + width/2, syn_values, width, label='Synthetic Data')
        
        plt.xlabel('Base')
        plt.ylabel('Percentage')
        plt.title('Base Composition Comparison')
        plt.xticks(x_pos, bases)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'base_composition_comparison.png'), dpi=300)
        plt.close()
    
    def generate_error_rate_report(self, alignment_file: str, 
                                  synthetic_alignment_file: str) -> Dict:
        """
        Generate error rate report based on alignments.
        
        Args:
            alignment_file: Path to real data alignment BAM file
            synthetic_alignment_file: Path to synthetic data alignment BAM file
            
        Returns:
            Dict: Dictionary of error statistics
        """
        import pysam
        
        logger.info("Generating error rate report...")
        
        # Error statistics to collect
        real_errors = {'sub': 0, 'ins': 0, 'del': 0, 'total_bases': 0}
        synth_errors = {'sub': 0, 'ins': 0, 'del': 0, 'total_bases': 0}
        
        # Process real data alignment
        with pysam.AlignmentFile(alignment_file, "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                
                # Count errors from CIGAR
                for op, length in read.cigartuples:
                    if op == 0:  # M (match or mismatch)
                        real_errors['total_bases'] += length
                        # Count mismatches (substitutions)
                        if read.has_tag('NM'):
                            real_errors['sub'] += read.get_tag('NM')
                    elif op == 1:  # I (insertion)
                        real_errors['ins'] += length
                    elif op == 2:  # D (deletion)
                        real_errors['del'] += length
        
        # Process synthetic data alignment
        with pysam.AlignmentFile(synthetic_alignment_file, "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                
                # Count errors from CIGAR
                for op, length in read.cigartuples:
                    if op == 0:  # M (match or mismatch)
                        synth_errors['total_bases'] += length
                        # Count mismatches (substitutions)
                        if read.has_tag('NM'):
                            synth_errors['sub'] += read.get_tag('NM')
                    elif op == 1:  # I (insertion)
                        synth_errors['ins'] += length
                    elif op == 2:  # D (deletion)
                        synth_errors['del'] += length
        
        # Calculate error rates
        real_error_rates = {
            'sub_rate': real_errors['sub'] / real_errors['total_bases'],
            'ins_rate': real_errors['ins'] / real_errors['total_bases'],
            'del_rate': real_errors['del'] / real_errors['total_bases'],
            'total_error_rate': (real_errors['sub'] + real_errors['ins'] + real_errors['del']) / real_errors['total_bases']
        }
        
        synth_error_rates = {
            'sub_rate': synth_errors['sub'] / synth_errors['total_bases'],
            'ins_rate': synth_errors['ins'] / synth_errors['total_bases'],
            'del_rate': synth_errors['del'] / synth_errors['total_bases'],
            'total_error_rate': (synth_errors['sub'] + synth_errors['ins'] + synth_errors['del']) / synth_errors['total_bases']
        }
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        
        error_types = ['sub_rate', 'ins_rate', 'del_rate', 'total_error_rate']
        x_pos = np.arange(len(error_types))
        
        real_values = [real_error_rates[e] * 100 for e in error_types]
        synth_values = [synth_error_rates[e] * 100 for e in error_types]
        
        width = 0.35
        plt.bar(x_pos - width/2, real_values, width, label='Real Data')
        plt.bar(x_pos + width/2, synth_values, width, label='Synthetic Data')
        
        plt.xlabel('Error Type')
        plt.ylabel('Error Rate (%)')
        plt.title('Error Rate Comparison')
        plt.xticks(x_pos, ['Substitution', 'Insertion', 'Deletion', 'Total'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'error_rate_comparison.png'), dpi=300)
        plt.close()
        
        return {
            'real_error_rates': real_error_rates,
            'synth_error_rates': synth_error_rates
        }
