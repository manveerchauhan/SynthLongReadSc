"""
Process real FASTQ files to extract error profiles and statistics.
"""

import os
import numpy as np
import pickle
import gzip
import logging
import pysam
from Bio import SeqIO
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DataProcessor')

# Constants
NUCLEOTIDES = ['A', 'C', 'G', 'T']
PLATFORMS = ['ONT', 'PacBio']

class DataProcessor:
    """Process real FASTQ files to extract error profiles and statistics"""
    
    def __init__(self, fastq_file: str, reference_file: str, platform: str = "ONT",
                 alignment_file: Optional[str] = None, threads: int = 4):
        """
        Initialize the DataProcessor.
        
        Args:
            fastq_file: Path to input FASTQ file
            reference_file: Path to reference sequence file
            platform: Sequencing platform ("ONT" or "PacBio")
            alignment_file: Path to existing alignment BAM file (optional)
            threads: Number of threads to use for processing
        """
        self.fastq_file = fastq_file
        self.reference_file = reference_file
        self.platform = platform
        self.alignment_file = alignment_file
        self.threads = threads
        self.read_stats = {}
        self.error_profiles = {
            'substitutions': np.zeros((4, 4)),  # A,C,G,T -> A,C,G,T
            'insertions': np.zeros(4),          # Insertion rates for A,C,G,T
            'deletions': np.zeros(4),           # Deletion rates for A,C,G,T
            'homopolymer_errors': {},           # Length -> error rate
            'position_effect': [],              # Error rate by normalized position
            'quality_by_error': []              # Quality score distribution by error type
        }
        
    def parse_fastq(self) -> Dict:
        """
        Parse FASTQ file and extract basic statistics.
        
        Returns:
            Dict: Dictionary of basic read statistics
        """
        logger.info(f"Parsing FASTQ file: {self.fastq_file}")
        
        read_lengths = []
        quality_scores = []
        gc_content = []
        
        # Use gzip if file is compressed
        open_func = gzip.open if self.fastq_file.endswith('.gz') else open
        mode = 'rt' if self.fastq_file.endswith('.gz') else 'r'
        
        with open_func(self.fastq_file, mode) as f:
            for record in SeqIO.parse(f, "fastq"):
                seq = str(record.seq)
                read_lengths.append(len(seq))
                quality_scores.extend(record.letter_annotations["phred_quality"])
                gc_content.append((seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else 0)
        
        self.read_stats["length_dist"] = np.array(read_lengths)
        self.read_stats["quality_dist"] = np.array(quality_scores)
        self.read_stats["gc_content"] = np.array(gc_content)
        self.read_stats["mean_length"] = np.mean(read_lengths)
        self.read_stats["std_length"] = np.std(read_lengths)
        self.read_stats["mean_quality"] = np.mean(quality_scores)
        
        logger.info(f"Processed {len(read_lengths)} reads")
        logger.info(f"Mean read length: {self.read_stats['mean_length']:.2f}")
        logger.info(f"Mean quality score: {self.read_stats['mean_quality']:.2f}")
        
        return self.read_stats
    
    def align_to_reference(self, aligner: str = "minimap2", output_bam: Optional[str] = None) -> str:
        """
        Align reads to reference to identify ground truth and error patterns.
        
        Args:
            aligner: Alignment tool to use (default: minimap2)
            output_bam: Path to output BAM file (optional)
            
        Returns:
            str: Path to alignment BAM file
        """
        if self.alignment_file and os.path.exists(self.alignment_file):
            logger.info(f"Using existing alignment file: {self.alignment_file}")
            return self.alignment_file
        
        if not output_bam:
            output_bam = f"{os.path.splitext(self.fastq_file)[0]}.bam"
        
        logger.info(f"Aligning reads to reference using {aligner}")
        
        # For simplicity, we assume minimap2 is installed and in the PATH
        preset = "map-ont" if self.platform == "ONT" else "map-pb"
        
        cmd = f"minimap2 -ax {preset} -t {self.threads} {self.reference_file} {self.fastq_file} | samtools sort -@ {self.threads} > {output_bam}"
        logger.info(f"Running command: {cmd}")
        os.system(cmd)
        
        cmd = f"samtools index {output_bam}"
        os.system(cmd)
        
        self.alignment_file = output_bam
        logger.info(f"Alignment completed: {self.alignment_file}")
        
        return self.alignment_file
    
    def extract_error_profiles(self) -> Dict:
        """
        Extract error profiles from alignments.
        
        Returns:
            Dict: Dictionary of error profiles
        """
        if not self.alignment_file:
            raise ValueError("Alignment file not set. Run align_to_reference first.")
        
        logger.info(f"Extracting error profiles from {self.alignment_file}")
        
        # Initialize matrices for error profiles
        sub_matrix = np.zeros((4, 4))  # A,C,G,T -> A,C,G,T
        ins_vector = np.zeros(4)      # Insertion rates for A,C,G,T
        del_vector = np.zeros(4)      # Deletion rates for A,C,G,T
        homopolymer_errors = defaultdict(list)
        pos_errors = []
        quality_by_error = []
        
        # Process BAM file
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
        
        sample_size = 10000  # Sample reads to make it faster
        
        with pysam.AlignmentFile(self.alignment_file, "rb") as bam:
            read_count = 0
            
            for read in bam.fetch():
                if read_count >= sample_size:
                    break
                    
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                
                read_count += 1
                if read_count % 1000 == 0:
                    logger.info(f"Processed {read_count} reads")
                
                # Extract aligned segments
                reference_seq = read.get_reference_sequence().upper()
                query_seq = read.query_sequence.upper()
                quality = read.query_qualities
                
                # Process CIGAR string to identify errors
                ref_pos = 0
                query_pos = 0
                
                for op, length in read.cigartuples:
                    # Match or mismatch
                    if op == 0:  # M
                        for i in range(length):
                            ref_base = reference_seq[ref_pos + i]
                            query_base = query_seq[query_pos + i]
                            
                            # Only consider ACGT
                            if ref_base in "ACGT" and query_base in "ACGT":
                                rel_pos = (ref_pos + i) / len(reference_seq)
                                pos_errors.append((rel_pos, ref_base != query_base))
                                
                                if ref_base != query_base:  # Substitution
                                    sub_matrix[base_to_idx[ref_base], base_to_idx[query_base]] += 1
                                    if query_pos + i < len(quality):
                                        quality_by_error.append(('sub', quality[query_pos + i]))
                                
                                # Check for homopolymers
                                if i > 0 and ref_pos + i < len(reference_seq) - 1:
                                    if reference_seq[ref_pos + i - 1] == ref_base and reference_seq[ref_pos + i + 1] != ref_base:
                                        # End of a homopolymer
                                        j = 1
                                        while ref_pos + i - j >= 0 and reference_seq[ref_pos + i - j] == ref_base:
                                            j += 1
                                        homopolymer_len = j
                                        homopolymer_errors[homopolymer_len].append(ref_base != query_base)
                        
                        ref_pos += length
                        query_pos += length
                    
                    # Insertion in read
                    elif op == 1:  # I
                        for i in range(length):
                            if query_pos + i < len(query_seq) and query_seq[query_pos + i] in "ACGT":
                                ins_vector[base_to_idx[query_seq[query_pos + i]]] += 1
                                if query_pos + i < len(quality):
                                    quality_by_error.append(('ins', quality[query_pos + i]))
                        
                        query_pos += length
                    
                    # Deletion in read
                    elif op == 2:  # D
                        for i in range(length):
                            if ref_pos + i < len(reference_seq) and reference_seq[ref_pos + i] in "ACGT":
                                del_vector[base_to_idx[reference_seq[ref_pos + i]]] += 1
                                # No quality score for deletions
                        
                        ref_pos += length
                    
                    # Skip other CIGAR operations
                    else:
                        if op == 4:  # Soft clip
                            query_pos += length
                        elif op == 3:  # N (skipped region)
                            ref_pos += length
        
        # Normalize error matrices
        row_sums = sub_matrix.sum(axis=1)
        self.error_profiles['substitutions'] = np.divide(sub_matrix, row_sums[:, np.newaxis], 
                                                     where=row_sums[:, np.newaxis]!=0)
        
        total_bases = row_sums + ins_vector + del_vector
        self.error_profiles['insertions'] = np.divide(ins_vector, total_bases, where=total_bases!=0)
        self.error_profiles['deletions'] = np.divide(del_vector, total_bases, where=total_bases!=0)
        
        # Process homopolymer errors
        for length, errors in homopolymer_errors.items():
            self.error_profiles['homopolymer_errors'][length] = np.mean(errors) if errors else 0
        
        # Process position effect
        pos_bins = np.linspace(0, 1, 20)
        binned_pos_errors = [[] for _ in range(19)]
        
        for pos, is_error in pos_errors:
            bin_idx = min(int(pos * 19), 18)
            binned_pos_errors[bin_idx].append(is_error)
        
        self.error_profiles['position_effect'] = [np.mean(errors) if errors else 0 for errors in binned_pos_errors]
        
        # Process quality scores by error type
        self.error_profiles['quality_by_error'] = {
            'sub': [q for t, q in quality_by_error if t == 'sub'],
            'ins': [q for t, q in quality_by_error if t == 'ins']
        }
        
        logger.info("Error profile extraction completed")
        return self.error_profiles
    
    def save_profiles(self, output_file: str):
        """
        Save extracted profiles to a file.
        
        Args:
            output_file: Path to output file
        """
        with open(output_file, 'wb') as f:
            pickle.dump({
                'read_stats': self.read_stats,
                'error_profiles': self.error_profiles,
                'platform': self.platform
            }, f)
        logger.info(f"Profiles saved to {output_file}")
    
    @classmethod
    def load_profiles(cls, input_file: str) -> 'DataProcessor':
        """
        Load profiles from a file.
        
        Args:
            input_file: Path to input file
            
        Returns:
            DataProcessor: DataProcessor instance with loaded profiles
        """
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        
        processor = cls("", "")  # Empty initialization
        processor.read_stats = data['read_stats']
        processor.error_profiles = data['error_profiles']
        processor.platform = data['platform']
        
        logger.info(f"Profiles loaded from {input_file}")
        return processor
