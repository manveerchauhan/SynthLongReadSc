"""
Generate synthetic FASTQ reads with realistic errors.
"""

import os
import numpy as np
import torch
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FASTQGenerator')

class FASTQGenerator:
    """Generate synthetic FASTQ reads with realistic errors"""
    
    def __init__(self, 
                 error_model_trainer,
                 isoform_synth,
                 cell_barcode_synth,
                 adapter_5p: str = "CCCATGTACTCTGCGTTGATACCACTGCTT",  # generic adapter 
                 adapter_3p: str = "AAAAAAAAAAAAAAAAAA",             # poly-A tail
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the FASTQGenerator.
        
        Args:
            error_model_trainer: Trained ErrorModelTrainer instance
            isoform_synth: IsoformSynthesizer instance
            cell_barcode_synth: CellBarcodeSynthesizer instance
            adapter_5p: 5' adapter sequence
            adapter_3p: 3' adapter sequence (poly-A tail)
            device: Device to use for inference ('cuda' or 'cpu')
        """
        self.error_model_trainer = error_model_trainer
        self.isoform_synth = isoform_synth
        self.cell_barcode_synth = cell_barcode_synth
        self.adapter_5p = adapter_5p
        self.adapter_3p = adapter_3p
        self.device = device
        
        # Ensure models are in evaluation mode
        self.error_model_trainer.seq_error_model.eval()
        self.error_model_trainer.quality_model.eval()
    
    def one_hot_encode(self, seq: str) -> List[float]:
        """
        One-hot encode a sequence.
        
        Args:
            seq: Input nucleotide sequence
            
        Returns:
            List[float]: One-hot encoded sequence
        """
        encoding = []
        for base in seq:
            if base == 'A':
                encoding.extend([1, 0, 0, 0])
            elif base == 'C':
                encoding.extend([0, 1, 0, 0])
            elif base == 'G':
                encoding.extend([0, 0, 1, 0])
            elif base == 'T':
                encoding.extend([0, 0, 0, 1])
            else:
                encoding.extend([0.25, 0.25, 0.25, 0.25])  # Unknown base
        return encoding
    
    def generate_read_with_errors(self, sequence: str, context_size: int = 5) -> Tuple[str, List[int]]:
        """
        Generate a synthetic read with realistic errors.
        
        Args:
            sequence: Input sequence
            context_size: Size of sequence context to consider
            
        Returns:
            Tuple[str, List[int]]: Output sequence and quality scores
        """
        with torch.no_grad():
            # Initialize output sequence and quality scores
            output_seq = []
            quality_scores = []
            
            # Add padding for context
            padded_seq = "N" * (context_size // 2) + sequence + "N" * (context_size // 2)
            
            i = 0
            while i < len(padded_seq) - context_size:
                # Extract context
                context = padded_seq[i:i + context_size]
                center_idx = context_size // 2
                center_base = context[center_idx]
                
                # Skip if center base is not ACGT
                if center_base not in "ACGT":
                    output_seq.append(center_base)
                    quality_scores.append(10)  # Low quality for non-ACGT
                    i += 1
                    continue
                
                # Normalized position in read
                rel_pos = i / (len(padded_seq) - context_size)
                
                # One-hot encode context
                context_vec = self.one_hot_encode(context)
                
                # Prepare inputs for models
                context_tensor = torch.tensor([context_vec], dtype=torch.float32).to(self.device)
                pos_tensor = torch.tensor([[rel_pos]], dtype=torch.float32).to(self.device)
                
                # Get error probabilities
                error_probs = self.error_model_trainer.seq_error_model(context_tensor, pos_tensor)
                error_probs = error_probs.squeeze().cpu().numpy()
                
                # Map center base to index
                base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
                base_idx = base_to_idx[center_base]
                
                # Extract probabilities for this base
                probs = error_probs[base_idx]
                
                # Sample error type based on probabilities
                error_type_idx = np.random.choice(3, p=probs)  # 0=match, 1=sub, 2=del
                
                if error_type_idx == 0:  # Match
                    output_seq.append(center_base)
                    
                    # Generate quality score
                    error_type_onehot = [1, 0, 0]
                    quality_tensor = self.error_model_trainer.quality_model(
                        context_tensor, 
                        torch.tensor([[1, 0, 0]], dtype=torch.float32).to(self.device),
                        pos_tensor
                    )
                    quality_scores.append(int(quality_tensor.item()))
                    
                    i += 1
                    
                elif error_type_idx == 1:  # Substitution
                    # Sample substitution target
                    options = list(set("ACGT") - {center_base})
                    sub_base = np.random.choice(options)
                    output_seq.append(sub_base)
                    
                    # Generate quality score
                    error_type_onehot = [0, 1, 0]
                    quality_tensor = self.error_model_trainer.quality_model(
                        context_tensor, 
                        torch.tensor([[0, 1, 0]], dtype=torch.float32).to(self.device),
                        pos_tensor
                    )
                    quality_scores.append(int(quality_tensor.item()))
                    
                    i += 1
                    
                elif error_type_idx == 2:  # Deletion
                    # Skip this base (deletion)
                    i += 1
                    continue
                
                # Occasionally add insertions (not directly from model but at observed rate)
                if np.random.random() < 0.01:  # 1% insertion rate
                    ins_base = np.random.choice(['A', 'C', 'G', 'T'])
                    output_seq.append(ins_base)
                    
                    # Generate quality score for insertion (typically lower)
                    quality_scores.append(max(1, int(np.random.normal(15, 5))))
            
            return ''.join(output_seq), quality_scores
    
    def _generate_read_length(self) -> int:
        """
        Sample read length from learned distribution.
        
        Returns:
            int: Generated read length
        """
        if self.error_model_trainer.read_length_model:
            length, = self.error_model_trainer.read_length_model.sample(1)[0]
            return max(100, int(length))
        else:
            # Fallback to normal distribution if no model
            return max(100, int(np.random.normal(1000, 300)))
    
    def generate_reads_from_transcript(self, 
                                      transcript_id: str, 
                                      n_reads: int,
                                      cell_barcode: Optional[str] = None) -> List[Tuple[str, List[int], str]]:
        """
        Generate reads from a specific transcript.
        
        Args:
            transcript_id: Transcript ID
            n_reads: Number of reads to generate
            cell_barcode: Cell barcode to add (optional)
            
        Returns:
            List[Tuple[str, List[int], str]]: List of (sequence, quality scores, read ID) tuples
        """
        if transcript_id not in self.isoform_synth.isoform_sequences:
            logger.warning(f"Transcript {transcript_id} not found in sequences")
            return []
        
        transcript_seq = self.isoform_synth.isoform_sequences[transcript_id]
        
        reads = []
        for _ in range(n_reads):
            # Determine read length
            read_length = self._generate_read_length()
            
            # For full-length protocols (like PacBio IsoSeq), try to get the whole transcript
            # For fragmented protocols (like ONT dRNA-seq), get a random fragment
            if np.random.random() < 0.3:  # 30% chance of full-length (adjust as needed)
                start = 0
                end = min(len(transcript_seq), read_length)
            else:
                # Random fragment
                if len(transcript_seq) <= read_length:
                    start = 0
                    end = len(transcript_seq)
                else:
                    start = np.random.randint(0, len(transcript_seq) - read_length)
                    end = start + read_length
            
            # Extract fragment
            fragment = transcript_seq[start:end]
            
            # Add adapters and barcode if scRNA-seq
            if cell_barcode:
                # Add barcode with potential errors
                if np.random.random() < 0.1:  # 10% chance of barcode error
                    barcode = self.cell_barcode_synth.introduce_barcode_errors(cell_barcode)
                else:
                    barcode = cell_barcode
                
                # Assemble read with adapters and barcode
                full_seq = self.adapter_5p + barcode + fragment + self.adapter_3p
            else:
                full_seq = fragment
            
            # Introduce errors using model
            read_seq, quality = self.generate_read_with_errors(full_seq)
            
            # Create read ID with ground truth info
            read_id = f"{transcript_id}_{start}_{end}"
            if cell_barcode:
                read_id += f"_cell_{cell_barcode}"
            
            reads.append((read_seq, quality, read_id))
        
        return reads
    
    def generate_cell_reads(self, 
                           cell_id: str, 
                           isoform_counts: Dict[str, int]) -> List[Tuple[str, List[int], str]]:
        """
        Generate reads for a single cell.
        
        Args:
            cell_id: Cell ID
            isoform_counts: Dictionary mapping transcript IDs to counts
            
        Returns:
            List[Tuple[str, List[int], str]]: List of (sequence, quality scores, read ID) tuples
        """
        all_reads = []
        
        # Generate cell barcode
        cell_barcode = self.cell_barcode_synth.generate_barcodes(1)[0]
        
        for transcript_id, count in isoform_counts.items():
            # Generate reads based on expression level
            # Higher expression → more reads
            n_reads = int(np.random.poisson(count * 2))  # Adjust scaling factor as needed
            
            reads = self.generate_reads_from_transcript(
                transcript_id, n_reads, cell_barcode
            )
            
            all_reads.extend(reads)
        
        return all_reads
    
    def write_fastq(self, reads: List[Tuple[str, List[int], str]], output_file: str):
        """
        Write reads to FASTQ file.
        
        Args:
            reads: List of (sequence, quality scores, read ID) tuples
            output_file: Output FASTQ file path
        """
        with open(output_file, 'w') as f:
            for i, (seq, quality, read_id) in enumerate(reads):
                # Format FASTQ entry
                f.write(f"@{read_id}_{i}\n")
                f.write(f"{seq}\n")
                f.write("+\n")
                
                # Convert quality scores to ASCII (Phred+33)
                qual_str = ''.join(chr(min(q, 41) + 33) for q in quality)
                f.write(f"{qual_str}\n")
    
    def generate_dataset(self, 
                        cell_matrix: Dict[str, Dict[str, int]],
                        output_file: str,
                        max_reads: Optional[int] = None) -> Dict[str, Dict[str, int]]:
        """
        Generate complete dataset from cell expression matrix.
        
        Args:
            cell_matrix: Dictionary mapping cell IDs to dictionaries of transcript counts
            output_file: Output FASTQ file path
            max_reads: Maximum number of reads to generate (optional)
            
        Returns:
            Dict[str, Dict[str, int]]: Ground truth expression matrix
        """
        logger.info(f"Generating dataset with {len(cell_matrix)} cells")
        
        all_reads = []
        ground_truth = {}
        
        for cell_id, isoform_counts in tqdm(cell_matrix.items()):
            reads = self.generate_cell_reads(cell_id, isoform_counts)
            all_reads.extend(reads)
            
            # Track ground truth
            ground_truth[cell_id] = isoform_counts
            
            if max_reads and len(all_reads) >= max_reads:
                logger.info(f"Reached maximum read count: {max_reads}")
                break
        
        # Shuffle reads (realistic FASTQ is not ordered by cell)
        np.random.shuffle(all_reads)
        
        # Write FASTQ file
        self.write_fastq(all_reads, output_file)
        
        logger.info(f"Generated {len(all_reads)} reads in {output_file}")
        
        return ground_truth
