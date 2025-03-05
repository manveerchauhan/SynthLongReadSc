"""
Model sequence-dependent errors using deep learning.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Tuple, Optional, Union
from Bio import SeqIO

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ErrorModel')

class SequenceErrorModel(nn.Module):
    """Model sequence-dependent errors using deep learning"""
    
    def __init__(self, context_size: int = 5, hidden_size: int = 128):
        """
        Initialize the sequence error model.
        
        Args:
            context_size: Size of sequence context to consider
            hidden_size: Size of hidden layers
        """
        super(SequenceErrorModel, self).__init__()
        self.context_size = context_size
        
        # Input: one-hot encoded context (4 nucleotides * context_size)
        input_size = 4 * context_size
        
        # Simple network for sequence context
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 12)  # 12 outputs: 4 bases * 3 error types (match, sub, del)
        )
        
        # Positional bias (for read position effect)
        self.pos_bias = nn.Sequential(
            nn.Linear(1, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 3)  # 3 outputs: scaling factors for match, sub, del
        )
    
    def forward(self, context, position=None):
        """
        Forward pass of the model.
        
        Args:
            context: One-hot encoded sequence context [batch_size, 4 * context_size]
            position: Normalized position in read [batch_size, 1] (optional)
            
        Returns:
            torch.Tensor: Error probabilities [batch_size, 4, 3]
        """
        # Base error probabilities from sequence context
        error_probs = self.network(context)
        
        # Reshape to [batch_size, 4, 3] for 4 bases and 3 error types
        error_probs = error_probs.view(-1, 4, 3)
        
        # Apply position bias if provided
        if position is not None:
            pos_scale = self.pos_bias(position)
            pos_scale = torch.softmax(pos_scale, dim=1).unsqueeze(1)  # [batch_size, 1, 3]
            error_probs = error_probs * pos_scale
        
        # Ensure probabilities sum to 1 for each base
        error_probs = torch.softmax(error_probs, dim=2)
        
        return error_probs


class QualityScoreModel(nn.Module):
    """Model quality scores based on error probabilities"""
    
    def __init__(self, context_size: int = 5, hidden_size: int = 64):
        """
        Initialize the quality score model.
        
        Args:
            context_size: Size of sequence context to consider
            hidden_size: Size of hidden layers
        """
        super(QualityScoreModel, self).__init__()
        
        # Input: one-hot encoded context + error probs + position
        input_size = 4 * context_size + 3 + 1  # context + error types + position
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, context, error_probs, position):
        """
        Forward pass of the model.
        
        Args:
            context: One-hot encoded sequence context [batch_size, 4 * context_size]
            error_probs: Error probabilities [batch_size, 3]
            position: Normalized position in read [batch_size, 1]
            
        Returns:
            torch.Tensor: Quality scores [batch_size, 1]
        """
        # Concatenate inputs
        inputs = torch.cat([context, error_probs, position], dim=1)
        
        # Predict quality score
        quality = self.network(inputs)
        
        # Clamp to valid Phred range (0-41 is typical max for most platforms)
        quality = torch.clamp(quality, min=0, max=41)
        
        return quality


class ErrorModelTrainer:
    """Train machine learning models on error profiles"""
    
    def __init__(self, processor, context_size: int = 5, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the error model trainer.
        
        Args:
            processor: DataProcessor instance with error profiles
            context_size: Size of sequence context to consider
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.processor = processor
        self.context_size = context_size
        self.device = device
        
        self.seq_error_model = SequenceErrorModel(context_size=context_size).to(device)
        self.quality_model = QualityScoreModel(context_size=context_size).to(device)
        
        self.read_length_model = None
        self.trained = False
        
        logger.info(f"Using device: {device}")
    
    def prepare_training_data(self, reference_file: str, 
                              sample_size: int = 100000) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training data from reference sequences and error profiles.
        
        Args:
            reference_file: Path to reference sequence file
            sample_size: Number of samples to generate
            
        Returns:
            Tuple[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader]]:
                Train and validation data loaders for error and quality models
        """
        logger.info("Preparing training data...")
        
        # Load reference sequences
        references = []
        with open(reference_file, 'r') as f:
            for record in SeqIO.parse(f, "fasta"):
                references.append(str(record.seq).upper())
        
        # Sample subsequences from references
        contexts = []
        for _ in range(sample_size):
            # Pick a random reference
            ref_idx = np.random.randint(0, len(references))
            ref_seq = references[ref_idx]
            
            if len(ref_seq) < self.context_size + 1:
                continue
            
            # Pick a random position
            pos = np.random.randint(0, len(ref_seq) - self.context_size)
            
            # Extract context and center base
            context = ref_seq[pos:pos + self.context_size]
            center_idx = self.context_size // 2
            center_base = context[center_idx]
            
            if center_base not in "ACGT":
                continue
            
            # One-hot encode the context
            one_hot = []
            for base in context:
                if base == 'A':
                    one_hot.extend([1, 0, 0, 0])
                elif base == 'C':
                    one_hot.extend([0, 1, 0, 0])
                elif base == 'G':
                    one_hot.extend([0, 0, 1, 0])
                elif base == 'T':
                    one_hot.extend([0, 0, 0, 1])
                else:
                    one_hot.extend([0.25, 0.25, 0.25, 0.25])  # Unknown base
            
            contexts.append((one_hot, center_base))
        
        # Create labels based on error profiles
        X_error = []
        y_error = []
        X_quality = []
        y_quality = []
        
        sub_matrix = self.processor.error_profiles['substitutions']
        ins_rates = self.processor.error_profiles['insertions']
        del_rates = self.processor.error_profiles['deletions']
        pos_effect = self.processor.error_profiles['position_effect']
        
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        # Interpolate position effect for any position
        from scipy.interpolate import interp1d
        pos_points = np.linspace(0, 1, len(pos_effect))
        pos_effect_func = interp1d(pos_points, pos_effect, fill_value="extrapolate")
        
        quality_hist = {}
        for err_type in ['sub', 'ins']:
            if self.processor.error_profiles['quality_by_error'] and err_type in self.processor.error_profiles['quality_by_error']:
                qualities = self.processor.error_profiles['quality_by_error'][err_type]
                if qualities:
                    hist, bins = np.histogram(qualities, bins=40, range=(0, 40), density=True)
                    quality_hist[err_type] = (hist, bins)
        
        for context, center_base in contexts:
            base_idx = base_to_idx[center_base]
            
            # Sample position in read
            position = np.random.random()
            pos_error_rate = pos_effect_func(position)
            
            # Calculate error probabilities for this context
            match_prob = 1.0 - np.sum(sub_matrix[base_idx]) - del_rates[base_idx]
            match_prob *= (1.0 - pos_error_rate)  # Adjust for position
            
            sub_probs = sub_matrix[base_idx] * (1.0 + pos_error_rate)
            del_prob = del_rates[base_idx] * (1.0 + pos_error_rate)
            
            # Normalize to ensure probabilities sum to 1
            total = match_prob + np.sum(sub_probs) + del_prob
            match_prob /= total
            sub_probs /= total
            del_prob /= total
            
            # Error type labels
            error_label = np.zeros(3)  # [match, sub, del]
            
            # Sample error type
            r = np.random.random()
            if r < match_prob:
                error_type = 'match'
                error_label[0] = 1
            elif r < match_prob + np.sum(sub_probs):
                error_type = 'sub'
                error_label[1] = 1
            else:
                error_type = 'del'
                error_label[2] = 1
            
            # Add to error model dataset
            X_error.append(context + [position])
            y_error.append(error_label)
            
            # Sample quality score based on error type
            if error_type in quality_hist:
                hist, bins = quality_hist[error_type]
                bin_idx = np.random.choice(len(hist), p=hist/hist.sum())
                quality = (bins[bin_idx] + bins[bin_idx + 1]) / 2
            else:
                # Fallback if no quality data
                if error_type == 'match':
                    quality = max(1, min(40, np.random.normal(30, 5)))
                else:
                    quality = max(1, min(40, np.random.normal(20, 8)))
            
            # Add to quality model dataset
            X_quality.append(context + list(error_label) + [position])
            y_quality.append([quality])
        
        # Create PyTorch datasets
        class ErrorDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32)
                
            def __len__(self):
                return len(self.X)
                
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
        
        error_dataset = ErrorDataset(X_error, y_error)
        quality_dataset = ErrorDataset(X_quality, y_quality)
        
        # Split into train/val
        train_size = int(0.8 * len(error_dataset))
        val_size = len(error_dataset) - train_size
        
        train_error_dataset, val_error_dataset = torch.utils.data.random_split(
            error_dataset, [train_size, val_size]
        )
        
        train_quality_dataset, val_quality_dataset = torch.utils.data.random_split(
            quality_dataset, [train_size, val_size]
        )
        
        # Create dataloaders
        train_error_loader = DataLoader(
            train_error_dataset, batch_size=64, shuffle=True, num_workers=2
        )
        
        val_error_loader = DataLoader(
            val_error_dataset, batch_size=64, shuffle=False, num_workers=2
        )
        
        train_quality_loader = DataLoader(
            train_quality_dataset, batch_size=64, shuffle=True, num_workers=2
        )
        
        val_quality_loader = DataLoader(
            val_quality_dataset, batch_size=64, shuffle=False, num_workers=2
        )
        
        logger.info(f"Prepared {len(error_dataset)} training samples")
        
        # Train read length model
        self._train_read_length_model()
        
        return (train_error_loader, val_error_loader), (train_quality_loader, val_quality_loader)
    
    def _train_read_length_model(self):
        """Train a model for read length distribution"""
        if 'length_dist' not in self.processor.read_stats:
            logger.warning("No read length distribution found")
            return
        
        logger.info("Training read length model...")
        
        # Use GMM to model read length distribution
        lengths = self.processor.read_stats['length_dist'].reshape(-1, 1)
        
        # Determine optimal number of components
        max_components = min(5, len(lengths) // 1000)
        best_bic = np.inf
        best_n = 1
        
        for n in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=n, random_state=0)
            gmm.fit(lengths)
            bic = gmm.bic(lengths)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        
        logger.info(f"Using {best_n} components for read length model")
        
        # Train final model
        self.read_length_model = GaussianMixture(n_components=best_n, random_state=0)
        self.read_length_model.fit(lengths)
    
    def train_models(self, epochs: int = 10, learning_rate: float = 0.001):
        """
        Train error and quality models.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        logger.info("Training models...")
        
        # Prepare data
        (train_error_loader, val_error_loader), (train_quality_loader, val_quality_loader) = \
            self.prepare_training_data(self.processor.reference_file)
        
        # Train error model
        logger.info("Training sequence error model...")
        self._train_model(
            self.seq_error_model,
            train_error_loader,
            val_error_loader,
            nn.BCEWithLogitsLoss(),
            epochs,
            learning_rate
        )
        
        # Train quality model
        logger.info("Training quality score model...")
        self._train_model(
            self.quality_model,
            train_quality_loader,
            val_quality_loader,
            nn.MSELoss(),
            epochs,
            learning_rate
        )
        
        self.trained = True
        logger.info("Model training completed")
    
    def _train_model(self, model, train_loader, val_loader, criterion, epochs, lr):
        """
        Generic training function for PyTorch models.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            epochs: Number of training epochs
            lr: Learning rate
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.6f}")
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f"Epoch {epoch+1}/{epochs} Train Loss: {avg_train_loss:.6f} Val Loss: {avg_val_loss:.6f}")
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Could save model checkpoint here if desired
    
    def save_models(self, output_dir: str):
        """
        Save trained models.
        
        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        torch.save(self.seq_error_model.state_dict(), 
                  os.path.join(output_dir, "seq_error_model.pt"))
        
        torch.save(self.quality_model.state_dict(),
                  os.path.join(output_dir, "quality_model.pt"))
        
        with open(os.path.join(output_dir, "read_length_model.pkl"), 'wb') as f:
            pickle.dump(self.read_length_model, f)
        
        logger.info(f"Models saved to {output_dir}")
    
    def load_models(self, model_dir: str):
        """
        Load trained models.
        
        Args:
            model_dir: Directory with saved models
        """
        self.seq_error_model.load_state_dict(
            torch.load(os.path.join(model_dir, "seq_error_model.pt"), 
                      map_location=self.device)
        )
        
        self.quality_model.load_state_dict(
            torch.load(os.path.join(model_dir, "quality_model.pt"),
                      map_location=self.device)
        )
        
        with open(os.path.join(model_dir, "read_length_model.pkl"), 'rb') as f:
            self.read_length_model = pickle.load(f)
        
        self.trained = True
        logger.info(f"Models loaded from {model_dir}")
