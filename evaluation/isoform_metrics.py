"""
Updated Isoform Benchmark class to handle internal priming evaluation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
import logging
from typing import Dict, List, Tuple, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('IsoformBenchmark')

class IsoformBenchmark:
    """Benchmarking class for evaluating isoform detection and quantification"""
    
    def __init__(self, ground_truth_file: str, results_file: str, output_dir: str,
                evaluate_internal_priming: bool = False):
        """
        Initialize IsoformBenchmark.
        
        Args:
            ground_truth_file: Path to the ground truth CSV file
            results_file: Path to the results file from the tool (e.g., FLAMES)
            output_dir: Directory to save evaluation results
            evaluate_internal_priming: Whether to evaluate internal priming effects
        """
        self.ground_truth_file = ground_truth_file
        self.results_file = results_file
        self.output_dir = output_dir
        self.evaluate_internal_priming = evaluate_internal_priming
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.ground_truth = pd.read_csv(ground_truth_file)
        self.results = pd.read_csv(results_file)
        
        # Check if ground truth has internal priming information
        self.has_internal_priming_data = 'has_internal_priming' in self.ground_truth.columns
        
        # Initialize metrics dictionary
        self.metrics = {}
    
    def preprocess_data(self):
        """
        Preprocess ground truth and results data for comparison.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Processed ground truth and results matrices
        """
        logger.info("Preprocessing data...")
        
        # 1. Convert ground truth to gene x cell matrix
        gt_pivot = self.ground_truth.pivot_table(
            index='transcript_id', 
            columns='cell_id', 
            values='count', 
            fill_value=0
        )
        
        # 2. Convert results to the same format
        # Assuming results has columns: transcript_id, cell_id, count
        results_pivot = self.results.pivot_table(
            index='transcript_id', 
            columns='cell_id', 
            values='count', 
            fill_value=0
        )
        
        # 3. Align matrices (keep only common transcripts and cells)
        common_transcripts = set(gt_pivot.index) & set(results_pivot.index)
        common_cells = set(gt_pivot.columns) & set(results_pivot.columns)
        
        logger.info(f"Common transcripts: {len(common_transcripts)}/{len(gt_pivot.index)}")
        logger.info(f"Common cells: {len(common_cells)}/{len(gt_pivot.columns)}")
        
        # Filter to common elements
        self.gt_matrix = gt_pivot.loc[list(common_transcripts), list(common_cells)]
        self.results_matrix = results_pivot.loc[list(common_transcripts), list(common_cells)]
        
        # Also compute gene-level matrices by summing transcript counts
        if 'gene_id' in self.ground_truth.columns:
            gene_map = self.ground_truth[['transcript_id', 'gene_id']].drop_duplicates().set_index('transcript_id')['gene_id']
            
            # Map transcript IDs to gene IDs
            self.ground_truth['gene_id'] = self.ground_truth['transcript_id'].map(gene_map)
            if 'gene_id' not in self.results.columns:
                self.results['gene_id'] = self.results['transcript_id'].map(gene_map)
            
            gt_gene_pivot = self.ground_truth.pivot_table(
                index='gene_id', 
                columns='cell_id', 
                values='count', 
                aggfunc='sum',
                fill_value=0
            )
            
            results_gene_pivot = self.results.pivot_table(
                index='gene_id', 
                columns='cell_id', 
                values='count', 
                aggfunc='sum',
                fill_value=0
            )
            
            common_genes = set(gt_gene_pivot.index) & set(results_gene_pivot.index)
            
            logger.info(f"Common genes: {len(common_genes)}/{len(gt_gene_pivot.index)}")
            
            self.gt_gene_matrix = gt_gene_pivot.loc[list(common_genes), list(common_cells)]
            self.results_gene_matrix = results_gene_pivot.loc[list(common_genes), list(common_cells)]
        
        # 4. If internal priming information is available, create matrices for internal priming analysis
        if self.has_internal_priming_data and self.evaluate_internal_priming:
            # Create a mapping of transcript IDs to internal priming status
            internal_priming_map = {}
            for _, row in self.ground_truth.iterrows():
                if 'has_internal_priming' in row:
                    transcript_id = row['transcript_id']
                    internal_priming_map[transcript_id] = row['has_internal_priming']
            
            # Tag transcripts for internal priming analysis
            self.internal_priming_transcripts = {t: p for t, p in internal_priming_map.items() if p and t in common_transcripts}
            self.non_internal_priming_transcripts = {t: p for t, p in internal_priming_map.items() if not p and t in common_transcripts}
            
            logger.info(f"Transcripts with internal priming: {len(self.internal_priming_transcripts)}")
            logger.info(f"Transcripts without internal priming: {len(self.non_internal_priming_transcripts)}")
            
            # Create separate matrices for internal priming vs. non-internal priming transcripts
            if self.internal_priming_transcripts:
                self.gt_internal_matrix = gt_pivot.loc[list(self.internal_priming_transcripts.keys()), list(common_cells)]
                self.results_internal_matrix = results_pivot.loc[list(self.internal_priming_transcripts.keys()), list(common_cells)]
            
            if self.non_internal_priming_transcripts:
                self.gt_non_internal_matrix = gt_pivot.loc[list(self.non_internal_priming_transcripts.keys()), list(common_cells)]
                self.results_non_internal_matrix = results_pivot.loc[list(self.non_internal_priming_transcripts.keys()), list(common_cells)]
        
        return self.gt_matrix, self.results_matrix
    
    def evaluate_detection(self, threshold: float = 1.0):
        """
        Evaluate isoform detection metrics (sensitivity, specificity, etc.).
        
        Args:
            threshold: Count threshold for binary classification
            
        Returns:
            Tuple[pd.DataFrame, Dict]: DataFrame of detection metrics per cell and average metrics
        """
        logger.info("Evaluating isoform detection...")
        
        # 1. Binarize matrices based on threshold
        gt_binary = (self.gt_matrix > threshold).astype(int)
        results_binary = (self.results_matrix > threshold).astype(int)
        
        # 2. Compute detection metrics per cell
        detection_metrics = []
        
        for cell in gt_binary.columns:
            y_true = gt_binary[cell].values
            y_pred = results_binary[cell].values
            
            # Compute precision, recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            
            # Compute ROC AUC if possible
            try:
                fpr, tpr, _ = roc_curve(y_true, self.results_matrix[cell].values)
                roc_auc = auc(fpr, tpr)
            except Exception:
                roc_auc = np.nan
            
            # Compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            # Calculate sensitivity and specificity
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            detection_metrics.append({
                'cell_id': cell,
                'precision': precision,
                'recall': recall,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            })
        
        self.detection_df = pd.DataFrame(detection_metrics)
        
        # Calculate average metrics
        avg_metrics = {
            'avg_precision': self.detection_df['precision'].mean(),
            'avg_recall': self.detection_df['recall'].mean(),
            'avg_sensitivity': self.detection_df['sensitivity'].mean(),
            'avg_specificity': self.detection_df['specificity'].mean(),
            'avg_f1_score': self.detection_df['f1_score'].mean(),
            'avg_roc_auc': self.detection_df['roc_auc'].mean()
        }
        
        self.metrics.update(avg_metrics)
        
        # 3. If internal priming data available, compute metrics separately
        if self.has_internal_priming_data and self.evaluate_internal_priming:
            internal_metrics = self._evaluate_detection_by_group(
                "internal_priming", 
                self.gt_internal_matrix if hasattr(self, 'gt_internal_matrix') else None,
                self.results_internal_matrix if hasattr(self, 'results_internal_matrix') else None,
                threshold
            )
            
            non_internal_metrics = self._evaluate_detection_by_group(
                "non_internal_priming",
                self.gt_non_internal_matrix if hasattr(self, 'gt_non_internal_matrix') else None,
                self.results_non_internal_matrix if hasattr(self, 'results_non_internal_matrix') else None,
                threshold
            )
            
            # Add internal priming specific metrics to overall metrics
            if internal_metrics and non_internal_metrics:
                self.metrics.update({
                    'internal_priming_f1_score': internal_metrics.get('avg_f1_score'),
                    'non_internal_priming_f1_score': non_internal_metrics.get('avg_f1_score'),
                    'internal_priming_sensitivity': internal_metrics.get('avg_sensitivity'),
                    'non_internal_priming_sensitivity': non_internal_metrics.get('avg_sensitivity')
                })
                
                # Log internal priming detection metrics
                logger.info(f"Internal priming detection F1 Score: {internal_metrics.get('avg_f1_score', 0):.4f}")
                logger.info(f"Non-internal priming detection F1 Score: {non_internal_metrics.get('avg_f1_score', 0):.4f}")
        
        # Save detection metrics
        self.detection_df.to_csv(os.path.join(self.output_dir, 'detection_metrics.csv'), index=False)
        
        logger.info(f"Average F1 Score: {avg_metrics['avg_f1_score']:.4f}")
        logger.info(f"Average Sensitivity: {avg_metrics['avg_sensitivity']:.4f}")
        logger.info(f"Average Specificity: {avg_metrics['avg_specificity']:.4f}")
        
        return self.detection_df, avg_metrics
    
    def _evaluate_detection_by_group(self, group_name, gt_matrix, results_matrix, threshold):
        """Helper function to evaluate detection metrics for a specific group of transcripts"""
        if gt_matrix is None or results_matrix is None or gt_matrix.empty or results_matrix.empty:
            logger.warning(f"No data available for {group_name} detection evaluation")
            return {}
        
        # Binarize matrices
        gt_binary = (gt_matrix > threshold).astype(int)
        results_binary = (results_matrix > threshold).astype(int)
        
        # Compute metrics per cell
        group_metrics = []
        
        for cell in gt_binary.columns:
            y_true = gt_binary[cell].values
            y_pred = results_binary[cell].values
            
            # Skip cells with no positive examples
            if sum(y_true) == 0:
                continue
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            group_metrics.append({
                'cell_id': cell,
                'precision': precision,
                'recall': recall,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'f1_score': f1
            })
        
        if not group_metrics:
            return {}
        
        # Create dataframe and save
        group_df = pd.DataFrame(group_metrics)
        group_df.to_csv(os.path.join(self.output_dir, f'{group_name}_detection_metrics.csv'), index=False)
        
        # Calculate averages
        avg_group_metrics = {
            'avg_precision': group_df['precision'].mean(),
            'avg_recall': group_df['recall'].mean(),
            'avg_sensitivity': group_df['sensitivity'].mean(),
            'avg_specificity': group_df['specificity'].mean(),
            'avg_f1_score': group_df['f1_score'].mean()
        }
        
        return avg_group_metrics
    
    def evaluate_quantification(self):
        """
        Evaluate isoform quantification accuracy (correlation, RMSE, etc.).
        
        Returns:
            Tuple[pd.DataFrame, Dict]: DataFrame of quantification metrics per cell and average metrics
        """
        logger.info("Evaluating isoform quantification...")
        
        # 1. Compute correlation metrics per cell
        quant_metrics = []
        
        for cell in self.gt_matrix.columns:
            # Filter for expressed transcripts in ground truth
            expressed = self.gt_matrix[cell] > 0
            if expressed.sum() < 5:  # Skip cells with too few expressed transcripts
                continue
                
            gt_counts = self.gt_matrix.loc[expressed, cell].values
            pred_counts = self.results_matrix.loc[expressed, cell].values
            
            # Compute correlations
            if len(set(gt_counts)) > 1 and len(set(pred_counts)) > 1:
                pearson_r, pearson_p = pearsonr(gt_counts, pred_counts)
                spearman_r, spearman_p = spearmanr(gt_counts, pred_counts)
            else:
                pearson_r = pearson_p = spearman_r = spearman_p = np.nan
            
            # Compute error metrics
            rmse = np.sqrt(np.mean((gt_counts - pred_counts) ** 2))
            mae = np.mean(np.abs(gt_counts - pred_counts))
            
            # Calculate normalized RMSE
            nrmse = rmse / np.mean(gt_counts) if np.mean(gt_counts) > 0 else np.nan
            
            quant_metrics.append({
                'cell_id': cell,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'rmse': rmse,
                'nrmse': nrmse,
                'mae': mae,
                'num_expressed': expressed.sum()
            })
        
        self.quantification_df = pd.DataFrame(quant_metrics)
        
        # Calculate average metrics
        avg_quant_metrics = {
            'avg_pearson_r': self.quantification_df['pearson_r'].mean(),
            'avg_spearman_r': self.quantification_df['spearman_r'].mean(),
            'avg_rmse': self.quantification_df['rmse'].mean(),
            'avg_nrmse': self.quantification_df['nrmse'].mean(),
            'avg_mae': self.quantification_df['mae'].mean()
        }
        
        self.metrics.update(avg_quant_metrics)
        
        # 2. If internal priming data available, compute metrics separately
        if self.has_internal_priming_data and self.evaluate_internal_priming:
            internal_metrics = self._evaluate_quantification_by_group(
                "internal_priming", 
                self.gt_internal_matrix if hasattr(self, 'gt_internal_matrix') else None,
                self.results_internal_matrix if hasattr(self, 'results_internal_matrix') else None
            )
            
            non_internal_metrics = self._evaluate_quantification_by_group(
                "non_internal_priming",
                self.gt_non_internal_matrix if hasattr(self, 'gt_non_internal_matrix') else None,
                self.results_non_internal_matrix if hasattr(self, 'results_non_internal_matrix') else None
            )
            
            # Add internal priming specific metrics to overall metrics
            if internal_metrics and non_internal_metrics:
                self.metrics.update({
                    'internal_priming_pearson_r': internal_metrics.get('avg_pearson_r'),
                    'non_internal_priming_pearson_r': non_internal_metrics.get('avg_pearson_r'),
                    'internal_priming_rmse': internal_metrics.get('avg_rmse'),
                    'non_internal_priming_rmse': non_internal_metrics.get('avg_rmse')
                })
                
                # Log internal priming quantification metrics
                logger.info(f"Internal priming quantification correlation: {internal_metrics.get('avg_pearson_r', 0):.4f}")
                logger.info(f"Non-internal priming quantification correlation: {non_internal_metrics.get('avg_pearson_r', 0):.4f}")
        
        # Save quantification metrics
        self.quantification_df.to_csv(os.path.join(self.output_dir, 'quantification_metrics.csv'), index=False)
        
        logger.info(f"Average Pearson correlation: {avg_quant_metrics['avg_pearson_r']:.4f}")
        logger.info(f"Average Spearman correlation: {avg_quant_metrics['avg_spearman_r']:.4f}")
        logger.info(f"Average RMSE: {avg_quant_metrics['avg_rmse']:.4f}")
        
        return self.quantification_df, avg_quant_metrics
    
    def _evaluate_quantification_by_group(self, group_name, gt_matrix, results_matrix):
        """Helper function to evaluate quantification metrics for a specific group of transcripts"""
        if gt_matrix is None or results_matrix is None or gt_matrix.empty or results_matrix.empty:
            logger.warning(f"No data available for {group_name} quantification evaluation")
            return {}
        
        # Compute metrics per cell
        group_metrics = []
        
        for cell in gt_matrix.columns:
            # Filter for expressed transcripts
            expressed = gt_matrix[cell] > 0
            if expressed.sum() < 5:  # Skip cells with too few expressed transcripts
                continue
                
            gt_counts = gt_matrix.loc[expressed, cell].values
            pred_counts = results_matrix.loc[expressed, cell].values
            
            # Compute correlations
            if len(set(gt_counts)) > 1 and len(set(pred_counts)) > 1:
                pearson_r, _ = pearsonr(gt_counts, pred_counts)
                spearman_r, _ = spearmanr(gt_counts, pred_counts)
            else:
                pearson_r = spearman_r = np.nan
            
            rmse = np.sqrt(np.mean((gt_counts - pred_counts) ** 2))
            mae = np.mean(np.abs(gt_counts - pred_counts))
            
            group_metrics.append({
                'cell_id': cell,
                'pearson_r': pearson_r,
                'spearman_r': spearman_r,
                'rmse': rmse,
                'mae': mae
            })
        
        if not group_metrics:
            return {}
        
        # Create dataframe and save
        group_df = pd.DataFrame(group_metrics)
        group_df.to_csv(os.path.join(self.output_dir, f'{group_name}_quantification_metrics.csv'), index=False)
        
        # Calculate averages
        avg_group_metrics = {
            'avg_pearson_r': group_df['pearson_r'].mean(),
            'avg_spearman_r': group_df['spearman_r'].mean(),
            'avg_rmse': group_df['rmse'].mean(),
            'avg_mae': group_df['mae'].mean()
        }
        
        return avg_group_metrics
    
    # The rest of the class remains largely unchanged
    # Methods like evaluate_isoform_ratios(), evaluate_gene_level(), and generate_plots()
    # would be modified in a similar way to add internal priming analysis
    
    def generate_plots(self):
        """Generate visualization plots for metrics"""
        logger.info("Generating visualization plots...")
        
        # [Existing plotting code...]
        
        # Add internal priming specific plots if data available
        if self.has_internal_priming_data and self.evaluate_internal_priming:
            self._generate_internal_priming_plots()
    
    def _generate_internal_priming_plots(self):
        """Generate plots specific to internal priming analysis"""
        if not hasattr(self, 'gt_internal_matrix') or not hasattr(self, 'results_internal_matrix'):
            return
            
        # 1. Detection comparison: internal priming vs non-internal priming
        if hasattr(self, 'metrics') and 'internal_priming_f1_score' in self.metrics and 'non_internal_priming_f1_score' in self.metrics:
            plt.figure(figsize=(10, 6))
            metrics = ['f1_score', 'sensitivity', 'specificity']
            
            internal_values = [
                self.metrics.get('internal_priming_f1_score', 0),
                self.metrics.get('internal_priming_sensitivity', 0),
                self.metrics.get('internal_priming_specificity', 0)
            ]
            
            non_internal_values = [
                self.metrics.get('non_internal_priming_f1_score', 0),
                self.metrics.get('non_internal_priming_sensitivity', 0),
                self.metrics.get('non_internal_priming_specificity', 0)
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, internal_values, width, label='With Internal Priming')
            plt.bar(x + width/2, non_internal_values, width, label='Without Internal Priming')
            
            plt.xlabel('Metric')
            plt.ylabel('Score')
            plt.title('Detection Performance: Internal Priming vs. Non-Internal Priming')
            plt.xticks(x, metrics)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'internal_priming_detection_comparison.png'), dpi=300)
            plt.close()
            
        # 2. Quantification comparison: internal priming vs non-internal priming
        if hasattr(self, 'metrics') and 'internal_priming_pearson_r' in self.metrics and 'non_internal_priming_pearson_r' in self.metrics:
            plt.figure(figsize=(10, 6))
            metrics = ['pearson_r', 'spearman_r', 'rmse']
            
            internal_values = [
                self.metrics.get('internal_priming_pearson_r', 0),
                self.metrics.get('internal_priming_spearman_r', 0),
                self.metrics.get('internal_priming_rmse', 0)
            ]
            
            non_internal_values = [
                self.metrics.get('non_internal_priming_pearson_r', 0),
                self.metrics.get('non_internal_priming_spearman_r', 0),
                self.metrics.get('non_internal_priming_rmse', 0)
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, internal_values, width, label='With Internal Priming')
            plt.bar(x + width/2, non_internal_values, width, label='Without Internal Priming')
            
            plt.xlabel('Metric')
            plt.ylabel('Score')
            plt.title('Quantification Performance: Internal Priming vs. Non-Internal Priming')
            plt.xticks(x, metrics)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'internal_priming_quantification_comparison.png'), dpi=300)
            plt.close()
    
    def run_all_evaluations(self):
        """
        Run all evaluation metrics.
        
        Returns:
            Dict: Dictionary of all metrics
        """
        # Preprocess data
        self.preprocess_data()
        
        # Run evaluations
        self.evaluate_detection()
        self.evaluate_quantification()
        self.evaluate_isoform_ratios()
        self.evaluate_gene_level()
        
        # Generate plots
        self.generate_plots()
        
        # Save summary metrics
        pd.DataFrame([self.metrics]).to_csv(
            os.path.join(self.output_dir, 'summary_metrics.csv'), index=False
        )
        
        logger.info("All evaluations completed")
        
        # Print internal priming metrics summary if available
        if self.has_internal_priming_data and self.evaluate_internal_priming:
            logger.info("\nInternal Priming Analysis Summary:")
            ip_metrics = {k: v for k, v in self.metrics.items() if 'internal_priming' in k}
            for metric, value in ip_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        return self.metrics