"""
Benchmarking class for evaluating isoform detection and quantification.
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
    
    def __init__(self, ground_truth_file: str, results_file: str, output_dir: str):
        """
        Initialize IsoformBenchmark.
        
        Args:
            ground_truth_file: Path to the ground truth CSV file
            results_file: Path to the results file from the tool (e.g., FLAMES)
            output_dir: Directory to save evaluation results
        """
        self.ground_truth_file = ground_truth_file
        self.results_file = results_file
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.ground_truth = pd.read_csv(ground_truth_file)
        self.results = pd.read_csv(results_file)
        
        # Initialize metrics dictionary
        self.metrics = {}
    
    def preprocess_data(self):
        """
        Preprocess ground truth and results data for comparison.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Processed ground truth and results matrices
        """
        logger.info("Preprocessing data...")
        
        # This would depend on the specific format of FLAMES or other tools' output
        # Example preprocessing for a hypothetical format:
        
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
        gene_map = self.ground_truth[['transcript_id', 'gene_id']].drop_duplicates().set_index('transcript_id')['gene_id']
        
        self.ground_truth['gene_id'] = self.ground_truth['transcript_id'].map(gene_map)
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
        
        # Save detection metrics
        self.detection_df.to_csv(os.path.join(self.output_dir, 'detection_metrics.csv'), index=False)
        
        logger.info(f"Average F1 Score: {avg_metrics['avg_f1_score']:.4f}")
        logger.info(f"Average Sensitivity: {avg_metrics['avg_sensitivity']:.4f}")
        logger.info(f"Average Specificity: {avg_metrics['avg_specificity']:.4f}")
        
        return self.detection_df, avg_metrics
    
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
        
        # Save quantification metrics
        self.quantification_df.to_csv(os.path.join(self.output_dir, 'quantification_metrics.csv'), index=False)
        
        logger.info(f"Average Pearson correlation: {avg_quant_metrics['avg_pearson_r']:.4f}")
        logger.info(f"Average Spearman correlation: {avg_quant_metrics['avg_spearman_r']:.4f}")
        logger.info(f"Average RMSE: {avg_quant_metrics['avg_rmse']:.4f}")
        
        return self.quantification_df, avg_quant_metrics
    
    def evaluate_isoform_ratios(self):
        """
        Evaluate accuracy of relative isoform usage within genes.
        
        Returns:
            Tuple[pd.DataFrame, Dict]: DataFrame of ratio metrics per gene-cell pair and average metrics
        """
        logger.info("Evaluating isoform ratio accuracy...")
        
        # Get gene to transcript mapping
        transcript_to_gene = self.ground_truth[['transcript_id', 'gene_id']].drop_duplicates()
        transcript_to_gene = transcript_to_gene.set_index('transcript_id')['gene_id']
        
        # Calculate isoform ratios for genes with multiple isoforms
        gene_isoform_counts = {}
        for gene in set(transcript_to_gene.values):
            # Get transcripts for this gene
            gene_transcripts = [t for t, g in transcript_to_gene.items() if g == gene]
            
            if len(gene_transcripts) > 1:
                # Filter matrices to these transcripts
                gt_gene_transcripts = self.gt_matrix.loc[gene_transcripts]
                res_gene_transcripts = self.results_matrix.loc[gene_transcripts]
                
                gene_isoform_counts[gene] = {
                    'transcripts': gene_transcripts,
                    'gt_counts': gt_gene_transcripts,
                    'res_counts': res_gene_transcripts
                }
        
        # Compute Jensen-Shannon divergence for isoform distributions
        ratio_metrics = []
        
        for gene, data in gene_isoform_counts.items():
            transcripts = data['transcripts']
            gt_counts = data['gt_counts']
            res_counts = data['res_counts']
            
            for cell in gt_counts.columns:
                # Get counts for this gene's isoforms in this cell
                gt_iso_counts = gt_counts[cell].values
                res_iso_counts = res_counts[cell].values
                
                # Skip if no expression in ground truth
                if np.sum(gt_iso_counts) == 0:
                    continue
                
                # Calculate isoform proportions
                gt_props = gt_iso_counts / np.sum(gt_iso_counts)
                
                # Handle zero sums in results
                if np.sum(res_iso_counts) == 0:
                    res_props = np.zeros_like(res_iso_counts)
                else:
                    res_props = res_iso_counts / np.sum(res_iso_counts)
                
                # Calculate Jensen-Shannon divergence
                js_div = jensenshannon(gt_props, res_props, base=2)
                
                # Calculate correlation if more than 2 isoforms
                if len(transcripts) > 2 and len(set(gt_props)) > 1 and len(set(res_props)) > 1:
                    ratio_pearson, _ = pearsonr(gt_props, res_props)
                    ratio_spearman, _ = spearmanr(gt_props, res_props)
                else:
                    ratio_pearson = ratio_spearman = np.nan
                
                # Calculate absolute error in proportions
                prop_mae = np.mean(np.abs(gt_props - res_props))
                
                ratio_metrics.append({
                    'gene_id': gene,
                    'cell_id': cell,
                    'js_divergence': js_div,
                    'proportion_mae': prop_mae,
                    'ratio_pearson': ratio_pearson,
                    'ratio_spearman': ratio_spearman,
                    'num_isoforms': len(transcripts)
                })
        
        self.ratio_df = pd.DataFrame(ratio_metrics)
        
        # Calculate average metrics
        avg_ratio_metrics = {
            'avg_js_divergence': self.ratio_df['js_divergence'].mean(),
            'avg_proportion_mae': self.ratio_df['proportion_mae'].mean(),
            'avg_ratio_pearson': self.ratio_df['ratio_pearson'].mean(),
            'avg_ratio_spearman': self.ratio_df['ratio_spearman'].mean()
        }
        
        self.metrics.update(avg_ratio_metrics)
        
        # Save ratio metrics
        self.ratio_df.to_csv(os.path.join(self.output_dir, 'isoform_ratio_metrics.csv'), index=False)
        
        logger.info(f"Average JS divergence: {avg_ratio_metrics['avg_js_divergence']:.4f}")
        logger.info(f"Average proportion MAE: {avg_ratio_metrics['avg_proportion_mae']:.4f}")
        
        return self.ratio_df, avg_ratio_metrics
    
    def evaluate_gene_level(self):
        """
        Evaluate gene-level detection and quantification.
        
        Returns:
            Tuple[pd.DataFrame, Dict]: DataFrame of gene-level metrics and average metrics
        """
        logger.info("Evaluating gene-level metrics...")
        
        # 1. Gene detection metrics
        gt_gene_binary = (self.gt_gene_matrix > 0).astype(int)
        results_gene_binary = (self.results_gene_matrix > 0).astype(int)
        
        gene_detection_metrics = []
        
        for cell in gt_gene_binary.columns:
            y_true = gt_gene_binary[cell].values
            y_pred = results_gene_binary[cell].values
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            gene_detection_metrics.append({
                'cell_id': cell,
                'gene_precision': precision,
                'gene_recall': recall,
                'gene_sensitivity': sensitivity,
                'gene_specificity': specificity,
                'gene_f1_score': f1
            })
        
        self.gene_detection_df = pd.DataFrame(gene_detection_metrics)
        
        # 2. Gene quantification metrics
        gene_quant_metrics = []
        
        for cell in self.gt_gene_matrix.columns:
            expressed = self.gt_gene_matrix[cell] > 0
            if expressed.sum() < 5:
                continue
                
            gt_counts = self.gt_gene_matrix.loc[expressed, cell].values
            pred_counts = self.results_gene_matrix.loc[expressed, cell].values
            
            if len(set(gt_counts)) > 1 and len(set(pred_counts)) > 1:
                pearson_r, _ = pearsonr(gt_counts, pred_counts)
                spearman_r, _ = spearmanr(gt_counts, pred_counts)
            else:
                pearson_r = spearman_r = np.nan
            
            rmse = np.sqrt(np.mean((gt_counts - pred_counts) ** 2))
            
            gene_quant_metrics.append({
                'cell_id': cell,
                'gene_pearson_r': pearson_r,
                'gene_spearman_r': spearman_r,
                'gene_rmse': rmse
            })
        
        self.gene_quant_df = pd.DataFrame(gene_quant_metrics)
        
        # Merge gene metrics
        self.gene_metrics_df = pd.merge(
            self.gene_detection_df, 
            self.gene_quant_df, 
            on='cell_id', 
            how='outer'
        )
        
        # Calculate average metrics
        avg_gene_metrics = {
            'avg_gene_f1_score': self.gene_detection_df['gene_f1_score'].mean(),
            'avg_gene_sensitivity': self.gene_detection_df['gene_sensitivity'].mean(),
            'avg_gene_specificity': self.gene_detection_df['gene_specificity'].mean(),
            'avg_gene_pearson_r': self.gene_quant_df['gene_pearson_r'].mean(),
            'avg_gene_spearman_r': self.gene_quant_df['gene_spearman_r'].mean(),
            'avg_gene_rmse': self.gene_quant_df['gene_rmse'].mean()
        }
        
        self.metrics.update(avg_gene_metrics)
        
        # Save gene metrics
        self.gene_metrics_df.to_csv(os.path.join(self.output_dir, 'gene_level_metrics.csv'), index=False)
        
        logger.info(f"Average gene F1 Score: {avg_gene_metrics['avg_gene_f1_score']:.4f}")
        logger.info(f"Average gene Pearson correlation: {avg_gene_metrics['avg_gene_pearson_r']:.4f}")
        
        return self.gene_metrics_df, avg_gene_metrics
    
    def generate_plots(self):
        """Generate visualization plots for metrics"""
        logger.info("Generating visualization plots...")
        
        # 1. Scatter plot of ground truth vs predicted counts (transcript level)
        plt.figure(figsize=(12, 8))
        
        # Flatten matrices for overall scatter plot
        gt_flat = self.gt_matrix.values.flatten()
        pred_flat = self.results_matrix.values.flatten()
        
        # Filter to non-zero values in ground truth
        nonzero = gt_flat > 0
        gt_nonzero = gt_flat[nonzero]
        pred_nonzero = pred_flat[nonzero]
        
        plt.hexbin(np.log1p(gt_nonzero), np.log1p(pred_nonzero), 
                  gridsize=50, cmap='viridis', mincnt=1, bins='log')
        
        plt.xlabel('Log(Ground Truth Count + 1)')
        plt.ylabel('Log(Predicted Count + 1)')
        plt.title('Transcript Quantification: Ground Truth vs Predicted')
        
        # Calculate correlation for plot
        if len(set(gt_nonzero)) > 1 and len(set(pred_nonzero)) > 1:
            r, _ = pearsonr(gt_nonzero, pred_nonzero)
            plt.annotate(f'Pearson r = {r:.4f}', xy=(0.05, 0.95), xycoords='axes fraction')
        
        plt.colorbar(label='Log(Count)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'transcript_quant_scatter.png'), dpi=300)
        plt.close()
        
        # 2. Distribution of F1 scores across cells
        plt.figure(figsize=(10, 6))
        sns.histplot(self.detection_df['f1_score'], kde=True)
        plt.xlabel('F1 Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Transcript Detection F1 Scores Across Cells')
        plt.axvline(self.metrics['avg_f1_score'], color='red', linestyle='--', 
                   label=f'Mean: {self.metrics["avg_f1_score"]:.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'f1_score_distribution.png'), dpi=300)
        plt.close()
        
        # 3. Correlation distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.quantification_df['pearson_r'], kde=True)
        plt.xlabel('Pearson Correlation')
        plt.ylabel('Frequency')
        plt.title('Distribution of Transcript Quantification Correlation Across Cells')
        plt.axvline(self.metrics['avg_pearson_r'], color='red', linestyle='--', 
                   label=f'Mean: {self.metrics["avg_pearson_r"]:.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_distribution.png'), dpi=300)
        plt.close()
        
        # 4. Isoform ratio accuracy
        plt.figure(figsize=(10, 6))
        sns.histplot(self.ratio_df['js_divergence'], kde=True)
        plt.xlabel('Jensen-Shannon Divergence')
        plt.ylabel('Frequency')
        plt.title('Distribution of Isoform Ratio Accuracy (Jensen-Shannon Divergence)')
        plt.axvline(self.metrics['avg_js_divergence'], color='red', linestyle='--', 
                   label=f'Mean: {self.metrics["avg_js_divergence"]:.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'isoform_ratio_accuracy.png'), dpi=300)
        plt.close()
        
        # 5. Compare transcript-level vs gene-level metrics
        plt.figure(figsize=(12, 8))
        
        metrics_to_compare = [
            ('f1_score', 'gene_f1_score', 'F1 Score'),
            ('pearson_r', 'gene_pearson_r', 'Pearson Correlation')
        ]
        
        for i, (transcript_metric, gene_metric, title) in enumerate(metrics_to_compare):
            plt.subplot(1, 2, i+1)
            
            # Merge dataframes on cell_id
            merged = pd.merge(
                self.detection_df if 'f1' in transcript_metric else self.quantification_df,
                self.gene_metrics_df,
                on='cell_id'
            )
            
            plt.scatter(merged[transcript_metric], merged[gene_metric], alpha=0.6)
            plt.xlabel(f'Transcript-level {title}')
            plt.ylabel(f'Gene-level {title}')
            plt.title(f'Transcript vs Gene {title}')
            
            # Add diagonal line
            min_val = min(merged[transcript_metric].min(), merged[gene_metric].min())
            max_val = max(merged[transcript_metric].max(), merged[gene_metric].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            # Add correlation
            if merged[[transcript_metric, gene_metric]].dropna().shape[0] > 1:
                r, _ = pearsonr(
                    merged[transcript_metric].dropna(), 
                    merged[gene_metric].dropna()
                )
                plt.annotate(f'r = {r:.4f}', xy=(0.05, 0.95), xycoords='axes fraction')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'transcript_vs_gene_metrics.png'), dpi=300)
        plt.close()
        
        logger.info(f"Plots saved to {self.output_dir}")
    
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
        return self.metrics
