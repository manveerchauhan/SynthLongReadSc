# SynthLongRead

A Python framework for generating synthetic long-read scRNA-seq data with realistic error profiles for benchmarking tools like FLAMES.

## Overview

SynthLongRead is a machine learning-based framework that learns error patterns from real long-read sequencing data (ONT or PacBio) and generates synthetic FASTQ files with realistic error profiles. This framework is particularly useful for benchmarking long-read scRNA-seq analysis pipelines by providing a ground truth dataset for evaluating isoform detection and quantification accuracy.

### Key Features

- Learns realistic error profiles from real FASTQ data
- Models sequence-dependent error patterns using deep learning
- Generates cell-specific isoform expression patterns
- Produces realistic read length distributions
- Supports both ONT and PacBio error profiles
- Includes comprehensive evaluation metrics for benchmarking

## Pipeline Overview

![SynthLongRead Pipeline Flowchart](docs/images/flowchart.png)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SynthLongRead.git
cd SynthLongRead

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
from synthlongread import SynthLongRead

# Initialize the framework
synth = SynthLongRead(
    reference_transcriptome="reference.fa",
    reference_gtf="annotation.gtf",
    platform="ONT",
    output_dir="./output"
)

# Learn error profiles from real data
synth.learn_from_real_data(
    real_fastq="real_data.fastq"
)

# Generate synthetic dataset
synthetic_fastq, ground_truth = synth.generate_synthetic_dataset(
    n_cells=100,
    sparsity=0.8
)

print(f"Generated synthetic data at: {synthetic_fastq}")
print(f"Ground truth saved to: {ground_truth}")
```

## Workflow

### 1. Data Preparation
- Provide a reference transcriptome (FASTA) and annotation (GTF)
- Supply real long-read scRNA-seq FASTQ data to learn error profiles

### 2. Error Profile Learning
- The framework aligns reads to the reference to identify errors
- Extracts substitution, insertion, and deletion patterns
- Learns sequence context-dependent error profiles
- Models quality score distributions

### 3. Synthetic Data Generation
- Creates realistic isoform expression patterns across cells
- Generates reads with learned error profiles
- Adds cell barcodes with realistic error rates
- Produces complete FASTQ files with quality scores

### 4. Benchmarking
- Run analysis tools (e.g., FLAMES) on the synthetic data
- Compare results to the known ground truth
- Evaluate isoform detection and quantification accuracy

## Evaluation Metrics

SynthLongRead includes comprehensive metrics for evaluating synthetic data quality:

- **Isoform Detection**: Precision, recall, F1-score
- **Isoform Quantification**: Correlation, RMSE, MAE
- **Isoform Ratio Accuracy**: Jensen-Shannon divergence
- **Read-level Characteristics**: Length, quality, and base composition comparison

## Example Usage

See the `examples/` directory for detailed examples:
- `basic_workflow.py`: Simple end-to-end example
- `custom_expression.py`: Creating custom expression patterns
- `benchmark_flames.py`: Benchmarking the FLAMES tool

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- BioPython
- Pysam
- Scikit-learn
- Matplotlib
- Seaborn

## Citation

If you use SynthLongRead in your research, please cite:

```
[Citation information will be added after publication]
```
