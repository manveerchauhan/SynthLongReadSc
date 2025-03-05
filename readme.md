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

![SynthLongRead Pipeline Flowchart](./docs/images/flowchat.png)

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

## How Our Error Models Work

SynthLongRead uses a sequence context-based approach to model and reproduce sequencing errors. Here's a technical explanation of how the models operate:

### The 5-Base Window Approach

We model errors using a sliding window of 5 consecutive bases:

```
    ↓ Center base
  ACGTT
  12345
```

1. For each position in a read, we examine the 5-base context (the center base and 2 bases on each side)
2. This context is converted to a numerical representation (one-hot encoding):
   - A = [1,0,0,0]
   - C = [0,1,0,0]
   - G = [0,0,1,0]
   - T = [0,0,0,1]
3. For example, "ACGTT" becomes a 20-feature vector: [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1]

### Model Training Process

We train our models on millions of these examples from real data:

1. **Data Collection**: For each aligned base in real data, we record:
   - The 5-base context
   - What actually happened (match, substitution to which base, deletion)
   - The quality score assigned
   - The relative position in the read

2. **Neural Network Training**: Our models learn the relationship between:
   - Input: 5-base context + position
   - Output: Probabilities of different error types

### How Synthetic Reads Are Generated

The read generation process works like this:

1. **Start with Perfect Sequence**: 
   - Begin with the correct transcript sequence from the reference

2. **Process Base-by-Base**:
   - For each base position, extract the 5-base context
   - Feed this context into our error model
   - The model outputs probabilities for different outcomes:
     ```
     Match (correct read): 95.2%
     Substitution to A: 1.2%
     Substitution to C: 0.8%
     Substitution to G: 0.5%
     Substitution to T: 0.3%
     Deletion: 2.0%
     ```
   - Randomly sample an outcome based on these probabilities
   - Apply the selected error (or keep the base if "match" is selected)
   - Use our quality score model to assign a realistic quality score

3. **Handle Insertions Separately**:
   - After processing each base, determine if an insertion should occur
   - If yes, insert a random base with appropriate quality score

### Ground Truth Tracking

Every synthetic read maintains a connection to its source:

1. **Read IDs Include**:
   - Original transcript ID
   - Cell barcode
   - Position within transcript
   - Any introduced errors

2. **Ground Truth Matrix**:
   - Records exact transcript counts per cell
   - Serves as the benchmark for evaluating analysis tool performance

### Example of Model in Action

Here's a simplified example of how a read might be processed:

Original transcript fragment: `AACGTACGT`

Processing steps:
1. Consider context `AACGT` (center base C)
   - Model predicts 97% match, 2% substitution, 1% deletion
   - Random sample selects "match"
   - Keep C, assign quality score Q30

2. Consider context `ACGTA` (center base G)
   - Model predicts 80% match, 15% substitution, 5% deletion
   - Random sample selects "substitution to T"
   - Replace G with T, assign quality score Q18

3. Continue through the sequence...

Resulting read: `AACTACGT` (with one substitution)

This approach reproduces the specific error patterns of your sequencing technology, including homopolymer errors, sequence-specific biases, and position-dependent error rates.


## Citation

If you use SynthLongRead in your research, please cite:

```
[Citation information will be added after publication]
```
