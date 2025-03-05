"""
SynthLongRead: A framework for generating synthetic long-read scRNA-seq data
with realistic error profiles for benchmarking tools like FLAMES.
"""

from .synthlongread import SynthLongRead
from .data_processor import DataProcessor
from .error_model import ErrorModelTrainer, SequenceErrorModel, QualityScoreModel
from .isoform_synth import IsoformSynthesizer
from .cell_barcode import CellBarcodeSynthesizer
from .fastq_generator import FASTQGenerator
from .evaluation.benchmarking import benchmark_FLAMES

__version__ = "0.1.0"
__all__ = [
    "SynthLongRead",
    "DataProcessor",
    "ErrorModelTrainer",
    "SequenceErrorModel",
    "QualityScoreModel",
    "IsoformSynthesizer",
    "CellBarcodeSynthesizer",
    "FASTQGenerator",
    "benchmark_FLAMES"
]
