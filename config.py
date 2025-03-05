"""
Configuration management for SynthLongRead.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Config')

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_file: Path to configuration YAML file
        
    Returns:
        Dict: Configuration dictionary
    """
    logger.info(f"Loading configuration from {config_file}")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate essential configuration elements
        _validate_config(config)
        
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def _validate_config(config: Dict[str, Any]):
    """
    Validate configuration for required fields.
    
    Args:
        config: Configuration dictionary
    
    Raises:
        ValueError: If required fields are missing
    """
    # Check required sections
    required_sections = ['input', 'output', 'platform']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Check required input fields
    required_inputs = ['reference_transcriptome', 'reference_gtf', 'real_fastq']
    for field in required_inputs:
        if field not in config['input'] or not config['input'][field]:
            raise ValueError(f"Missing required input field: {field}")
    
    # Validate platform type
    valid_platforms = ['ONT', 'PacBio']
    platform_type = config.get('platform', {}).get('type')
    if platform_type not in valid_platforms:
        raise ValueError(f"Invalid platform type: {platform_type}. Must be one of {valid_platforms}")
    
    logger.info("Configuration validation successful")

def save_config(config: Dict[str, Any], output_file: str):
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        output_file: Path to output YAML file
    """
    logger.info(f"Saving configuration to {output_file}")
    
    try:
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        raise

def create_template_config(output_file: str):
    """
    Create a template configuration file.
    
    Args:
        output_file: Path to output template file
    """
    template = {
        "input": {
            "reference_transcriptome": "/path/to/transcriptome.fa",
            "reference_gtf": "/path/to/annotation.gtf",
            "real_fastq": "/path/to/real_data.fastq",
            "reference_genome": "/path/to/genome.fa",
            "alignment_file": None
        },
        "output": {
            "directory": "./synthlongread_output",
            "overwrite": True
        },
        "platform": {
            "type": "ONT",
            "is_single_nucleus": False,
            "adapter_5p": "CCCATGTACTCTGCGTTGATACCACTGCTT",
            "adapter_3p": "AAAAAAAAAAAAAAAAAA"
        },
        "dataset": {
            "n_cells": 100,
            "sparsity": 0.8,
            "max_reads": 100000
        },
        "error_model": {
            "context_size": 5,
            "load_existing": False,
            "model_dir": None
        },
        "internal_priming": {
            "enabled": False,
            "rate": None,
            "min_a_content": 0.65,
            "window_size": 10,
            "infer_from_data": True
        },
        "benchmark": {
            "run_flames": False,
            "evaluate_internal_priming": False,
            "flames_path": "flames"
        },
        "performance": {
            "threads": 4,
            "device": "auto",
            "seed": 42
        }
    }
    
    save_config(template, output_file)
    logger.info(f"Template configuration created at {output_file}")
