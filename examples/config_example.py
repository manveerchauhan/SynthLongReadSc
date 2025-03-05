"""
Example of using configuration files with SynthLongRead.
"""

import os
import argparse
import logging
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthlongread import SynthLongRead
from synthlongread.config import create_template_config

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ConfigExample')

def main():
    parser = argparse.ArgumentParser(description='SynthLongRead configuration example')
    
    parser.add_argument('--create-template', action='store_true',
                        help='Create a template configuration file')
    parser.add_argument('--config-file', default='synthlongread_config.yaml',
                        help='Path to configuration file (to create or use)')
    parser.add_argument('--run', action='store_true',
                        help='Run SynthLongRead with the configuration file')
    
    args = parser.parse_args()
    
    # Create template configuration if requested
    if args.create_template:
        logger.info(f"Creating template configuration file: {args.config_file}")
        create_template_config(args.config_file)
        
        logger.info("Template created. Please edit it with your specific parameters before running.")
        logger.info("You need to update at minimum:")
        logger.info("  - input.reference_transcriptome")
        logger.info("  - input.reference_gtf")
        logger.info("  - input.real_fastq")
        
        if not args.run:
            return
    
    # Run SynthLongRead with the configuration file
    if args.run:
        # Check if config file exists
        if not os.path.exists(args.config_file):
            logger.error(f"Configuration file not found: {args.config_file}")
            logger.error("Run with --create-template to create a template first")
            return
        
        logger.info(f"Running SynthLongRead with configuration: {args.config_file}")
        
        # Initialize SynthLongRead with config file
        synth = SynthLongRead(config_file=args.config_file)
        
        # Learn from real data
        synth.learn_from_real_data()
        
        # Generate synthetic dataset
        output_fastq, ground_truth = synth.generate_synthetic_dataset()
        
        logger.info(f"Generated synthetic data: {output_fastq}")
        logger.info(f"Ground truth: {ground_truth}")
        
        logger.info("Configuration example completed successfully")

if __name__ == "__main__":
    main()
