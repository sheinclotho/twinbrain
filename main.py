#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TwinBrain - Digital Twin Brain System
Unified entry point for all workflows with configuration-driven behavior.

Usage:
    # Training with default config
    python main.py train --config config/default.yaml
    
    # Training with v3 legacy config
    python main.py train --config config/v3_legacy.yaml
    
    # Training with custom subject directory
    python main.py train --config config/default.yaml --base-dir /path/to/data
    
    # Export latent representations
    python main.py export --config config/default.yaml --subject test_file3/sub-01
"""

import argparse
import sys
from pathlib import Path

# Setup import path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import load_config
from utils.logging_utils import setup_logging, get_logger

# NOTE: Do NOT import torch-dependent modules here (e.g., workflows.training)
# They must be imported after random seed initialization in main()

logger = None  # Will be initialized after setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TwinBrain - Digital Twin Brain System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default v4 config
  python main.py train --config config/default.yaml
  
  # Train with legacy v3 config (for reproduction)
  python main.py train --config config/v3_legacy.yaml
  
  # Train with custom data directory
  python main.py train --config config/default.yaml --base-dir /path/to/subjects
  
  # Export latent representations
  python main.py export --config config/default.yaml --subject sub-01
        """
    )
    
    parser.add_argument(
        'workflow',
        choices=['train', 'export', 'infer'],
        help='Workflow to execute'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=None,
        help='Base directory containing subject data (default: test_file3/)'
    )
    
    parser.add_argument(
        '--subject',
        type=str,
        default=None,
        help='Specific subject to process (for export/infer workflows)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for results (overrides config)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and exit without running'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    global logger
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"✓ Loaded configuration from {args.config}")
        print(f"  Version: {config.get('version', 'unknown')}")
        print(f"  Description: {config.get('description', 'N/A')}")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Setup logging
    log_level = getattr(__import__('logging'), args.log_level)
    output_dir = args.output_dir or Path('logs')
    setup_logging(output_dir=output_dir, level=log_level)
    logger = get_logger(__name__)
    
    logger.info("=" * 80)
    logger.info("TwinBrain - Digital Twin Brain System")
    logger.info("=" * 80)
    logger.info(f"Workflow: {args.workflow}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Config version: {config.get('version', 'unknown')}")
    
    # Handle CUDA
    if args.no_cuda:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logger.info("CUDA disabled by command line flag")
    
    # Initialize random seeds for reproducibility and proper CUDA initialization
    # This must be done before any model creation or CUDA operations
    from utils.utils import set_random_seed
    seed = config.get('random_seed', 42)
    set_random_seed(seed)
    logger.info(f"Random seed initialized: {seed}")
    
    # Dry run - just print config and exit
    if args.dry_run:
        logger.info("Dry run mode - printing configuration")
        import yaml
        print("\n" + "=" * 80)
        print("Configuration:")
        print("=" * 80)
        print(yaml.dump(config.to_dict(), default_flow_style=False))
        print("=" * 80)
        logger.info("Dry run complete - exiting")
        return 0
    
    # Execute workflow
    try:
        if args.workflow == 'train':
            logger.info("Starting training workflow")
            # Import after seed initialization to prevent THPGenerator errors
            from workflows.training import run_training
            base_dir = args.base_dir or Path(__file__).parent / "test_file3"
            run_training(config, base_dir=base_dir)
            logger.info("Training workflow completed successfully")
            
        elif args.workflow == 'export':
            logger.info("Starting export workflow")
            # Import after seed initialization
            from workflows.export_latent import run_export
            subject = args.subject or "sub-01"
            run_export(config, subject=subject)
            logger.info("Export workflow completed")
            
        elif args.workflow == 'infer':
            logger.error("Inference workflow not yet implemented")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Workflow interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
