#!/usr/bin/env python3
"""
PLINDER Dynamics Pipeline - Simulation Runner

A molecular dynamics simulation runner that uses the refactored simulation engine
with improved error handling, logging, and checkpoint recovery.

Author: Maciej Wisniewski
Email: m.wisniewski@datascience.edu.pl
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path

from molecular_dynamics_pipeline.utils.logger import setup_logger, log_info, log_error, log_warning, log_debug
from molecular_dynamics_pipeline.simulation.simulation_refactored import MDSimulation


def validate_config(config_path: str) -> dict:
    """
    Load and validate configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
        
    Returns
    -------
    dict
        Loaded and validated configuration.
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML configuration: {e}")
    
    # Validate required sections
    required_sections = ['info', 'paths', 'preprocessing', 'forcefield', 'simulation_params']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate input files exist
    if config['paths'].get('raw_protein_files'):
        for protein_file in config['paths']['raw_protein_files']:
            if not Path(protein_file).exists():
                raise FileNotFoundError(f"Protein file not found: {protein_file}")
    
    if config['paths'].get('raw_ligand_files'):
        for ligand_file in config['paths']['raw_ligand_files']:
            if not Path(ligand_file).exists():
                raise FileNotFoundError(f"Ligand file not found: {ligand_file}")
    
    return config


def run_simulation(config: dict, logger: logging.Logger) -> bool:
    """
    Run the molecular dynamics simulation pipeline.
    
    Parameters
    ----------
    config : dict
        Simulation configuration.
    logger : logging.Logger
        Logger instance.
        
    Returns
    -------
    bool
        True if simulation completed successfully, False otherwise.
    """
    try:
        log_info(logger, f"Starting simulation for system: {config['info']['system_id']}")
        
        # Initialize simulation
        sim = MDSimulation(config)
        
        # Set up the system (preprocessing, solvation, force field assignment)
        log_info(logger, "Setting up simulation system...")
        sim.set_system()
        
        # Run simulation stages
        stages = ['warmup', 'backbone_removal', 'nvt', 'npt', 'production', 'energy_calculation']
        
        for stage in stages:
            if sim.config['info']['simulation_status'][stage] == 'Not Done':
                log_info(logger, f"Running {stage} stage...")
                
                # Get the stage method from the simulation object
                stage_method = getattr(sim, stage)
                stage_method()
                
                # Update status
                sim.update_simulation_status(stage, 'Done')
                log_info(logger, f"Completed {stage} stage successfully")
            else:
                log_info(logger, f"Skipping {stage} stage (already completed)")
        
        log_info(logger, "Simulation pipeline completed successfully")
        return True
        
    except Exception as e:
        log_error(logger, f"Simulation failed: {str(e)}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='PLINDER Dynamics Pipeline - Molecular Dynamics Simulation Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic simulation
  python run_simulation.py --config config/my_simulation.yaml
  
  # With custom output directory
  python run_simulation.py --config config.yaml --output-dir /path/to/output
  
  # Debug mode with verbose logging
  python run_simulation.py --config config.yaml --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory from configuration'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without running simulation'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logger(name="plinder_dynamics_runner", log_level=log_level)
    
    try:
        # Load and validate configuration
        log_info(logger, f"Loading configuration from: {args.config}")
        config = validate_config(args.config)
        
        # Override output directory if specified
        if args.output_dir:
            config['paths']['output_dir'] = str(Path(args.output_dir).resolve())
            log_info(logger, f"Output directory overridden: {config['paths']['output_dir']}")
        
        # Validation-only mode
        if args.validate_only:
            log_info(logger, "Configuration validation successful - exiting")
            return 0
        
        # Run simulation
        success = run_simulation(config, logger)
        
        if success:
            log_info(logger, "Simulation completed successfully")
            return 0
        else:
            log_error(logger, "Simulation failed")
            return 1
            
    except Exception as e:
        log_error(logger, f"Error: {str(e)}")
        if args.log_level.upper() == 'DEBUG':
            import traceback
            log_error(logger, traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 