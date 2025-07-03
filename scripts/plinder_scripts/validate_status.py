#!/usr/bin/env python3

import os
import yaml
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import logging

from molecular_dynamics_pipeline.utils.logger import setup_logger, log_info, log_error, log_warning, log_debug

logger = setup_logger(name="validate_status", log_level=logging.INFO)

def get_simulation_id(plinder_id: str, config_template: str) -> str:
    """Get the simulation ID from a PLINDER ID and config template."""
    try:
        with open(config_template, 'r') as f:
            config = yaml.safe_load(f)
        
        if config['info']['bound_state'] == True:
            return f"{plinder_id}_simulation_bound_state"
        else:
            return f"{plinder_id}_simulation_unbound_state"
    except Exception as e:
        log_error(logger, f"Error reading config template {config_template}: {str(e)}")
        return f"{plinder_id}_simulation_bound_state"  # Default fallback

def check_simulation_status(plinder_id: str, output_dir: str, config_template: str) -> str:
    """
    Check the status of a simulation for a given PLINDER ID.
    
    Returns:
        str: Status of the simulation - 'Complete', 'Incomplete', 'Not Started', or 'Error'
    """
    simulation_id = get_simulation_id(plinder_id, config_template)
    sim_dir = Path(output_dir) / simulation_id
    
    # Check if simulation directory exists
    if not sim_dir.exists():
        return "Not Started"
    
    # Check if config file exists
    config_path = sim_dir / 'config.yaml'
    if not config_path.exists():
        return "Error"
    
    try:
        # Load config file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if required sections exist
        if 'paths' not in config or 'topologies' not in config['paths']:
            return "Error"
        
        stages = ['warmup', 'backbone_removal', 'nvt', 'npt', 'production']
        
        # Check which stages are configured to run
        configured_stages = []
        if 'simulation_params' in config:
            for stage in stages:
                if stage in config['simulation_params'] and config['simulation_params'][stage].get('run', False):
                    configured_stages.append(stage)
        
        # If no stages are configured to run, check all stages
        if not configured_stages:
            configured_stages = stages
        
        # Check status by examining topology files (just like _set_simulation_info does)
        completed_stages = []
        for stage in configured_stages:
            topology_key = f'{stage}_topology_filepath'
            if topology_key in config['paths']['topologies']:
                topology_path = Path(config['paths']['topologies'][topology_key])
                if topology_path.exists():
                    completed_stages.append(stage)
                    log_debug(logger, f"Found topology file for {plinder_id} stage {stage}: {topology_path}")
                else:
                    log_debug(logger, f"Missing topology file for {plinder_id} stage {stage}: {topology_path}")
            else:
                log_warning(logger, f"Topology key {topology_key} not found in config for {plinder_id}")
        
        # Determine overall status
        if len(completed_stages) == len(configured_stages):
            return "Complete"
        elif len(completed_stages) > 0:
            return "Incomplete"
        else:
            return "Not Started"
            
    except Exception as e:
        log_error(logger, f"Error checking status for {plinder_id}: {str(e)}")
        return "Error"

def validate_simulations(plinder_ids: List[str], output_dir: str, config_template: str) -> pd.DataFrame:
    """
    Validate the status of multiple simulations.
    
    Args:
        plinder_ids: List of PLINDER IDs to check
        output_dir: Directory containing simulation outputs
        config_template: Path to config template file
        
    Returns:
        pd.DataFrame: DataFrame with columns 'id' and 'status'
    """
    results = []
    
    log_info(logger, f"Validating {len(plinder_ids)} simulations...")
    
    for plinder_id in plinder_ids:
        status = check_simulation_status(plinder_id, output_dir, config_template)
        results.append({
            'id': plinder_id,
            'status': status
        })
        
        if status == "Error":
            log_warning(logger, f"Error detected for simulation {plinder_id}")
        elif status == "Complete":
            log_debug(logger, f"Simulation {plinder_id} is complete")
    
    df = pd.DataFrame(results)
    
    # Print summary
    status_counts = df['status'].value_counts()
    log_info(logger, f"Validation complete. Status summary:")
    for status, count in status_counts.items():
        log_info(logger, f"  {status}: {count}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Validate status of molecular dynamics simulations')
    parser.add_argument('--plinder_ids_file', type=str, required=True, 
                       help='File containing PLINDER IDs to validate (one per line)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing simulation outputs')
    parser.add_argument('--config_template', type=str, required=False,
                       help='Path to config template file', 
                       default='config/plinder_parameters_bound.yaml')
    parser.add_argument('--output_file', type=str, required=False,
                       help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.plinder_ids_file):
        log_error(logger, f"PLINDER IDs file not found: {args.plinder_ids_file}")
        return
    
    if not os.path.exists(args.output_dir):
        log_error(logger, f"Output directory not found: {args.output_dir}")
        return
    
    if not os.path.exists(args.config_template):
        log_error(logger, f"Config template not found: {args.config_template}")
        return
    
    # Load PLINDER IDs
    try:
        with open(args.plinder_ids_file, 'r') as f:
            plinder_ids = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        log_error(logger, f"Error reading PLINDER IDs file: {str(e)}")
        return
    
    log_info(logger, f"Loaded {len(plinder_ids)} PLINDER IDs from {args.plinder_ids_file}")
    
    # Validate simulations
    df = validate_simulations(plinder_ids, args.output_dir, args.config_template)
    
    # Save results if output file specified
    if args.output_file:
        df.to_csv(args.output_file, index=False)
        log_info(logger, f"Results saved to {args.output_file}")
    
    # Print results
    print("\nValidation Results:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
