#!/usr/bin/env python3

import os
import yaml
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
from molecular_dynamics_pipeline.utils.logger import setup_logger, log_info, log_error, log_warning, log_debug

logger = setup_logger(name="validate_status", log_level=logging.INFO)

def _check_for_errors(sim_dir: Path, system_id: str, configured_stages: List[str], completed_stages: List[str]) -> bool:
    """
    Check for indicators of simulation errors vs normal incomplete status.
    
    Returns:
        bool: True if errors detected, False if just incomplete
    """
    # Check for log files with error messages
    logs_dir = sim_dir / 'logs'
    if logs_dir.exists():
        for log_file in logs_dir.glob('*.log'):
            try:
                with open(log_file, 'r') as f:
                    content = f.read().lower()
                    # Look for common error indicators
                    error_keywords = ['error']
                    if any(keyword in content for keyword in error_keywords):
                        log_debug(logger, f"Error keywords found in log file: {log_file}")
                        return True
            except Exception:
                pass  # Skip if can't read log file

    return False

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
        
        # Get system_id from config
        if 'info' not in config or 'system_id' not in config['info']:
            return "Error"
        
        system_id = config['info']['system_id']
        stages = ['warmup', 'backbone_removal', 'nvt', 'npt', 'production', 'energy_calculation']
        
        # Check which stages are configured to run
        configured_stages = []
        if 'simulation_params' in config:
            for stage in stages:
                if stage in config['simulation_params'] and config['simulation_params'][stage].get('run', False):
                    configured_stages.append(stage)
        
        # If no stages are configured to run, check all stages
        if not configured_stages:
            configured_stages = stages
        
        # Build expected topology paths manually (like _setup_paths does)
        # Pattern: {output_dir}/{simulation_id}/topologies/{system_id}_{stage}_topology.cif
        topologies_dir = sim_dir / 'topologies'
        
        completed_stages = []
        for stage in configured_stages:
            # Special handling for energy_calculation stage
            if stage == 'energy_calculation':
                # Check for energy output files instead of topology file
                energy_output_dir = sim_dir / 'energies'
                energy_matrix_file = energy_output_dir / 'interaction_energy_matrix.npz'
                energy_json_file = energy_output_dir / 'component_energies.json'
                
                if energy_matrix_file.exists() and energy_json_file.exists():
                    completed_stages.append(stage)
                    log_debug(logger, f"Found energy output files for {plinder_id} stage {stage}")
                else:
                    log_debug(logger, f"Missing energy output files for {plinder_id} stage {stage}")
            else:
                # Construct expected topology file path
                topology_filename = f"{system_id}_{stage}_topology.cif"
                topology_path = topologies_dir / topology_filename
                
                if topology_path.exists():
                    completed_stages.append(stage)
                    log_debug(logger, f"Found topology file for {plinder_id} stage {stage}: {topology_path}")
                else:
                    log_debug(logger, f"Missing topology file for {plinder_id} stage {stage}: {topology_path}")
        
        # Determine overall status
        if len(completed_stages) == len(configured_stages):
            return "Complete"
        elif len(completed_stages) > 0:
            # Check if this is an error or just incomplete
            error_detected = _check_for_errors(sim_dir, system_id, configured_stages, completed_stages)
            return "Error" if error_detected else "Incomplete"
        else:
            # No stages completed - check if simulation was attempted
            error_detected = _check_for_errors(sim_dir, system_id, configured_stages, completed_stages)
            return "Error" if error_detected else "Not Started"
            
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
    
    for plinder_id in tqdm(plinder_ids, desc="Validating simulations", total=len(plinder_ids)):
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
    parser.add_argument('--plinder_ids_file', type=str, required=False, default='scripts/plinder_scripts/plinder_md_ids.txt', 
                       help='File containing PLINDER IDs to validate (one per line)')
    parser.add_argument('--output_dir', type=str, required=False, default='/mnt/raid/mwisniewski/Data/plinder_md/eden',
                       help='Directory containing simulation outputs')
    parser.add_argument('--config_template', type=str, required=False, default='config/plinder_parameters_bound.yaml',
                       help='Path to config template file')
    parser.add_argument('--output_file', type=str, required=False, default='eden_check.csv',
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
