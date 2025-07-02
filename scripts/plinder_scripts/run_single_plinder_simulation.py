#!/usr/bin/env python3

import os
import yaml
import argparse
import dotenv
dotenv.load_dotenv('plinder.env')

from typing import Optional
import random
import multiprocessing as mp
from functools import partial

import logging
from dynamics_pipeline.utils.logger import setup_logger, log_info, log_error, log_warning, log_debug
from dynamics_pipeline.simulation.simulation_refactored import MDSimulation
from dynamics_pipeline.data.plinder import load_plinder_ids, create_system_config

random.seed(42)


# Setup global logger
logger = setup_logger(name="plinder_dynamics", log_level=logging.INFO)

def check_if_simulation_exists(plinder_id, output_dir, config_template):
    """Check if a simulation exists for a given PLINDER ID.
    Args:
        plinder_id (str): The PLINDER ID to check
        output_dir (str): The output directory
        config_template (str): The path to the configuration template
    """
    with open(config_template, 'r') as f:
        config = yaml.safe_load(f)
 
    if config['info']['bound_state'] == True:
        config['info']['simulation_id'] = f"{plinder_id}_simulation_bound_state"
    else:
        config['info']['simulation_id'] = f"{plinder_id}_simulation_unbound_state"

    system_config_filepath = os.path.join(output_dir, config['info']['simulation_id'], 'config.yaml')

    output_dir = os.path.join(output_dir, config['info']['simulation_id'])

    if os.path.exists(output_dir):
        return True
    else:
        return False


def run_simulation(config, sim_logger):
    # Initialize and run simulation
    sim = MDSimulation(config)
    
    # Preprocessing Step
    log_info(sim_logger, f"Starting system setup for {config['info']['simulation_id']}")
    sim.set_system()
    
    # Heat up
    if sim.config['info']['simulation_status']['warmup'] == 'Not Done':
        log_info(sim_logger, "Starting warmup phase")
        sim.warmup()
        sim.update_simulation_status('warmup', 'Done')
        log_info(sim_logger, "Completed warmup phase")
        
    # Removal of Backbone Constraints
    if sim.config['info']['simulation_status']['backbone_removal'] == 'Not Done':
        log_info(sim_logger, "Starting backbone constraints removal")
        sim.remove_backbone_constraints()
        sim.update_simulation_status('backbone_removal', 'Done')
        log_info(sim_logger, "Completed backbone constraints removal")

    # NVT
    if sim.config['info']['simulation_status']['nvt'] == 'Not Done':
        log_info(sim_logger, "Starting NVT equilibration")
        sim.nvt()
        sim.update_simulation_status('nvt', 'Done')
        log_info(sim_logger, "Completed NVT equilibration")

    # NPT
    if sim.config['info']['simulation_status']['npt'] == 'Not Done':
        log_info(sim_logger, "Starting NPT equilibration")
        sim.npt()
        sim.update_simulation_status('npt', 'Done')
        log_info(sim_logger, "Completed NPT equilibration")

    # Production
    if sim.config['info']['simulation_status']['production'] == 'Not Done':
        log_info(sim_logger, "Starting production run")
        sim.production()
        sim.update_simulation_status('production', 'Done')
        log_info(sim_logger, "Completed production run")

    #if sim.config['info'].get('calculate_energy', False) == True:
        

def process_single_system(plinder_id, config_template, output_dir):
    try:
        # Create system-specific config
        system_config_filepath = create_system_config(template_config=config_template, system_id=plinder_id, output_dir=output_dir)
        with open(system_config_filepath, 'r') as f:
            config = yaml.safe_load(f)
            
        # Create system-specific output directory
        system_output_dir = os.path.join(output_dir, config['info']['simulation_id'])
        os.makedirs(system_output_dir, exist_ok=True)
        
        # Setup system-specific logger
        log_dir = os.path.join(system_output_dir, 'logs')
        sim_logger = setup_logger(
            name=f"plinder_dynamics_{plinder_id}",
            log_level=logging.INFO,
            log_dir=log_dir,
            console_output=True
        )
        
        log_info(sim_logger, f"Starting simulation for system {plinder_id}")
        log_info(sim_logger, f"Configuration loaded from {system_config_filepath}")
        
        run_simulation(config, sim_logger)
        
        log_info(sim_logger, f"Completed all simulation stages for system {plinder_id}")
        
    except Exception as e:
        if 'sim_logger' in locals():
            log_error(sim_logger, f"Error in simulation {plinder_id}: {str(e)}")
        else:
            log_error(logger, f"Error setting up simulation {plinder_id}: {str(e)}")
        return

    
def main():
    parser = argparse.ArgumentParser(description='Run Molecular Dynamics Simulation with OpenMM')
    parser.add_argument('--config_template', type=str, required=False, help='Path to config template file', default='config/plinder_parameters_bound.yaml')
    parser.add_argument('--filters', type=str, required=False, help='Filters to apply to the PLINDER systems', default='scripts/plinder_scripts/filters/train_plinder.yaml')
    parser.add_argument('--plinder_id', type=str, required=False, help='Single PLINDER ID to run (overrides filters)')
    parser.add_argument('--plinder_ids_file', type=str, required=False, help='File containing PLINDER IDs to run')
    parser.add_argument('--output_dir', type=str, required=False, help='Output directory', default=os.getenv('OUTPUT_DIR'))
    parser.add_argument('--overwrite', type=bool, required=False, help='Overwrite existing simulation', default=False)
    parser.add_argument('--parallel', type=int, required=False, help='Number of parallel processes to use', default=1)

    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    # Load Config
    assert os.path.exists(args.config_template), f"Config file {args.config_template} does not exist!"
    assert args.plinder_id is None or args.plinder_ids_file is None, "Only one of --plinder_id or --plinder_ids_file can be provided"
    assert args.plinder_ids_file is None or os.path.exists(args.plinder_ids_file), f"File {args.plinder_ids_file} does not exist!"
    if args.plinder_ids_file is not None and args.plinder_id is not None:
        raise ValueError("Only one of --plinder_id or --plinder_ids_file can be provided")
    
    # Determine which PLINDER IDs to process
    if args.plinder_id:
        # Single PLINDER ID provided
        plinder_ids = [args.plinder_id]
        log_info(logger, f"Running single simulation for PLINDER ID: {args.plinder_id}")
    elif args.plinder_ids_file:
        with open(args.plinder_ids_file, 'r') as f:
            plinder_ids = [line.strip() for line in f.readlines()]
        log_info(logger, f"Loaded {len(plinder_ids)} PLINDER IDs from file {args.plinder_ids_file}")
    else:
        # Load from filters
        plinder_ids = load_plinder_ids(args.filters)
        log_info(logger, f"Loaded {len(plinder_ids)} PLINDER IDs from filters")
    
    # Run Simulations
    if args.parallel > 1 and len(plinder_ids) > 1:
        process_func = partial(process_single_system, 
                               config_template=args.config_template,
                               output_dir=args.output_dir)
        with mp.Pool(processes=args.parallel) as pool:
            pool.map(process_func, plinder_ids)
    else:
        log_info(logger, f"Running {len(plinder_ids)} simulations sequentially")
        # Sequential processing
        for plinder_id in plinder_ids: 
            process_single_system(plinder_id, args.config_template, args.output_dir)

    log_info(logger, "All simulations completed")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main() 