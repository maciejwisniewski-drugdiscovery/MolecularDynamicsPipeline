#!/usr/bin/env python3

import os
import yaml
import argparse
from typing import Optional
import random
import multiprocessing as mp
from functools import partial

import logging
from dynamics_pipeline.utils.logger import setup_logger, log_info, log_error, log_warning, log_debug
from dynamics_pipeline.simulation.simulation import MDSimulation
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


def run_simulation(config):
    # Initialize and run simulation
    sim = MDSimulation(config)
    
    # Preprocessing Step
    sim.set_system()
    
    # Heat up
    if sim.config['info']['simulation_status']['warmup'] == 'Not Done':
        sim.warmup()
        sim.update_simulation_status('warmup', 'Done')
        
    # Removal of Backbone Constraints
    if sim.config['info']['simulation_status']['backbone_removal'] == 'Not Done':
        sim.remove_backbone_constraints()
        sim.update_simulation_status('backbone_removal', 'Done')

    # NVT
    if sim.config['info']['simulation_status']['nvt'] == 'Not Done':
        sim.nvt()
        sim.update_simulation_status('nvt', 'Done')

    # NPT
    if sim.config['info']['simulation_status']['npt'] == 'Not Done':
        sim.npt()
        sim.update_simulation_status('npt', 'Done')

    # Production
    if sim.config['info']['simulation_status']['production'] == 'Not Done':
        sim.production()
        sim.update_simulation_status('production', 'Done')


def process_single_system(plinder_id, config_template, output_dir):
    # if check_if_simulation_exists(plinder_id, output_dir, config_template):
    #     return
    try:
        system_config_filepath = create_system_config(template_config=config_template, system_id=plinder_id, output_dir=output_dir)
        with open(system_config_filepath, 'r') as f:
            config = yaml.safe_load(f)
        run_simulation(config)
    except Exception as e:
        log_error(logger, f"Error: {e}")
        return

    
def main():
    parser = argparse.ArgumentParser(description='Run Molecular Dynamics Simulation with OpenMM')
    parser.add_argument('--config_template', type=str, required=False, help='Path to config template file', default='config/plinder_parameters.yaml')
    parser.add_argument('--filters', type=str, required=False, help='Filters to apply to the PLINDER systems', default='scripts/filters/train_plinder.yaml')
    parser.add_argument('--output_dir', type=str, required=False, help='Output directory', default=None)
    parser.add_argument('--overwrite', type=bool, required=False, help='Overwrite existing simulation', default=False)
    parser.add_argument('--parallel', type=int, required=False, help='Number of parallel processes to use', default=1)

    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    # Load Config
    assert os.path.exists(args.config_template), f"Config file {args.config} does not exist!"

    plinder_ids = load_plinder_ids(args.filters)

    # Run Simulations
    if args.parallel > 1:
        # Create a partial function with fixed arguments
        process_func = partial(process_single_system, 
                             config_template=args.config_template,
                             output_dir=args.output_dir)
        
        # Create a pool of workers
        with mp.Pool(processes=args.parallel) as pool:
            # Map the function to all plinder_ids
            pool.map(process_func, plinder_ids)
    else:
        # Sequential processing
        for plinder_id in plinder_ids:                
            process_single_system(plinder_id, args.config_template, args.output_dir)


if __name__ == "__main__":
    main() 


#export PLINDER_MOUNT='/mnt/evafs/groups/sfglab/mwisniewski/Data'
#export PLINDER_RELEASE='2024-06'
#export PLINDER_ITERATION='v2'
#export PLINDER_OFFLINE='true'
#export OPENMM_DEFAULT_PLATFORM='CUDA'