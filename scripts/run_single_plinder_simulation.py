#!/usr/bin/env python3

import os
import yaml
import argparse
from typing import Optional
import random

import logging
from dynamics_pipeline.utils.logger import setup_logger, log_info, log_error, log_warning, log_debug
from dynamics_pipeline.simulation.simulation import MDSimulation
from dynamics_pipeline.data.plinder import load_plinder_ids, create_system_config

random.seed(42)


# Setup global logger
logger = setup_logger(name="plinder_dynamics", log_level=logging.INFO)


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

    
def single_simulation():
    parser = argparse.ArgumentParser(description='Run Molecular Dynamics Simulation with OpenMM')
    parser.add_argument('--config_template', type=str, required=False, help='Path to config template file', default='config/plinder_parameters.yaml')
    parser.add_argument('--filters', type=str, required=False, help='Filters to apply to the PLINDER systems', default='scripts/filters/train_plinder.yaml')
    parser.add_argument('--output_dir', type=str, required=False, help='Output directory', default=None)
    parser.add_argument('--overwrite', type=bool, required=False, help='Overwrite existing simulation', default=False)

    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    # Load Config
    assert os.path.exists(args.config_template), f"Config file {args.config} does not exist!"

    plinder_ids = load_plinder_ids(args.filters)

    # Run Simulation
    for plinder_id in plinder_ids:
        os.makedirs(os.path.join(args.output_dir, plinder_id), exist_ok=True)
                
        system_config_filepath = create_system_config(template_config = args.config_template, system_id = plinder_id, output_dir = args.output_dir)
        
        with open(system_config_filepath, 'r') as f:
            config = yaml.safe_load(f)
        run_simulation(config)


if __name__ == "__main__":
    single_simulation() 


#export PLINDER_MOUNT='/mnt/evafs/groups/sfglab/mwisniewski/Data'
#export PLINDER_RELEASE='2024-06'
#export PLINDER_ITERATION='v2'
#export PLINDER_OFFLINE='true'
#export OPENMM_DEFAULT_PLATFORM='CUDA'