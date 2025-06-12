#!/usr/bin/env python3

import os
import yaml
import argparse
from typing import Optional
import random

import logging
from dynamics_pipeline.utils.logger import setup_logger, log_info, log_error, log_warning, log_debug
from dynamics_pipeline.simulation.simulation import MDSimulation
from dynamics_pipeline.data.misato import load_misato_ids, create_system_config, split_misato_complex_filepath


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
    parser.add_argument('--config_template', type=str, required=False, help='Path to config template file', default='config/misato_parameters.yaml')
    parser.add_argument('--output_dir', type=str, required=False, help='Output directory', default=None)
    parser.add_argument('--overwrite', type=bool, required=False, help='Overwrite existing simulation', default=False)
    parser.add_argument('--misato_ids_filepath', type=str, required=False, help='Path to misato ids file', default=None)
    parser.add_argument('--ccd_pkl', type=str, required=False, help='Path to ccd pkl file', default=None)
    
    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    # Load Config
    assert os.path.exists(args.config_template), f"Config file {args.config} does not exist!"

    misato_ids = load_misato_ids(args.misato_ids_filepath)

    # Run Simulation
    for misato_id in misato_ids:  
        try:
            system_config_filepath = create_system_config(template_config = args.config_template, system_id = misato_id, output_dir = args.output_dir)
            with open(system_config_filepath, 'r') as f:
                config = yaml.safe_load(f)
            if any(not os.path.exists(fp) for fp in config['paths']['raw_protein_files'] + config['paths']['raw_ligand_files']):
                split_misato_complex_filepath(config, ccd_pkl_filepath = args.ccd_pkl)    
            run_simulation(config)
        except Exception as e:
            log_error(logger, f"Error: {e}")
            continue


if __name__ == "__main__":
    single_simulation() 


#export PLINDER_MOUNT='/mnt/evafs/groups/sfglab/mwisniewski/Data'
#export PLINDER_RELEASE='2024-06'
#export PLINDER_ITERATION='v2'
#export PLINDER_OFFLINE='true'
#export OPENMM_DEFAULT_PLATFORM='CUDA'