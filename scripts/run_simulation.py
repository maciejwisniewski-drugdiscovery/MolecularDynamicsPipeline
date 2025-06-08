#!/usr/bin/env python3

import os
import yaml
import argparse

import logging
from dynamics_pipeline.utils.logger import setup_logger, log_info, log_error, log_warning, log_debug
from dynamics_pipeline.simulation.simulation import MDSimulation

# VARIABLES
PLINDER_DIR = os.getenv('PLINDER_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')

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
    parser.add_argument('--config', type=str, required=False, help='Path to config file', default='/mnt/raid/mwisniewski/Projects/plinder_dynamics/projects/student_projects/skoczylas/8RWR_ligand_113.yaml')
    args = parser.parse_args()


    assert os.path.exists(args.config), f"Config file {args.config} does not exist!"
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Run Simulation
    run_simulation(config)


if __name__ == "__main__":
    single_simulation() 