#!/usr/bin/env python3

import os
import yaml
import argparse
from typing import Optional
import random

import logging
from molecular_dynamics_pipeline.utils.logger import setup_logger, log_info, log_error, log_warning, log_debug
from molecular_dynamics_pipeline.simulation.simulation import MDSimulation
from molecular_dynamics_pipeline.data.misato import load_misato_ids, create_system_config, split_misato_complex_filepath


random.seed(42)


# Setup global logger
logger = setup_logger(name="plinder_dynamics", log_level=logging.INFO)


def run_simulation(config, sim_logger):
    # Initialize and run simulation
    sim = MDSimulation(config)
    
    # Preprocessing Step
    log_info(sim_logger, f"Starting system setup for {config['info']['system_id']}")
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

    # Energy Calculations
    energy_stages = ['nvt_energy_calculation', 'npt_energy_calculation', 'production_energy_calculation']
    for energy_stage in energy_stages:
        base_stage = energy_stage.replace('_energy_calculation', '')
        if sim.config['info']['simulation_status'][energy_stage] == 'Not Done':
            # Only run energy calculation if the corresponding base stage is done
            if sim.config['info']['simulation_status'][base_stage] == 'Done':
                log_info(sim_logger, f"Starting {energy_stage}")
                energy_method = getattr(sim, energy_stage)
                energy_method()
                sim.update_simulation_status(energy_stage, 'Done')
                log_info(sim_logger, f"Completed {energy_stage}")
            else:
                log_info(sim_logger, f"Skipping {energy_stage} as {base_stage} is not completed")

    
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
            
            # Create system-specific output directory
            system_output_dir = os.path.join(args.output_dir, config['info']['system_id'])
            os.makedirs(system_output_dir, exist_ok=True)
            
            # Setup system-specific logger
            log_dir = os.path.join(system_output_dir, 'logs')
            sim_logger = setup_logger(
                name=f"misato_dynamics_{misato_id}",
                log_level=logging.INFO,
                log_dir=log_dir,
                console_output=True
            )
            
            log_info(sim_logger, f"Starting simulation for system {misato_id}")
            log_info(sim_logger, f"Configuration loaded from {system_config_filepath}")
                
            if any(not os.path.exists(fp) for fp in config['paths']['raw_protein_files'] + config['paths']['raw_ligand_files']):
                split_misato_complex_filepath(config, ccd_pkl_filepath = args.ccd_pkl)    
            run_simulation(config, sim_logger)
            
            log_info(sim_logger, f"Completed all simulation stages for system {misato_id}")
            
        except Exception as e:
            if 'sim_logger' in locals():
                log_error(sim_logger, f"Error in simulation {misato_id}: {str(e)}")
            else:
                log_error(logger, f"Error setting up simulation {misato_id}: {str(e)}")
            continue


if __name__ == "__main__":
    single_simulation() 


#export PLINDER_MOUNT='/mnt/evafs/groups/sfglab/mwisniewski/Data'
#export PLINDER_RELEASE='2024-06'
#export PLINDER_ITERATION='v2'
#export PLINDER_OFFLINE='true'
#export OPENMM_DEFAULT_PLATFORM='CUDA'