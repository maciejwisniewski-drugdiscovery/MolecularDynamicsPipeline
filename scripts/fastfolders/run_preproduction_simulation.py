import os
import csv
import yaml
import argparse
from pathlib import Path
import logging

from molecular_dynamics_pipeline.utils.logger import setup_logger, log_info, log_error
from molecular_dynamics_pipeline.simulation.simulation import MDSimulation


def run_simulation_from_config(config_path: str, logger: logging.Logger):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    sim = MDSimulation(config)

    log_info(logger, f"Starting system setup for {config['info']['simulation_id']}")
    sim.set_system()
    sim._save_charges()
    sim._save_sigmas()
    sim._save_epsilons()
    sim._save_harmonic_bond_parameters()
    sim._save_harmonic_angle_parameters()
    sim._save_periodic_torsion_parameters()

    if sim.config['info']['simulation_status']['warmup'] == 'Not Done':
        log_info(logger, "Starting warmup phase")
        sim.warmup()
        sim.update_simulation_status('warmup', 'Done')
        log_info(logger, "Completed warmup phase")

    if sim.config['info']['simulation_status']['backbone_removal'] == 'Not Done':
        log_info(logger, "Starting backbone constraints removal")
        sim.remove_backbone_constraints()
        sim.update_simulation_status('backbone_removal', 'Done')
        log_info(logger, "Completed backbone constraints removal")

    if sim.config['info']['simulation_status']['nvt'] == 'Not Done':
        log_info(logger, "Starting NVT equilibration")
        sim.nvt()
        sim.update_simulation_status('nvt', 'Done')
        log_info(logger, "Completed NVT equilibration")

    if sim.config['info']['simulation_status']['npt'] == 'Not Done':
        log_info(logger, "Starting NPT equilibration")
        sim.npt()
        sim.update_simulation_status('npt', 'Done')
        log_info(logger, "Completed NPT equilibration")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run PreProduction Molecular Dynamics Simulation for FastFolder Protein')
    parser.add_argument('--config', type=str, default='/mnt/raid/mwisniewski/projects/phd/MolecularDynamicsPipeline/config/fastfolder_preproduction.yaml', required=False, help='Path to config template')
    args = parser.parse_args()

    sim_logger = setup_logger(name="fastfolder_preproduction", log_level=logging.INFO)
    
    run_simulation_from_config(args.config, sim_logger)