#!/usr/bin/env python3

import os
import csv
import yaml
import argparse
from pathlib import Path
import logging

from molecular_dynamics_pipeline.utils.logger import setup_logger, log_info, log_error
from molecular_dynamics_pipeline.simulation.simulation import MDSimulation


def create_system_config_from_template(
    template_config: Path,
    protein_path: Path,
    output_dir: Path,
) -> str:
    """
    Create a system configuration for a production simulation.
    Args:
        template_config: The path to the template configuration file.
        protein_path: The path to the protein file.
        output_dir: The path to the output directory.
    Returns:
        The path to the system configuration file.
    """
    with open(template_config, 'r') as f:
        config = yaml.safe_load(f)
    
    system_id = f"{protein_path.stem}"
    config['info']['system_id'] = system_id
    config['info']['simulation_id'] = f"{system_id}_simulation_production"

    system_output_dir = output_dir / config['info']['simulation_id']
    system_output_dir.mkdir(parents=True, exist_ok=True)

    config['paths']['output_dir'] = str(system_output_dir)
    config['paths']['raw_protein_files'] = [str(protein_path)]
    config['paths']['raw_ligand_files'] = False

    system_config_path = system_output_dir / 'config.yaml'
    with open(system_config_path, 'w') as f:
        yaml.safe_dump(config, f)

    return system_config_path

def run_simulation_from_config(config_path: Path, logger: logging.Logger):
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

    if sim.config['info']['simulation_status']['production'] == 'Not Done':
        log_info(logger, "Starting production run")
        sim.production()
        sim.update_simulation_status('production', 'Done')
        log_info(logger, "Completed production run")


def main():
    parser = argparse.ArgumentParser(description='Run Molecular Dynamics Simulation for Tomek inputs')
    parser.add_argument('--config_template', type=Path, required=False, default='config/fastfolders_production.yaml', help='Path to config template')
    parser.add_argument('--input_dir', type=Path, required=True, help='Path to input directory')
    parser.add_argument('--protein_number', type=int, required=True, help='Protein number')
    parser.add_argument('--output_dir', type=Path, required=True, help='Base output directory')

    args = parser.parse_args()

    logger = setup_logger(name="fastfolders_production", log_level=logging.INFO)

    os.makedirs(args.output_dir, exist_ok=True)
    
    protein_filepath = args.input_dir / f"representative_frame_{args.protein_number}.cif"

    config_path = create_system_config_from_template(
        template_config = args.config_template,
        protein_path = protein_filepath,
        output_dir=args.output_dir,
    )

    # Setup Logger
    log_dir = Path(args.output_dir) / Path(config_path).parent.name / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    sim_logger = setup_logger(
        name=f"fastfolders_production_{Path(protein_filepath).stem}",
        log_level=logging.INFO,
        log_dir=str(log_dir),
        console_output=True,
    )

    # Run Simulation
    log_info(sim_logger, f"Starting simulation for protein={protein_filepath}")
    log_info(sim_logger, f"Configuration loaded from {config_path}")
    run_simulation_from_config(config_path, sim_logger)
    log_info(sim_logger, "Completed all simulation stages")


if __name__ == "__main__":
    main()