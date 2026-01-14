#!/usr/bin/env python3

import os
import csv
import yaml
import argparse
from pathlib import Path
import logging

from molecular_dynamics_pipeline.utils.logger import setup_logger, log_info, log_error
from molecular_dynamics_pipeline.simulation.simulation import MDSimulation


def create_system_config_from_template(template_config: str, protein_path: str, ligand_path: str, output_dir: str, ligand_id: str | None = None) -> str:
    with open(template_config, 'r') as f:
        config = yaml.safe_load(f)

    protein_path = str(Path(protein_path).resolve())
    ligand_path = str(Path(ligand_path).resolve())
    base_ligand_name = ligand_id if ligand_id else Path(ligand_path).stem
    system_id = f"{Path(protein_path).stem}__{base_ligand_name}"
    sim_id_suffix = 'bound_state'
    config['info']['system_id'] = system_id
    config['info']['simulation_id'] = f"{system_id}_simulation_{sim_id_suffix}"

    system_output_dir = Path(output_dir) / config['info']['simulation_id']
    system_output_dir.mkdir(parents=True, exist_ok=True)

    config['paths']['output_dir'] = str(system_output_dir)
    config['paths']['raw_protein_files'] = [protein_path]
    config['paths']['raw_ligand_files'] = [ligand_path]

    # Provide ligand identifiers so chain/residue IDs are meaningful in outputs
    config.setdefault('ligand_info', {})
    config['ligand_info']['ligand_names'] = [base_ligand_name]
    config['ligand_info']['ligand_ccd_codes'] = [base_ligand_name if ligand_id else 'LIG']

    system_config_path = system_output_dir / 'config.yaml'
    with open(system_config_path, 'w') as f:
        yaml.safe_dump(config, f)

    return str(system_config_path)


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

    if sim.config['info']['simulation_status']['production'] == 'Not Done':
        log_info(logger, "Starting production run")
        sim.production()
        sim.update_simulation_status('production', 'Done')
        log_info(logger, "Completed production run")


def main():
    parser = argparse.ArgumentParser(description='Run Molecular Dynamics Simulation for Tomek inputs')
    parser.add_argument('--config_template', type=str, required=False, default='config/tomek_parameters_bound.yaml', help='Path to config template')
    parser.add_argument('--csv', type=str, required=False, help='CSV with columns: protein_pdb,ligand_id,ligand_sdf,smiles')
    parser.add_argument('--protein', type=str, required=False, help='Protein file path')
    parser.add_argument('--ligand', type=str, required=False, help='Ligand file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Base output directory')

    args = parser.parse_args()

    logger = setup_logger(name="tomek_dynamics", log_level=logging.INFO)

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = []
    if args.csv:
        csv_path = Path(args.csv).resolve()
        base_dir = csv_path.parent
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            required = {'protein_pdb', 'ligand_sdf'}
            assert required.issubset(set(reader.fieldnames or [])), 'CSV must contain protein_pdb and ligand_sdf columns'
            for row in reader:
                protein_rel = (row.get('protein_pdb') or '').strip()
                ligand_rel = (row.get('ligand_sdf') or '').strip()
                ligand_id = (row.get('ligand_id') or '').strip() or None
                if not protein_rel or not ligand_rel:
                    continue
                protein = str((base_dir / protein_rel).resolve())
                ligand = str((base_dir / ligand_rel).resolve())
                tasks.append((protein, ligand, ligand_id))
    else:
        assert args.protein and args.ligand, 'Provide --protein and --ligand when --csv is not used'
        tasks.append((args.protein, args.ligand, None))

    for protein_path, ligand_path, ligand_id in tasks:
        try:
            config_path = create_system_config_from_template(
                template_config=args.config_template,
                protein_path=protein_path,
                ligand_path=ligand_path,
                output_dir=args.output_dir,
                ligand_id=ligand_id,
            )

            log_dir = Path(args.output_dir) / Path(config_path).parent.name / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            sim_logger = setup_logger(
                name=f"tomek_dynamics_{Path(protein_path).stem}",
                log_level=logging.INFO,
                log_dir=str(log_dir),
                console_output=True,
            )
            log_info(sim_logger, f"Starting simulation for protein={protein_path}, ligand={ligand_path}")
            log_info(sim_logger, f"Configuration loaded from {config_path}")
            run_simulation_from_config(config_path, sim_logger)
            log_info(sim_logger, "Completed all simulation stages")
        except Exception as e:
            log_error(logger, f"Error running simulation for protein=={protein_path} ligand={ligand_path}: {e}")


if __name__ == "__main__":
    main()


