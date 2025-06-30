"""Module for handling Plinder data processing and system configuration.

This module provides functionality for:
- Loading and filtering Plinder system IDs
- Creating system configurations for molecular dynamics simulations
- Processing ligand and protein files for both bound and unbound states
"""

import os
import dotenv
dotenv.load_dotenv('plinder.env')
import random
from typing import Optional
from pathlib import Path
from plinder.core.scores import query_index
import yaml
from dynamics_pipeline.data.small_molecule import fix_molecule_with_pybel, create_unbound_ligand_files
from dynamics_pipeline.data.biomolecules import fix_biomolecule_with_pdb2pqr

class PlinderFilters:
    """Class for handling Plinder filtering configurations.

    This class reads and processes filter configurations from a YAML file to be used
    in querying the Plinder index.

    Attributes:
        columns (list): List of columns to be included in the query
        splits (list): List of data splits to be included
        filters (list): List of tuples containing (column, operator, value) for filtering
    """

    def __init__(self, filters: Path):
        """Initialize PlinderFilters with a filter configuration file.

        Args:
            filters (Path): Path to the YAML filter configuration file
        """
        config = self.read_yaml(filters)

        self.columns = config["columns"]
        self.splits = config["splits"]
        self.filters = [(filter['column'], filter['operator'], filter['value']) for filter in config["filters"]]

    def read_yaml(self, filters: Path):
        """Read and parse a YAML configuration file.

        Args:
            filters (Path): Path to the YAML configuration file

        Returns:
            dict: Parsed YAML configuration
        """
        with open(filters, "r") as f:
            config = yaml.safe_load(f)
        return config


def load_plinder_ids(filters: Optional[str] = None):
    """Load Plinder system IDs based on specified filters.

    Args:
        filters (Optional[str], optional): Path to the filter configuration file. Defaults to None.

    Returns:
        set: Set of Plinder system IDs that match the specified filters
    """
    plinder_filters = PlinderFilters(filters)
    PLINDEX = query_index(
        columns=plinder_filters.columns,
        splits=plinder_filters.splits,
        filters=plinder_filters.filters
    )
    PLINDEX = PLINDEX.iloc[10:11]
    plinder_ids = set(PLINDEX['system_id'].tolist())
    return plinder_ids


def create_system_config(template_config: Path, system_id: str, output_dir: Path):
    """Create a system configuration for molecular dynamics simulation.

    This function generates a configuration file for a specific system based on a template,
    handling both bound and unbound states. It processes protein and ligand files,
    sets up appropriate directories, and saves the configuration.

    Args:
        template_config (Path): Path to the template configuration file
        system_id (str): Identifier for the molecular system
        output_dir (Path): Directory where the configuration and processed files will be saved

    Returns:
        str: Path to the created system configuration file
    """
    with open(template_config, 'r') as f:
        config = yaml.safe_load(f)

    config['info']['system_id'] = system_id
    if config['info']['bound_state'] == True:
        config['info']['simulation_id'] = f"{system_id}_simulation_bound_state"
    else:
        config['info']['simulation_id'] = f"{system_id}_simulation_unbound_state"

    system_config_filepath = os.path.join(output_dir, config['info']['simulation_id'], 'config.yaml')

    if not os.path.exists(system_config_filepath):

        config['paths']['output_dir'] = os.path.join(output_dir, config['info']['simulation_id'])
        os.makedirs(os.path.join(output_dir, config['info']['simulation_id']), exist_ok=True)

        config['paths']['raw_protein_files'] = [os.path.join(os.getenv('PLINDER_MOUNT'), 'plinder', os.getenv('PLINDER_RELEASE'),os.getenv('PLINDER_ITERATION'), 'systems', system_id, 'receptor.cif')]

        plinder_ligand_dir = os.path.join(os.getenv('PLINDER_MOUNT'), 'plinder',os.getenv('PLINDER_RELEASE'),
                                          os.getenv('PLINDER_ITERATION'), 'systems', system_id, 'ligand_files')
        
        plinder_ligand_filepaths = [os.path.join(plinder_ligand_dir, x) for x in os.listdir(plinder_ligand_dir)]
        plinder_ligand_filepaths = [fix_molecule_with_pybel(input_filepath = ligand_filepath, output_dir = config['paths']['output_dir']) for ligand_filepath in plinder_ligand_filepaths]

        if config['info']['bound_state'] == True:
            config['paths']['raw_ligand_files'] = plinder_ligand_filepaths
        else:
            config['paths']['raw_ligand_files'] = [
                unbound_lig_filepath for unbound_lig_filepath in
                create_unbound_ligand_files(plinder_ligand_filepaths, output_dir=config['paths']['output_dir'])
            ]

        with open(system_config_filepath, 'w') as f:
            yaml.dump(config, f)

    return system_config_filepath