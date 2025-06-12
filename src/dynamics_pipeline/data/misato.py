import os
import gzip
import random
import pickle
import argparse
from pathlib import Path
from typing import Tuple
import yaml
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.mol as mol

from rdkit import Chem
from rdkit.Chem import AllChem

from dynamics_pipeline.data.small_molecule import make_unbound

def load_misato_ids(misato_ids_filepath: str):
    assert os.path.exists(misato_ids_filepath), f"Misato IDs file {misato_ids_filepath} does not exist!"
    with open(misato_ids_filepath, 'r') as f:
        misato_ids = f.read().splitlines()
    return misato_ids

def create_system_config(template_config: Path, system_id: str, output_dir: Path):
    with open(template_config, 'r') as f:
        config = yaml.safe_load(f)

    config['info']['system_id'] = system_id
    if config['info']['bound_state'] == True:
        config['info']['simulation_id'] = f"{system_id}_simulation_bound_state"
    else:
        config['info']['simulation_id'] = f"{system_id}_simulation_unbound_state"

    system_config_filepath = os.path.join(output_dir, config['info']['simulation_id'], 'config.yaml')

    if not os.path.exists(system_config_filepath):
        config['paths']['raw_misato_file'] = os.path.join(os.getenv('MISATO_DIR'), system_id+'.cif.gz')
        config['paths']['output_dir'] = os.path.join(output_dir, config['info']['simulation_id'])
        config['paths']['raw_protein_files'] = [os.path.join(output_dir, config['info']['simulation_id'],'raw_protein.cif')]
        config['paths']['raw_ligand_files'] = [os.path.join(output_dir, config['info']['simulation_id'],'raw_ligand.sdf')]
        os.makedirs(os.path.join(output_dir, config['info']['simulation_id']), exist_ok=True)
        with open(system_config_filepath, 'w') as f:
            yaml.dump(config, f)

    return system_config_filepath

def split_misato_complex_filepath(config: dict, ccd_pkl_filepath: str):
    """
    Split a compressed CIF complex file into separate protein (CIF) and ligand (SDF) files.
    
    Args:
        complex_filepath: Path to the input complex file (.cif.gz)
        output_dir: Directory to save the output files
        
    Returns:
        Tuple containing paths to (protein_file, ligand_file)
    """
    # Create output directory if it doesn't exist
    complex_filepath = Path(config['paths']['raw_misato_file'])
    output_dir = Path(config['paths']['output_dir'])

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filenames
    base_name = complex_filepath.stem.replace('.cif.gz', '')  # Remove both .cif and .gz
    protein_filepath = config['paths']['raw_protein_files'][0]
    ligand_filepath = config['paths']['raw_ligand_files'][0]
    
    # Read the compressed CIF file
    with gzip.open(complex_filepath, 'rt') as f:
        complex_file = pdbx.CIFFile.read(f)
        array = pdbx.get_structure(complex_file, include_bonds=True)
    
    # Split into protein and ligand based on residue names
    # Protein residues are typically standard amino acids
    
    protein_structure = array[0][array.hetero == False]
    
    # Save protein as CIF

    protein_file = pdbx.CIFFile()
    pdbx.set_structure(protein_file, protein_structure)
    protein_file.write(protein_filepath)

    # Save ligand as SDF
    if config['info']['bound_state'] == True:
        ligand_structure = array[0][array.hetero == True]
        ligand_file = mol.SDFile()
        mol.set_structure(ligand_file, ligand_structure)
        ligand_file.write(ligand_filepath)
    else:
        with open(ccd_pkl_filepath, 'rb') as f:
            ccd_pkl = pickle.load(f)
        ligand_resname = array[0][array.hetero == True].res_name[0]+'_misato'
        ligand_mol = ccd_pkl[ligand_resname]

        unbound_ligand_mol = make_unbound(ligand_mol, ref_structure = protein_structure)

        w = Chem.SDWriter(ligand_filepath)
        w.write(unbound_ligand_mol)
        w.close()

        