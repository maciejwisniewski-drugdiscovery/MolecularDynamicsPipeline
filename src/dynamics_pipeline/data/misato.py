import os
import gzip
import random
import pickle
import argparse
from pathlib import Path
from typing import Tuple
import yaml
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.mol as mol
import biotite.interface.rdkit as rdk
import logging
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

from dynamics_pipeline.data.small_molecule import make_unbound

def load_misato_ids(misato_ids_filepath: str):
    assert os.path.exists(misato_ids_filepath), f"Misato IDs file {misato_ids_filepath} does not exist!"
    with open(misato_ids_filepath, 'r') as f:
        misato_ids = f.read().splitlines()
    return misato_ids


# --- MISATO DYNAMICS ---

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


# --- MISATO MC ---

def get_misato_protein_structure(complex_filepath: str):
    with gzip.open(complex_filepath, 'rt') as f:
        complex_file = pdbx.CIFFile.read(f)
        array = pdbx.get_structure(complex_file, include_bonds=True)
    
    protein_structure = array[0][array.hetero == False]
    return protein_structure

def get_misato_ligand_structure(complex_filepath: str):
    with gzip.open(complex_filepath, 'rt') as f:
        complex_file = pdbx.CIFFile.read(f)
        array = pdbx.get_structure(complex_file, include_bonds=True)
    
    ligand_structure = array[0][array.hetero == True]
    return ligand_structure


def monte_carlo_ligand_sampling(
    ligand_mol: Chem.Mol,
    protein_structures: list[struc.AtomArray],
    offset_a: float,
    distance_a: float,
    distance_b: float,
    distance_c: float,
    ligand_subset_indices: list[int],
    num_conformers_to_generate: int = 50,
    n_samples: int = 500,
    z_samples: int = 100,
    max_trials: int = 100000,
    logger: logging.Logger = None
) -> list[tuple[Chem.Mol, struc.AtomArray]]:
    """
    Generate diverse ligand conformations around a protein using Monte Carlo sampling.

    Args:
        ligand_mol: The ligand as an RDKit Molecule object.
        protein_structures: A list of protein conformers as Biotite AtomArrays.
        offset_a: Minimum distance from the protein's center of gravity, added to the radius of gyration.
        distance_a: Maximum distance from the nearest protein atom.
        distance_b: Minimum distance between two different generated ligand conformations.
        distance_c: Minimum distance for specific ligand atoms.
        ligand_subset_indices: List of indices of ligand atoms to check for distance_c.
        num_conformers_to_generate: The number of conformers to generate for the ligand.
        n_samples: The number of valid conformations to generate before diversity selection.
        z_samples: The final number of diverse conformations to return.
        logger: A logger object for progress updates.

    Returns:
        A list of tuples, where each tuple contains an RDKit Mol object
        with a single conformation and the corresponding Biotite AtomArray
        of the protein it was placed against.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    if not protein_structures:
        raise ValueError("The protein_structures list cannot be empty.")

    # Pre-compute protein properties and build KD-Trees for faster access
    precomputed_protein_data = []
    logger.info("Pre-computing protein properties and KD-Trees...")
    for protein_structure in tqdm(protein_structures, desc="Preprocessing Proteins"):
        precomputed_protein_data.append({
            'com': struc.mass_center(protein_structure),
            'rgyr': struc.gyration_radius(protein_structure),
            'kdtree': cKDTree(protein_structure.coord),
            'structure': protein_structure
        })

    accepted_confs = []
    ligand_template = Chem.Mol(ligand_mol)
    ligand_template.RemoveAllConformers()

    # Generate multiple conformers for the ligand to increase diversity
    num_rotatable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(ligand_template)
    num_conformers_to_generate = num_conformers_to_generate if num_rotatable_bonds > 0 else 1

    if num_rotatable_bonds > 0:
        AllChem.EmbedMultipleConfs(ligand_template, numConfs=num_conformers_to_generate, params=AllChem.ETKDG())
    else:
        AllChem.EmbedMolecule(ligand_template, AllChem.ETKDG())
    
    if ligand_template.GetNumConformers() == 0:
        raise ValueError("Could not generate any conformers for the ligand.")

    ligand_conformers = ligand_template.GetConformers()

    # Define a global sampling box that encompasses all protein conformers
    all_protein_coords = np.vstack([p.coord for p in protein_structures])
    box_min = all_protein_coords.min(axis=0) - distance_a
    box_max = all_protein_coords.max(axis=0) + distance_a
    box_size = box_max - box_min

    trials = 0
    with tqdm(total=n_samples, desc="Generating Conformations") as pbar:
        while len(accepted_confs) < n_samples and trials < max_trials:
            trials += 1

            # Select a random pre-processed protein conformer for this trial
            selected_protein_data = random.choice(precomputed_protein_data)
            protein_com = selected_protein_data['com']
            protein_rgyr = selected_protein_data['rgyr']
            protein_kdtree = selected_protein_data['kdtree']

            # Select a random ligand conformer to place
            selected_conformer = random.choice(ligand_conformers)
            initial_ligand_coords = selected_conformer.GetPositions()
            initial_ligand_com = initial_ligand_coords.mean(axis=0)

            # 1. Random rotation
            rotation = R.random().as_matrix()
            coords = np.dot(initial_ligand_coords - initial_ligand_com, rotation)

            # 2. Random translation
            translation = box_min + np.random.rand(3) * box_size
            coords += translation

            # 3. Check constraints against the selected protein conformer
            ligand_com = coords.mean(axis=0)
            
            # Constraint 1: Distance from protein CoM
            dist_to_prot_com = np.linalg.norm(ligand_com - protein_com)
            if dist_to_prot_com <= protein_rgyr + offset_a:
                logger.debug(f"Trial {trials}: Rejected. Distance to CoM ({dist_to_prot_com:.2f}) is less than threshold.")
                continue

            # Constraint 2: Min distance to any protein atom using KD-Tree
            # This is much faster than cdist for large numbers of atoms.
            min_dist_to_protein, _ = protein_kdtree.query(coords, k=1)
            if min_dist_to_protein.min() >= distance_a:
                logger.debug(f"Trial {trials}: Rejected. Min distance to protein ({min_dist_to_protein.min():.2f}) is greater than threshold.")
                continue
            
            # Constraint 3: Min distance for specific ligand atoms
            ligand_subset_coords = coords[ligand_subset_indices]
            min_dist_subset, _ = protein_kdtree.query(ligand_subset_coords, k=1)
            if min_dist_subset.min() < distance_c:
                logger.debug(f"Trial {trials}: Rejected. Min distance for ligand subset ({min_dist_subset.min():.2f}) is less than threshold {distance_c}.")
                continue
            
            # Conformation accepted
            new_mol = Chem.Mol(ligand_template)
            new_mol.RemoveAllConformers()
            conformer = Chem.Conformer(new_mol.GetNumAtoms())
            for i in range(new_mol.GetNumAtoms()):
                x, y, z = coords[i]
                conformer.SetAtomPosition(i, (x, y, z))
            new_mol.AddConformer(conformer, assignId=True)
            accepted_confs.append((new_mol, selected_protein_data['structure']))
            pbar.update(1)
            logger.info(f"Accepted conformation {pbar.n}/{pbar.total}. Trials: {trials}.")

    if len(accepted_confs) < n_samples:
        logger.warning(f"Warning: Only generated {len(accepted_confs)}/{n_samples} conformations satisfying the criteria.")
        if not accepted_confs:
            return []

    # 4. Select diverse conformations
    diverse_confs = []
    
    # Shuffle to pick a random starting point
    random.shuffle(accepted_confs)
    
    if not accepted_confs:
         return []
    
    # Start with the first accepted conformation and its associated protein
    first_entry = accepted_confs.pop(0)
    diverse_confs.append(first_entry)
    
    # Keep track of coordinates of diverse ligands
    # TODO: maybe clustering the diverse ligands would be better?
    diverse_coords_list = [first_entry[0].GetConformer().GetPositions()]

    for candidate_entry in accepted_confs:
        if len(diverse_confs) >= z_samples:
            break

        candidate_mol, _ = candidate_entry
        candidate_coords = candidate_mol.GetConformer().GetPositions()
        is_diverse = True
        for existing_coords in diverse_coords_list:
            min_inter_dist = cdist(candidate_coords, existing_coords).min()
            if min_inter_dist < distance_b:
                is_diverse = False
                break
        
        if is_diverse:
            diverse_confs.append(candidate_entry)
            diverse_coords_list.append(candidate_coords)

    return diverse_confs[:z_samples]

def mcmc_ligand_sampling(
    ligand_mol: Chem.Mol,
    protein_structures: list[struc.AtomArray],
    offset_a: float,
    distance_a: float,
    distance_b: float,
    distance_c: float,
    ligand_subset_indices: list[int],
    step_size_translation: float = 0.5, # Angstrom
    step_size_rotation: float = 5.0, # Degrees
    num_conformers_to_generate: int = 50,
    n_samples: int = 500,
    z_samples: int = 100,
    max_trials: int = 100000,
    thinning: int = 20,
    logger: logging.Logger = None
) -> list[tuple[Chem.Mol, struc.AtomArray]]:
    """
    Generate diverse ligand conformations around a protein using MCMC sampling.
    In each step, both protein and ligand conformers are randomly selected.
    This version is optimized for speed and includes internal coordinate moves.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    if not protein_structures:
        raise ValueError("The protein_structures list cannot be empty.")

    # --- Pre-computation Step ---
    logger.info("Pre-computing protein and ligand data...")
    
    # Pre-compute protein properties and build KD-Trees
    precomputed_protein_data = []
    for protein_structure in tqdm(protein_structures, desc="Preprocessing Proteins"):
        precomputed_protein_data.append({
            'structure': protein_structure,
            'com': struc.mass_center(protein_structure),
            'rgyr': struc.gyration_radius(protein_structure),
            'kdtree': cKDTree(protein_structure.coord)
        })

    # Generate and pre-compute ligand conformers
    ligand_template = Chem.Mol(ligand_mol)
    ligand_template.RemoveAllConformers()
    num_rotatable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(ligand_template)
    num_conformers_to_generate = num_conformers_to_generate if num_rotatable_bonds > 0 else 1
    if num_rotatable_bonds > 0:
        AllChem.EmbedMultipleConfs(ligand_template, numConfs=num_conformers_to_generate, params=AllChem.ETKDG())
    else:
        AllChem.EmbedMolecule(ligand_template, AllChem.ETKDG())
    if ligand_template.GetNumConformers() == 0:
        raise ValueError("Could not generate any conformers for the ligand.")
    
    # --- MCMC Initialization: Find a starting position ---
    logger.info("Finding a valid starting position for the MCMC chain...")
    initial_conf = None
    trials = 0
    all_protein_coords = np.vstack([p['kdtree'].data for p in precomputed_protein_data])
    box_min = all_protein_coords.min(axis=0) - distance_a
    box_max = all_protein_coords.max(axis=0) + distance_a
    box_size = box_max - box_min

    while trials < max_trials:
        trials += 1
        current_protein_info = random.choice(precomputed_protein_data)
        
        initial_ligand_conformer = random.choice(ligand_template.GetConformers())
        initial_ligand_coords = initial_ligand_conformer.GetPositions()
        initial_ligand_com = initial_ligand_coords.mean(axis=0)
        
        rotation = R.random().as_matrix()
        coords = np.dot(initial_ligand_coords - initial_ligand_com, rotation)
        translation = box_min + np.random.rand(3) * box_size
        coords += translation

        ligand_com = coords.mean(axis=0)
        dist_to_prot_com = np.linalg.norm(ligand_com - current_protein_info['com'])
        
        min_dist_to_protein, _ = current_protein_info['kdtree'].query(coords, k=1)
        min_dist_subset, _ = current_protein_info['kdtree'].query(coords[ligand_subset_indices], k=1)
        
        if (dist_to_prot_com > current_protein_info['rgyr'] + offset_a and 
            min_dist_to_protein.min() < distance_a and
            min_dist_subset.min() >= distance_c):
            new_mol = Chem.Mol(ligand_template)
            new_mol.RemoveAllConformers()
            conformer = Chem.Conformer(new_mol.GetNumAtoms())
            for i in range(new_mol.GetNumAtoms()):
                x, y, z = coords[i]
                conformer.SetAtomPosition(i, (x, y, z))
            new_mol.AddConformer(conformer, assignId=True)
            initial_conf = (new_mol, current_protein_info)
            logger.info(f"Found initial position after {trials} trials.")
            break
    
    if initial_conf is None:
        raise RuntimeError(f"Could not find a valid starting position after {max_trials} trials.")

    # --- MCMC Sampling Loop ---
    collected_confs = [initial_conf]
    current_mol, current_protein_info = initial_conf
    
    total_steps = n_samples * thinning
    with tqdm(total=total_steps, desc="Running MCMC Sampling") as pbar:
        for step in range(total_steps):
            pbar.update(1)
            
            # Combined MCMC move: protein swap, internal geometry change, and rigid body transformation
            
            # 1. Swap protein
            proposed_protein_info = random.choice(precomputed_protein_data)

            # 2. Internal geometry change (new conformer) aligned to previous state
            temp_mol = Chem.Mol(ligand_template)
            temp_mol.RemoveAllConformers()
            conf = random.choice(ligand_template.GetConformers())
            temp_mol.AddConformer(conf, assignId=True)
            AllChem.AlignMol(temp_mol, current_mol)
            proposed_coords = temp_mol.GetConformer().GetPositions()

            # 3. Rigid body transformation
            current_com = proposed_coords.mean(axis=0)
            translation = (np.random.rand(3) - 0.5) * 2 * step_size_translation
            rotation_angle = np.deg2rad(step_size_rotation)
            random_axis = np.random.rand(3)
            norm = np.linalg.norm(random_axis)
            if norm > 1e-6: # Avoid division by zero
                random_axis /= norm
            
            rotation = R.from_rotvec(rotation_angle * (np.random.rand() * 2 - 1) * random_axis)
            proposed_coords = rotation.apply(proposed_coords - current_com) + current_com + translation

            proposed_ligand_com = proposed_coords.mean(axis=0)
            dist_to_prot_com = np.linalg.norm(proposed_ligand_com - proposed_protein_info['com'])
            
            min_dist_to_protein, _ = proposed_protein_info['kdtree'].query(proposed_coords, k=1)
            min_dist_subset, _ = proposed_protein_info['kdtree'].query(proposed_coords[ligand_subset_indices], k=1)

            if (dist_to_prot_com > proposed_protein_info['rgyr'] + offset_a and 
                min_dist_to_protein.min() < distance_a and
                min_dist_subset.min() >= distance_c):
                
                new_mol = Chem.Mol(ligand_template)
                new_mol.RemoveAllConformers()
                conformer = Chem.Conformer(new_mol.GetNumAtoms())
                for i in range(new_mol.GetNumAtoms()):
                    x, y, z = proposed_coords[i]
                    conformer.SetAtomPosition(i, (x, y, z))
                new_mol.AddConformer(conformer, assignId=True)
                current_mol = new_mol
                current_protein_info = proposed_protein_info
                
                if step % thinning == 0:
                    collected_confs.append((current_mol, current_protein_info))

    logger.info(f"MCMC finished. Collected {len(collected_confs)} samples.")
    if not collected_confs:
        return []

    # --- Diversification Step (Optimized) ---
    logger.info("Selecting diverse conformations...")
    diverse_confs = []
    random.shuffle(collected_confs)
    
    if not collected_confs:
        return []
        
    first_mol, first_protein_info = collected_confs.pop(0)
    diverse_confs.append((first_mol, first_protein_info['structure']))
    
    diverse_coords_stack = first_mol.GetConformer().GetPositions()

    with tqdm(total=min(z_samples, len(collected_confs)), desc="Diversifying Conformations") as pbar:
        pbar.update(1)
        for candidate_mol, candidate_protein_info in collected_confs:
            if len(diverse_confs) >= z_samples:
                break

            candidate_coords = candidate_mol.GetConformer().GetPositions()
            
            min_inter_dist = cdist(candidate_coords, diverse_coords_stack).min()
            
            if min_inter_dist >= distance_b:
                diverse_confs.append((candidate_mol, candidate_protein_info['structure']))
                diverse_coords_stack = np.vstack([diverse_coords_stack, candidate_coords])
                pbar.update(1)

    return diverse_confs[:z_samples]

def generate_misato_unbound_data(misato_id, 
                                ccd_pkl_filepath, 
                                misato_dir, 
                                output_dir, 
                                method: str = 'mcmc',
                                offset_a = 8,
                                distance_a = 30,
                                distance_b = 5,
                                distance_c = 6,
                                num_conformers_to_generate = 50,
                                n_samples = 500,
                                z_samples = 100,
                                max_trials = 100000,
                                overwrite = False,
                                logger=None):

    if logger is None:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    misato_complex_filepaths = [os.path.join(misato_dir, x) for x in os.listdir(misato_dir) if misato_id in x]
    if len(misato_complex_filepaths) == 0:
        logger.warning(f"No misato complex filepaths found for {misato_id} in {misato_dir}.")
        return []
    
    misato_complex_output_dir = os.path.join(output_dir, misato_id)
    if os.path.exists(misato_complex_output_dir):
        if len(os.listdir(misato_complex_output_dir)) == z_samples:
            logger.info(f"Found {z_samples} conformations for {misato_id} in {misato_complex_output_dir}. Skipping generation.")
            return []
        else:
            logger.info(f"Found {len(os.listdir(misato_complex_output_dir))} conformations for {misato_id} in {misato_complex_output_dir}. Continuing generation.")
    os.makedirs(misato_complex_output_dir, exist_ok=True)

    protein_structures = [get_misato_protein_structure(misato_complex_filepath) for misato_complex_filepath in misato_complex_filepaths]
    if len(misato_complex_filepaths) == 0:
        raise ValueError(f"No misato complex filepaths found for {misato_id}")
    ligand_structure = get_misato_ligand_structure(misato_complex_filepaths[0])

    # Load CCD pkl and get ligand mol
    with open(ccd_pkl_filepath, 'rb') as f:
        ccd_pkl = pickle.load(f)
    ligand_resname = ligand_structure.res_name[0]+'_misato'
    ligand_mol = ccd_pkl[ligand_resname]
    
    # Define the subset of ligand atoms to check for distance_c (e.g., heavy atoms)
    ligand_heavy_atom_indices = [atom.GetIdx() for atom in ligand_mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if not ligand_heavy_atom_indices:
        logger.warning("Could not find any heavy atoms in the ligand. distance_c constraint will be ignored.")
        ligand_heavy_atom_indices = list(range(ligand_mol.GetNumAtoms()))
    
    if method == 'mc':
        logger.info("Using Monte Carlo (MC) sampling method.")
        results = monte_carlo_ligand_sampling(
            ligand_mol=ligand_mol,
            protein_structures=protein_structures,
            offset_a=offset_a,
            distance_a=distance_a,
            distance_b=distance_b,
            distance_c=distance_c,
            ligand_subset_indices=ligand_heavy_atom_indices,
            num_conformers_to_generate=num_conformers_to_generate,
            n_samples=n_samples,
            z_samples=z_samples,
            max_trials=max_trials,
            logger=logger)
    elif method == 'mcmc':
        logger.info("Using Markov Chain Monte Carlo (MCMC) sampling method.")
        results = mcmc_ligand_sampling(
            ligand_mol=ligand_mol,
            protein_structures=protein_structures,
            offset_a=offset_a,
            distance_a=distance_a,
            distance_b=distance_b,
            distance_c=distance_c,
            ligand_subset_indices=ligand_heavy_atom_indices,
            num_conformers_to_generate=num_conformers_to_generate,
            n_samples=n_samples,
            z_samples=z_samples,
            max_trials=max_trials,
            logger=logger)
    else:
        raise ValueError(f"Unknown sampling method: {method}. Choose 'mc' or 'mcmc'.")
    
    ligand_chain = ligand_structure.chain_id[0]
    ligand_resid = ligand_structure.res_id[0]
    ligand_resname = ligand_structure.res_name[0]
    
    for res_idx, result in enumerate(results):
        new_ligand = rdk.from_mol(result[0])
        new_ligand.chain_id = np.array([ligand_chain] * len(new_ligand[0]))
        new_ligand.res_id = np.array([ligand_resid] * len(new_ligand[0]))
        new_ligand.res_name = np.array([ligand_resname] * len(new_ligand[0]))
        new_ligand.hetero = np.array([True] * len(new_ligand[0]))
        new_ligand.atom_name = new_ligand.name

        new_complex = struc.concatenate([result[1], new_ligand[0]])

        new_file = pdbx.CIFFile()
        pdbx.set_structure(new_file, new_complex)

        output_path = os.path.join(misato_complex_output_dir, f"{misato_id}_unbound_{str(res_idx)}.cif.gz")

        with gzip.open(output_path, 'wt') as f:
            new_file.write(f)


    return results



