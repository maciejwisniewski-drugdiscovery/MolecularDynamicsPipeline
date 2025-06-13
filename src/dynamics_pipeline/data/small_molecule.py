import os
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from openff.toolkit.topology import Molecule
from openbabel import pybel
import logging
from pathlib import Path
from typing import Union, List
import numpy as np
import biotite
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb
from openbabel import pybel
from dynamics_pipeline.utils.logger import setup_logger, log_info, log_error, log_warning, log_debug
import biotite.structure as struc

logger = setup_logger(name="plinder_dynamics", log_level=logging.INFO)

def load_molecule_to_openmm(input_filepath: str, input_format: str):
    '''
    There are 6 methods to read Ligands. We will try all of them.
    List of methods:
        * Mol2 File -> OpenBabel -> Mol2 Block -> RDKit Molecule -> OpenFF-Toolkit Molecule
        * SDF File -> OpenBabel -> Mol2 Block -> RDKit Molecule -> OpenFF-Toolkit Molecule
        * Mol2 File -> RDKit Molecule -> OpenFF-Toolkit Molecule
        * SDF File -> RDKit Molecule -> OpenFF-Toolkit Molecule
        * SDF File -> OpenFF-Toolkit Molecule
    :return: OpenFF-Toolkit Molecule
    '''
    assert os.path.exists(input_filepath), 'Ligand file doesn\'t exist.'
    if input_format == '.sdf':
        try:
            openff_molecule = Molecule.from_file(input_filepath)
            return openff_molecule
        except Exception as e:
            log_warning(logger, f"Failed to load SDF file: {input_filepath} with error: {e}")
        try:
            suppl = Chem.ForwardSDMolSupplier(input_filepath,sanitize=False,removeHs=True)
            rdkit_molecule = next(suppl)
            for atom in rdkit_molecule.GetAtoms():
                atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
            openff_molecule = Molecule.from_rdkit(rdkit_molecule, allow_undefined_stereo=True)
            return openff_molecule
        except Exception as e:
            log_warning(logger, f"Failed to load SDF file: {input_filepath} with error: {e}")
        try:
            # Load SDF File to PyBel
            pybel_molecule = next(pybel.readfile('sdf', input_filepath))
            pybel_molecule.removeh()
            pybel_molecule.addh()
            pybel_molecule.removeh()
            # Convert to Mol2 Block
            mol2_block = pybel_molecule.write('mol2')
            # Convert to RDKit Molecule
            rdkit_molecule = Chem.MolFromMol2Block(mol2_block,sanitize=False,removeHs=True)
            # Convert to OpenFF-Toolkit Molecule
            openff_molecule = Molecule.from_rdkit(rdkit_molecule,allow_undefined_stereo=True)
            return openff_molecule
        except Exception as e:
            log_warning(logger, f"Failed to load SDF file: {input_filepath} with error: {e}")
            raise ValueError(f"Failed to load SDF file: {input_filepath} with error: {e}")
    if input_format == '.mol2':
        try:
            openff_molecule = Molecule.from_file(input_filepath)
            return openff_molecule
        except:
            log_warning(logger, f"Failed to load MOL2 file: {input_filepath}")
        try:
            rdkit_molecule = Chem.MolFromMol2File(input_filepath,sanitize=False,removeHs=True)
            openff_molecule = Molecule.from_rdkit(rdkit_molecule, allow_undefined_stereo=True)
            return openff_molecule
        except:
            log_warning(logger, f"Failed to load MOL2 file: {input_filepath}")
        try:
            pybel_molecule = next(pybel.readfile('mol2', input_filepath))
            pybel_molecule.removeh()
            mol2_block = pybel_molecule.write('mol2')
            rdkit_molecule = Chem.MolFromMol2Block(mol2_block,sanitize=False,removeHs=True)
            openff_molecule = Molecule.from_rdkit(rdkit_molecule, allow_undefined_stereo=True)
            return openff_molecule
        except:
            log_warning(logger, f"Failed to load MOL2 file: {input_filepath}")
            raise ValueError(f"Failed to load MOL2 file: {input_filepath}")
    return


def match_atoms_based_on_coords(ref_coords: list, coords: list):
    atom_map = []
    used_coords = []
    for ref_idx, ref_coord in enumerate(ref_coords):
        for idx, coord in enumerate(coords):
            if np.allclose(ref_coord, coord, atol=1e-3) and idx not in used_coords:
                atom_map.append((ref_idx,idx))
                used_coords.append(idx)
                break
    assert len(atom_map) == len(ref_coords), "Failed to match all atoms"
    return atom_map


def fix_autodock_output_ligand(reference_sdf_filepath: str, reference_pdbqt_filepath: str, docked_sdf_filepath: str, output_sdf_filepath: str, sanitize: bool = True):
    '''
    This function is used to fix the autodock output ligand.
    '''
    # Load Reference Molecule
    ref_sdf_mol = next(pybel.readfile('sdf', reference_sdf_filepath))
    ref_sdf_mol.removeh()
    ref_sdf_mol = Chem.MolFromMol2Block(ref_sdf_mol.write('mol2'), sanitize=False, removeHs=True)
    if ref_sdf_mol is None:
        log_warning(logger, f"Failed to load reference SDF file: {reference_sdf_filepath}")
        raise ValueError(f"Failed to load reference SDF file: {reference_sdf_filepath}")
    
    ref_pdbqt_mol = next(pybel.readfile('pdbqt', reference_pdbqt_filepath))
    ref_pdbqt_mol.removeh()
    ref_pdbqt_mol = Chem.MolFromMol2Block(ref_pdbqt_mol.write('mol2'), sanitize=False, removeHs=True)
    if ref_pdbqt_mol is None:
        log_warning(logger, f"Failed to load reference PDBQT file: {reference_pdbqt_filepath}")
        raise ValueError(f"Failed to load reference PDBQT file: {reference_pdbqt_filepath}")
    
    # Load Docked Molecule
    docked_sdf_mol = next(pybel.readfile('sdf', docked_sdf_filepath))
    docked_sdf_mol.removeh()
    docked_sdf_mol = Chem.MolFromMol2Block(docked_sdf_mol.write('mol2'), sanitize=False, removeHs=True)
    if docked_sdf_mol is None:
        log_warning(logger, f"Failed to load docked SDF file: {docked_sdf_filepath}")
        raise ValueError(f"Failed to load docked SDF file: {docked_sdf_filepath}")
    
    # Read Molecule and Conformation
    ref_sdf_mol = neutralizeRadicals(ref_sdf_mol)
    ref_sdf_conf = ref_sdf_mol.GetConformer()
    ref_sdf_coords = [(round(ref_sdf_conf.GetAtomPosition(i).x,3),
                        round(ref_sdf_conf.GetAtomPosition(i).y,3),
                        round(ref_sdf_conf.GetAtomPosition(i).z,3)) for i in range(ref_sdf_conf.GetNumAtoms())]
    
    ref_pdbqt_mol = neutralizeRadicals(ref_pdbqt_mol)
    ref_pdbqt_conf = ref_pdbqt_mol.GetConformer()
    ref_pdbqt_coords = [(round(ref_pdbqt_conf.GetAtomPosition(i).x,3),
                        round(ref_pdbqt_conf.GetAtomPosition(i).y,3),
                        round(ref_pdbqt_conf.GetAtomPosition(i).z,3)) for i in range(ref_pdbqt_conf.GetNumAtoms())]
    
    # The main problem is that the coords from pdbqt are bad and indices destroy everything.
    # So to map SDF outputs i have to make this strange alignment.
    atom_map = match_atoms_based_on_coords(ref_sdf_coords, ref_pdbqt_coords)

    docked_sdf_mol = neutralizeRadicals(docked_sdf_mol)
    docked_sdf_conformer = next(docked_sdf_mol.GetConformers())
    docked_sdf_coords = [(round(docked_sdf_conformer.GetAtomPosition(i).x,3),
                      round(docked_sdf_conformer.GetAtomPosition(i).y,3),
                      round(docked_sdf_conformer.GetAtomPosition(i).z,3)) for i in range(docked_sdf_conformer.GetNumAtoms())]

    docked_atom_indices = []
    for idx, atom in enumerate(docked_sdf_mol.GetAtoms()):
        if atom.GetSymbol() != 'H':
            docked_atom_indices.append(idx)

    filtered_docked_sdf_coords = [coord for idx, coord in enumerate(docked_sdf_coords) if idx in docked_atom_indices]
    fixed_filtered_docked_sdf_coords = [filtered_docked_sdf_coords[old_idx] for new_idx, old_idx in sorted(atom_map)]

    assert docked_sdf_conformer.Is3D(), "Docked Molecule 'mol' does not have 3D conformer"

    conf = Chem.Conformer(ref_sdf_mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(fixed_filtered_docked_sdf_coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    
    
    ref_sdf_mol.AddConformer(conf, assignId=True)
    # ADD CONFORMER AND SAVE

    return ref_sdf_mol


def fix_molecule_with_pybel(input_filepath: str, output_dir: str, overwrite: bool = False):
    output_name = os.path.splitext(os.path.basename(input_filepath))[0]
    output_path = os.path.join(output_dir, f"{output_name}.sdf")
    if os.path.exists(output_path) and not overwrite:
        return output_path
    pybel_molecule = next(pybel.readfile('sdf', input_filepath))
    pybel_molecule.removeh()
    pybel_molecule.addh()
    pybel_molecule.removeh()
    pybel_molecule.write('sdf', output_path)
    return output_path

def make_unbound(
    mol: Mol, 
    ref_structure: Union[struc.AtomArray, struc.AtomArrayStack, Path], 
    offset: float = 8.0,
    max_dist: float = 30.0,
    clash_threshold: float = 3.0,
    max_retries: int = 50,
    max_dist_between_ligands: float = 10.0
) -> Union[Mol, None]:
    """
    Generates an unbound conformation of a ligand with respect to a reference protein structure.

    This function takes a ligand and a reference protein structure, and places the ligand in a random, clash-free
    position near the protein. It ensures the ligand is not too close to cause steric clashes but not excessively
    far, making it suitable for starting molecular dynamics simulations of binding.

    Parameters:
    - mol (Mol): RDKit Mol object for the ligand.
    - ref_structure (Union[struc.AtomArray, struc.AtomArrayStack, Path]): The reference protein structure.
      Can be a Biotite AtomArray, AtomArrayStack, or a file Path to a PDB/CIF file.
    - offset (float, optional): Additional distance to add between protein and ligand. Defaults to 5.0 Å.
    - max_dist (float, optional): Maximum allowed distance for ligand placement. Defaults to 20.0 Å.
    - clash_threshold (float, optional): The minimum distance between any ligand and protein atom.
      Distances below this are considered clashes. Defaults to 2.0 Å.
    - max_retries (int, optional): Number of attempts to find a clash-free position. Defaults to 50.

    Returns:
    - Union[Mol, None]: The RDKit Mol with a new 3D conformer in an unbound position,
      or None if a suitable position could not be found.
    """
    if isinstance(ref_structure, Path):
        if str(ref_structure).endswith('.cif'):
            ref_structure = pdbx.CIFFile.read(ref_structure).get_structure(model=1)
        elif str(ref_structure).endswith('.pdb'):
            ref_structure = pdb.PDBFile.read(ref_structure).get_structure(model=1)
        else:
            raise ValueError(f"Unsupported file format: {ref_structure}")
    elif isinstance(ref_structure, struc.AtomArrayStack):
        ref_structure = ref_structure[0]
    
    protein_coords = ref_structure.coord
    protein_center = struc.mass_center(ref_structure)
    radius_of_gyration = struc.gyration_radius(ref_structure)

    mol.RemoveAllConformers()
    mol_with_hs = Chem.AddHs(mol, addCoords=True)
    if AllChem.EmbedMolecule(mol_with_hs, randomSeed=-1) == -1:
        log_warning(logger, "Failed to generate conformer, attempting with random coordinates.")
        try:
            AllChem.EmbedMolecule(mol_with_hs, useRandomCoords=True)
        except Exception:
            log_warning(logger, "Failed to generate 3D conformer for the ligand.")
            return None
    AllChem.MMFFOptimizeMolecule(mol_with_hs)
    
    ligand_conf = mol_with_hs.GetConformer()
    ligand_coords = ligand_conf.GetPositions()
    ligand_centroid = ligand_coords.mean(axis=0)
    ligand_radius = np.max(np.linalg.norm(ligand_coords - ligand_centroid, axis=1))

    min_placement_dist = radius_of_gyration + ligand_radius + offset

    for attempt in range(max_retries):
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)

        placement_dist = np.random.uniform(min_placement_dist, max_dist)
        
        new_ligand_centroid = protein_center + direction * placement_dist
        translation_vector = new_ligand_centroid - ligand_centroid
        
        new_ligand_coords = ligand_coords + translation_vector

        clash = False
        for lig_atom_coord in new_ligand_coords:
            distances = np.linalg.norm(protein_coords - lig_atom_coord, axis=1)
            if np.min(distances) < clash_threshold:
                clash = True
                break
        
        if not clash:
            for i in range(mol_with_hs.GetNumAtoms()):
                x, y, z = new_ligand_coords[i]
                ligand_conf.SetAtomPosition(i, Point3D(x, y, z))
            
            log_info(logger, f"Placed ligand at {placement_dist:.2f} Å from protein center (attempt {attempt + 1}).")
            return mol_with_hs

    log_warning(logger, f"Failed to find a clash-free position after {max_retries} attempts.")
    return None

def create_unbound_ligand_files(ligand_filepaths: List[str],
                                ref_protein: Union[struc.AtomArray, struc.AtomArrayStack, Path],
                                output_dir: str,
                                min_distance: float = 10.0,
                                max_placement_attempts: int = 50,
                                max_retries: int = 50) -> List[str]:
    """
    Create unbound conformations for multiple ligands, ensuring they are properly spaced.
    
    Parameters:
    -----------
    ligand_filepaths : List[str]
        List of paths to ligand files (supported formats: .sdf, .mol2)
    ref_protein : Union[struc.AtomArray, struc.AtomArrayStack, Path]
        Reference protein structure to place ligands around
    output_dir : str
        Directory to save the generated conformations
    
    Returns:
    --------
    List[str]
        List of paths to the saved unbound ligand conformations
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_conformations = []
    placed_ligand_coords = []  # List to store centroids of placed ligands
    
    for ligand_path in ligand_filepaths:
        file_format = os.path.splitext(ligand_path)[1].lower()
        if file_format not in ['.sdf', '.mol2']:
            log_warning(logger, f"Unsupported file format for {ligand_path}. Skipping.")
            continue
            
        try:
            ligand_supplier = Chem.SDMolSupplier(ligand_path)
            ligand_mol = ligand_supplier[0]
            if ligand_mol is None:
                log_warning(logger, f"Failed to load ligand from {ligand_path}")
                continue
                            
            # Try to generate unbound conformation up to 10 times
            placed_successfully = False
            
            for attempt in range(max_placement_attempts):
                # Generate unbound conformation
                unbound_mol = make_unbound(ligand_mol, ref_protein, max_retries=max_retries)
                if unbound_mol is None:
                    continue
                    
                # Get centroid of new ligand
                conf = unbound_mol.GetConformer()
                coords = conf.GetPositions()
                centroid = coords.mean(axis=0)
                
                # Check distance to all previously placed ligands
                too_close = False
                for placed_centroid in placed_ligand_coords:
                    distance = np.linalg.norm(centroid - placed_centroid)
                    if distance < max_dist_between_ligands:  # 10 Å minimum distance
                        too_close = True
                        break
                
                if not too_close:
                    placed_successfully = True
                    placed_ligand_coords.append(centroid)
                    
                    # Save the conformation
                    output_name = os.path.splitext(os.path.basename(ligand_path))[0]
                    output_path = os.path.join(output_dir, f"{output_name}_unbound{file_format}")
                    
                    
                    writer = Chem.SDWriter(output_path)
                    writer.write(unbound_mol)
                    writer.close()
                    
                    saved_conformations.append(output_path)
                    log_info(logger, f"Successfully placed and saved unbound conformation for {ligand_path}")
                    break
            
            if not placed_successfully:
                log_warning(logger, f"Failed to place {ligand_path} after {max_placement_attempts} attempts")
                
        except Exception as e:
            log_warning(logger, f"Error processing {ligand_path}: {str(e)}")
            continue
    
    return saved_conformations