import os
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from openff.toolkit.topology import Molecule
from openbabel import pybel
import logging
from pathlib import Path
from typing import Union
import numpy as np
import biotite
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb
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
            log_error(logger, f"Failed to load SDF file: {input_filepath} with error: {e}")
        try:
            suppl = Chem.ForwardSDMolSupplier(input_filepath,sanitize=False,removeHs=True)
            rdkit_molecule = next(suppl)
            for atom in rdkit_molecule.GetAtoms():
                atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
            openff_molecule = Molecule.from_rdkit(rdkit_molecule, allow_undefined_stereo=True)
            return openff_molecule
        except Exception as e:
            log_error(logger, f"Failed to load SDF file: {input_filepath} with error: {e}")
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
            log_error(logger, f"Failed to load MOL2 file: {input_filepath}")
        try:
            rdkit_molecule = Chem.MolFromMol2File(input_filepath,sanitize=False,removeHs=True)
            openff_molecule = Molecule.from_rdkit(rdkit_molecule, allow_undefined_stereo=True)
            return openff_molecule
        except:
            log_error(logger, f"Failed to load MOL2 file: {input_filepath}")
        try:
            pybel_molecule = next(pybel.readfile('mol2', input_filepath))
            pybel_molecule.removeh()
            mol2_block = pybel_molecule.write('mol2')
            rdkit_molecule = Chem.MolFromMol2Block(mol2_block,sanitize=False,removeHs=True)
            openff_molecule = Molecule.from_rdkit(rdkit_molecule, allow_undefined_stereo=True)
            return openff_molecule
        except:
            log_error(logger, f"Failed to load MOL2 file: {input_filepath}")
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


def neutralizeRadicals(mol):
    Chem.AddHs(mol)
    editable_mol = Chem.RWMol(mol)
    for atom in editable_mol.GetAtoms():
        atom.SetNumRadicalElectrons(0)
        atom.SetFormalCharge(0)
    mol = editable_mol.GetMol()
    #Chem.RemoveHs(mol)
    return mol


def fix_autodock_output_ligand(reference_sdf_filepath: str, reference_pdbqt_filepath: str, docked_sdf_filepath: str, output_sdf_filepath: str, sanitize: bool = True):
    '''
    This function is used to fix the autodock output ligand.
    '''
    # Load Reference Molecule
    ref_sdf_mol = next(pybel.readfile('sdf', reference_sdf_filepath))
    ref_sdf_mol.removeh()
    ref_sdf_mol = Chem.MolFromMol2Block(ref_sdf_mol.write('mol2'), sanitize=False, removeHs=True)
    if ref_sdf_mol is None:
        log_error(logger, f"Failed to load reference SDF file: {reference_sdf_filepath}")
        raise ValueError(f"Failed to load reference SDF file: {reference_sdf_filepath}")
    
    ref_pdbqt_mol = next(pybel.readfile('pdbqt', reference_pdbqt_filepath))
    ref_pdbqt_mol.removeh()
    ref_pdbqt_mol = Chem.MolFromMol2Block(ref_pdbqt_mol.write('mol2'), sanitize=False, removeHs=True)
    if ref_pdbqt_mol is None:
        log_error(logger, f"Failed to load reference PDBQT file: {reference_pdbqt_filepath}")
        raise ValueError(f"Failed to load reference PDBQT file: {reference_pdbqt_filepath}")
    
    # Load Docked Molecule
    docked_sdf_mol = next(pybel.readfile('sdf', docked_sdf_filepath))
    docked_sdf_mol.removeh()
    docked_sdf_mol = Chem.MolFromMol2Block(docked_sdf_mol.write('mol2'), sanitize=False, removeHs=True)
    if docked_sdf_mol is None:
        log_error(logger, f"Failed to load docked SDF file: {docked_sdf_filepath}")
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


def make_unbound(
    mol: Mol, 
    ref_structure: Union[struc.AtomArray, struc.AtomArrayStack, Path], 
    offset: float = 8.0,
    max_dist: float = 30.0,
    clash_threshold: float = 3.0,
    max_retries: int = 50
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
            log_error(logger, "Failed to generate 3D conformer for the ligand.")
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

    log_error(logger, f"Failed to find a clash-free position after {max_retries} attempts.")
    return None