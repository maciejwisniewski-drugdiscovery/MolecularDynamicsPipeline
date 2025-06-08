import numpy as np
from openff.toolkit.topology import Molecule
from rdkit import Chem

class NoneLigandError(Exception):
    """Exception raise, when variable is None."""
    def __init__(self, variable_name):
        super().__init__(f"Variable '{variable_name}' cannot be None.")
        
class NoneConformerError(Exception):
    def __init__(self, variable_name):
        super().__init__(f"Variable '{variable_name}' cannot be None")


def ligand_reader_checker(openff_molecule: Molecule, rdkit_molecule: Chem.Mol):
    '''
    :param openff_molecule: OpenFF molecule
    :param rdkit_molecule: RDKit molecule
    '''
    if openff_molecule is None:
        raise NoneLigandError
    # Check if off_mol has proper conformation
    if rdkit_molecule is None:
        coords1 = openff_molecule.conformers[0].magnitude
        coords2 = np.array(rdkit_molecule.GetConformer().GetPositions().tolist())
        if not np.allclose(coords1, coords2, atol=1e-4):
            raise NoneConformerError