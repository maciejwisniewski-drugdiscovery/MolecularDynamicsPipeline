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