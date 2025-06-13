from pdb2pqr.main import run_pdb2pqr
import os

def fix_biomolecule_with_pdb2pqr(receptor_pdb_filepath: str, output_pdb_filepath: str, ph: float = 7.4):
    output_pqr_filepath = output_pdb_filepath.replace('.pdb', '.pqr')
    args = [
        '--ff=AMBER',
        '--with-ph='+str(ph),
        '--pdb-output='+output_pdb_filepath,
        receptor_pdb_filepath,
        output_pqr_filepath
    ]
    run_pdb2pqr(args)
    if os.path.exists(output_pdb_filepath):
        return output_pdb_filepath
    else:
        raise FileNotFoundError(f"Failed to fix biomolecule {receptor_pdb_filepath}")