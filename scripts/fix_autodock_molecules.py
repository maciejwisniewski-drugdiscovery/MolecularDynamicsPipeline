import os
from rdkit import Chem
from rdkit.Chem import Mol
from openbabel import pybel
from rdkit.Chem import Draw
from molecular_dynamics_pipeline.data.small_molecule import fix_autodock_output_ligand

if __name__ == "__main__":
    REFERENCE_SDF_FILEPATH = "/mnt/raid/mwisniewski/data/WUM/ACmodeling/ligands/1M17__10__dock.sdf"
    SMILES=""
    

    if not os.path.exists(REFERENCE_SDF_FILEPATH) and SMILES is None:
        raise ValueError("Reference SDF file doesn't exist and SMILES is not provided")
    elif not os.path.exists(REFERENCE_SDF_FILEPATH) and SMILES:
        ref_mol = pybel.readstring('smi', SMILES)
        ref_mol.make3D()
        ref_mol.localopt(forcefield="mmff94", steps=500)
        ref_mol.removeh()
        ref_mol.write('sdf', REFERENCE_SDF_FILEPATH)
        ref_mol.write('pdbqt', REFERENCE_PDBQT_FILEPATH)
    a=1
    fixed_mol = fix_autodock_output_ligand(
        reference_sdf_filepath=REFERENCE_SDF_FILEPATH,
        reference_pdbqt_filepath=REFERENCE_PDBQT_FILEPATH,
        docked_sdf_filepath=DOCKED_SDF_FILEPATH,
        output_sdf_filepath=OUTPUT_SDF_FILEPATH
    )
    with Chem.SDWriter(OUTPUT_SDF_FILEPATH) as w:
        w.write(fixed_mol)
    fixed_mol.RemoveAllConformers()
    img = Draw.MolToImage(fixed_mol, size=(400, 300), kekulize=True)
    img.save('molecule.png')