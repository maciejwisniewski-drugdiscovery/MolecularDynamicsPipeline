import os
from tqdm import tqdm
from pathlib import Path
import pickle
import json
import pandas as pd

from typing import Optional, List

import itertools
import numpy as np
import biotite.structure as struc
import biotite.structure.io as bsio
import biotite.structure.io.xtc as xtc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from biotite.interface import openmm as biotite_openmm

from openmm import *
from openmm.app import *
from openmm.unit import *
from openmm import XmlSerializer

from openmm import LangevinIntegrator, Context, Platform
from openmm import Platform, XmlSerializer, LangevinIntegrator, NonbondedForce, CustomExternalForce, MonteCarloBarostat


def create_subset_topology(topology: Topology, atom_indices: List[int]) -> Topology:
    """
    Create a new OpenMM Topology object containing only a subset of atoms.

    Parameters
    ----------
    topology : openmm.app.Topology
        The original topology.
    atom_indices : list of int
        A list of atom indices to include in the new topology.

    Returns
    -------
    openmm.app.Topology
        A new topology containing only the specified atoms, their residues,
        chains, and the bonds between them.
    """
    new_top = Topology()
    old_to_new_atoms = {}
    indices_set = set(atom_indices)
    
    # Create new chains, residues, and atoms
    for chain in topology.chains():
        new_chain = new_top.addChain(chain.id)
        for residue in chain.residues():
            residue_atoms = list(residue.atoms())
            # Check if any atom from this residue is in our subset
            if any(atom.index in indices_set for atom in residue_atoms):
                new_res = new_top.addResidue(residue.name, new_chain, residue.id)
                # Add atoms that are in our subset
                for atom in residue_atoms:
                    if atom.index in indices_set:
                        new_atom = new_top.addAtom(atom.name, atom.element, new_res, atom.id)
                        old_to_new_atoms[atom] = new_atom
    
    # Create new bonds between atoms that are both in the subset
    for bond in topology.bonds():
        atom1, atom2 = bond
        if atom1 in old_to_new_atoms and atom2 in old_to_new_atoms:
            new_top.addBond(old_to_new_atoms[atom1], old_to_new_atoms[atom2])
    
    return new_top


def split_molecular_dynamics_into_components(topology_filepath: Path,
    trajectory_filepath: Path,
    output_dir: Path,
    remove_waters: bool = True,
    renamed_chains: Optional[List[str]] = None):
    """
    Split a molecular dynamics trajectory into components using OpenMM for topology
    manipulation and Biotite for coordinate handling.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    single_output_dir = output_dir / 'single'
    single_output_dir.mkdir(parents=True, exist_ok=True)
    pair_output_dir = output_dir / 'pair'
    pair_output_dir.mkdir(parents=True, exist_ok=True)

    # Load topology and positions with OpenMM
    openmm_cif = PDBxFile(str(topology_filepath))
    full_openmm_topology = openmm_cif.getTopology()
    full_openmm_positions = openmm_cif.getPositions(asNumpy=True)

    # Load structure and trajectory with Biotite for masking and coordinate access
    biotite_cif = pdbx.CIFFile.read(topology_filepath)
    biotite_structure = pdbx.get_structure(biotite_cif, model=1, include_bonds=True)
    biotite_trajectory = xtc.XTCFile.read(trajectory_filepath)
    all_coords = biotite_trajectory.get_coord()

    solute_biotite_structure = biotite_structure
    solute_openmm_topology = full_openmm_topology
    solute_openmm_positions = full_openmm_positions
    solute_coords = all_coords

    if remove_waters:
        water_mask = (biotite_structure.res_name == 'HOH')
        ion_mask = np.isin(biotite_structure.res_name, ['NA', 'CL', 'K']) # Common ions
        solvent_mask = water_mask | ion_mask
        solute_mask = ~solvent_mask
        
        solute_indices = np.where(solute_mask)[0].tolist()
        
        # Filter all data structures to only include solute atoms
        solute_biotite_structure = biotite_structure[solute_mask]
        solute_coords = all_coords[:, solute_mask, :]
        solute_openmm_positions = full_openmm_positions[solute_mask, :]
        solute_openmm_topology = create_subset_topology(full_openmm_topology, solute_indices)

    chain_ids = np.unique(solute_biotite_structure.chain_id)

    all_tqdm_bar = tqdm(total=len(chain_ids) + len(list(itertools.combinations(chain_ids, 2))),
                        desc='Splitting Trajectory')

    # Generate Single Component Topologies and Trajectories
    for chain_id in chain_ids:
        component_topology_filename = single_output_dir / f'{chain_id}.cif'
        component_trajectory_filename = single_output_dir / f'{chain_id}.xtc'

        chain_mask = (solute_biotite_structure.chain_id == chain_id)
        component_indices = np.where(chain_mask)[0].tolist()
        
        component_topology = create_subset_topology(solute_openmm_topology, component_indices)
        component_positions = solute_openmm_positions[chain_mask]
        component_coords = solute_coords[:, chain_mask, :]

        with open(str(component_topology_filename), 'w') as f:
            PDBxFile.writeFile(component_topology, component_positions, f, keepIds=True)

        component_xtc = xtc.XTCFile()
        component_xtc.set_coord(component_coords)
        component_xtc.write(str(component_trajectory_filename))

        all_tqdm_bar.update(1)

    # Generate Paired Component Topologies and Trajectories
    for chain_id_a, chain_id_b in itertools.combinations(chain_ids, 2):
        component_topology_filename = pair_output_dir / f'{chain_id_a}_{chain_id_b}.cif'
        component_trajectory_filename = pair_output_dir / f'{chain_id_a}_{chain_id_b}.xtc'
        
        chain_a_mask = (solute_biotite_structure.chain_id == chain_id_a)
        chain_b_mask = (solute_biotite_structure.chain_id == chain_id_b)
        component_mask = chain_a_mask | chain_b_mask
        component_indices = np.where(component_mask)[0].tolist()

        component_topology = create_subset_topology(solute_openmm_topology, component_indices)
        component_positions = solute_openmm_positions[component_mask]
        component_coords = solute_coords[:, component_mask, :]

        with open(str(component_topology_filename), 'w') as f:
            PDBxFile.writeFile(component_topology, component_positions, f, keepIds=True)

        component_xtc = xtc.XTCFile()
        component_xtc.set_coord(component_coords)
        component_xtc.write(str(component_trajectory_filename))

        all_tqdm_bar.update(1)

    all_tqdm_bar.close()


def calculate_component_energy(component_topology_filepath: Path, 
                                component_trajectory_filepath: Path,
                                forcefield_dirpath: Path) -> List[float]:
    """
    Calculate the potential energy for each frame of a component's trajectory.
    """
    component_name = component_topology_filepath.stem
    
    component_topology_file = pdbx.CIFFile.read(component_topology_filepath)
    component_topology_coords = pdbx.get_structure(component_topology_file, model=1, include_bonds=True).coord

    component_topology = PDBxFile(str(component_topology_filepath)).topology
    coords = xtc.XTCFile.read(component_trajectory_filepath).get_coord()

    all_coords = np.concatenate([component_topology_coords[None, ...], coords], axis=0)

    forcefield = ForceField(
        "amber/protein.ff14SB.xml",
        "amber/tip3p_standard.xml",
        "amber/tip3p_HFE_multivalent.xml",
    )
    for forcefield_file in forcefield_dirpath.glob('*.xml'):
        forcefield.loadFile(str(forcefield_file))
    
    # For vacuum calculations, NoCutoff is appropriate. #self.system = XmlSerializer.deserialize(f.read())
    system = forcefield.createSystem(component_topology, nonbondedMethod=NoCutoff)
    # Integrator settings don't matter much as we are not running dynamics.
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    platform = Platform.getPlatformByName('CUDA')
    simulation = Simulation(component_topology, system, integrator, platform)

    energies = []
    for frame in tqdm(all_coords, desc=f'Processing {component_name} energies', leave=False):
        simulation.context.setPositions(frame)
        state = simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        energies.append(energy)

    return energies

def calculate_interaction_energies(
    topology_filepath: Path,
    trajectory_filepath: Path,
    forcefield_dirpath: Path,
    output_dir: Path,
    remove_waters: bool = True
):
    """
    Orchestrates splitting a trajectory, calculating energies for all components,
    and computing an MMPBSA-like interaction matrix.
    """
    # 1. Split trajectory into components
    split_dir = output_dir / 'components'
    split_molecular_dynamics_into_components(
        topology_filepath=topology_filepath,
        trajectory_filepath=trajectory_filepath,
        output_dir=split_dir,
        remove_waters=remove_waters
    )

    # 2. Calculate energies for all components
    all_energies = {}
    
    single_component_dir = split_dir / 'single'
    pair_component_dir = split_dir / 'pair'

    single_topologies = sorted(list(single_component_dir.glob('*.cif')))
    pair_topologies = sorted(list(pair_component_dir.glob('*.cif')))

    all_topologies = single_topologies + pair_topologies
    
    print("Calculating energies for all components...")
    for top_file in tqdm(all_topologies, desc="Overall Progress"):
        component_name = top_file.stem
        traj_file = top_file.with_suffix('.xtc')
        
        energies = calculate_component_energy(
            component_topology_filepath=top_file,
            component_trajectory_filepath=traj_file,
            forcefield_dirpath=forcefield_dirpath
        )
        all_energies[component_name] = energies

    # 3. Create JSON output
    json_output_list = []
    for name, energy_list in all_energies.items():
        components = name.split('_')
        json_output_list.append({
            'components': components,
            'energies': energy_list
        })
    
    json_filepath = output_dir / 'component_energies.json'
    print(f"Saving component energies to {json_filepath}")
    with open(json_filepath, 'w') as f:
        json.dump(json_output_list, f, indent=2)

    # 4. Create and save the N x N x Frames interaction energy matrix.
    single_component_names = [p.stem for p in single_topologies]
    if not single_component_names:
        print("No single components found. Skipping matrix generation.")
        print("Energy calculation finished.")
        return

    N = len(single_component_names)
    # Get number of frames from the first component's energy list
    F = len(next(iter(all_energies.values())))

    # Create a mapping from component name to matrix index
    component_to_idx = {name: i for i, name in enumerate(single_component_names)}

    # Initialize the 3D matrix for all-vs-all interaction energies
    interaction_matrix_3d = np.zeros((N, N, F), dtype=float)

    # Convert all energy lists to numpy arrays for vectorized operations
    energies_np = {name: np.array(energy_list) for name, energy_list in all_energies.items()}

    # Iterate over unique pairs of components to calculate interaction energies
    for r_name, c_name in itertools.combinations(single_component_names, 2):
        r_idx = component_to_idx[r_name]
        c_idx = component_to_idx[c_name]
        
        # Find the energy time-series for the pair complex
        pair_name_1 = f"{r_name}_{c_name}"
        pair_name_2 = f"{c_name}_{r_name}"
        
        if pair_name_1 in energies_np:
            pair_energy_ts = energies_np[pair_name_1]
        elif pair_name_2 in energies_np:
            pair_energy_ts = energies_np[pair_name_2]
        else:
            print(f"Warning: Pair energy time series for {r_name} and {c_name} not found. Skipping this pair.")
            continue
            
        # Calculate interaction energy for all frames at once (vectorized)
        interaction_energy_ts = pair_energy_ts - (energies_np[r_name] + energies_np[c_name])
        
        # Populate the symmetric matrix
        interaction_matrix_3d[r_idx, c_idx, :] = interaction_energy_ts
        interaction_matrix_3d[c_idx, r_idx, :] = interaction_energy_ts

    # Save the 3D matrix and component labels to a single .npz file
    matrix_filepath = output_dir / 'interaction_energy_matrix.npz'
    print(f"Saving {N}x{N}x{F} interaction energy matrix to {matrix_filepath}")
    np.savez(
        matrix_filepath,
        matrix=interaction_matrix_3d,
        components=np.array(single_component_names)
    )

    print("Energy calculation finished.")


if __name__ == '__main__':
    plinder_id = '2c27__1__1.A__1.C'
    output_dir = Path(f'/mnt/raid/mwisniewski/Data/plinder_md/plinder_bound/{plinder_id}_simulation_bound_state')

    calculate_interaction_energies(
        topology_filepath=output_dir / f'{plinder_id}_init_topology.cif',
        trajectory_filepath=output_dir / f'trajectories/{plinder_id}_warmup_trajectory.xtc',
        forcefield_dirpath = output_dir / f'forcefields',
        output_dir=output_dir / 'energies'
    )