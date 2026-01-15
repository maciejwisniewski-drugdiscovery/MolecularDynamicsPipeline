import os
from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster

from tqdm import tqdm

import biotite
import biotite.structure
import biotite.structure.io.xtc as xtc
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb


def load_topology(topology_filepath: Path) -> biotite.structure.AtomArray:
    """
    Load the topology from a file.
    Args:
        topology_filepath: The path to the topology file.
    
    Returns:
        The topology.
    """
    cif_file = pdbx.CIFFile.read(topology_filepath)
    return pdbx.get_structure(cif_file, model=1, include_bonds=True)

def get_reference_structure(reference_pdb_id: str, output_dirpath: Path) -> biotite.structure.AtomArray:
    """
    Get the reference structure from the RCSB PDB.
    Args:
        output_dirpath: The path to the output directory.
    
    Returns:
        The reference structure.
    """
    biotite.database.rcsb.fetch(
        pdb_ids = reference_pdb_id,
        format = 'cif',
        target_path = output_dirpath,
        overwrite = False,
    )
    output_filepath = output_dirpath / "2WXC.cif"
    return load_topology(output_filepath)

def load_trajectories(trajectory_filepaths: List[Path]) -> np.ndarray:
    """
    Load the trajectories from the given paths.
    Args:
        trajectory_filepaths: The paths to the trajectories.
    Returns:
        The trajectories.
    """
    trajectories = []
    for trajectory_filepath in trajectory_filepaths:
        xtc_file = xtc.XTCFile.read(str(trajectory_filepath))
        traj = xtc_file.get_coord()
        trajectories.append(traj)
    trajectories = np.concatenate(trajectories, axis=0)
    return trajectories

def calculate_fraction_of_native_contacts(
    reference_structure: biotite.structure.AtomArray,
    trajectory: NDArray[np.floating],
    cutoff: float = 8.0,
) -> NDArray[np.floating]:
    """
    Compute fraction of native contacts (Q / f_native) for a protein trajectory.
    Args:
        reference_structure: The reference structure.
        trajectory: The trajectory.
        cutoff: The cutoff distance for native contacts.
    Returns:
        The fraction of native contacts.
    """
    reference_xyz = reference_structure.coord
    N_atoms = reference_xyz.shape[0]

    assert trajectory.shape[1] == N_atoms, "Trajectory and reference atom count mismatch"

    # Typically res_id is unique per residue, chain-aware
    res_id = reference_structure.res_id

    # --- define native contacts ONCE ---
    D_ref = cdist(reference_xyz, reference_xyz)

    contact_mask = (D_ref < cutoff) & (D_ref > 0.0)

    # --- remove intra-residue contacts ---
    same_residue = res_id[:, None] == res_id[None, :]
    contact_mask &= ~same_residue

    # --- keep only upper triangle ---
    native_pairs = np.array(np.where(np.triu(contact_mask, k=1))).T
    n_native = len(native_pairs)

    if n_native == 0:
        return np.full(trajectory.shape[0], np.nan)

    # --- compute Q for each frame ---
    Q = np.zeros(trajectory.shape[0], dtype=float)

    for k, xyz in enumerate(trajectory):
        D = cdist(xyz, xyz)
        present = D[native_pairs[:, 0], native_pairs[:, 1]] < cutoff
        Q[k] = present.mean()

    return Q

def calculate_rmsd_matrix(trajectory: np.ndarray) -> np.ndarray:
    """
    Calculate the RMSD matrix for a given trajectory.
    Args:
        trajectory: The trajectory to calculate the RMSD matrix for.
    Returns:
        The RMSD matrix.
    """
    rmsd_matrix = np.zeros((trajectory.shape[0], trajectory.shape[0]))
    for i in tqdm(range(trajectory.shape[0]), total=trajectory.shape[0], desc="Computing RMSD matrix"):
        ith_rmsd = biotite.structure.rmsd(trajectory[i], trajectory)
        rmsd_matrix[i, :] = ith_rmsd
        rmsd_matrix[:, i] = ith_rmsd.T
    return rmsd_matrix

def select_representative_frames(
    rmsd_matrix: np.ndarray,
    fnat_mask: np.ndarray,
    n_representatives: int,
    linkage_method: str = "average",
) -> np.ndarray:
    """
    Cluster allowed (fnat_mask == True) structures using RMSD
    and select representative medoids.

    Returns indices in the ORIGINAL trajectory.
    """
    assert rmsd_matrix.shape[0] == rmsd_matrix.shape[1]
    assert rmsd_matrix.shape[0] == len(fnat_mask)

    # Restrict to allowed structures
    allowed_idx = np.where(fnat_mask)[0]
    if len(allowed_idx) == 0:
        raise ValueError("No structures allowed by fnat_mask")

    # Submatrix
    D = rmsd_matrix[np.ix_(allowed_idx, allowed_idx)]

    # Edge case: Fewer Structures than requested Representatives
    if len(allowed_idx) <= n_representatives:
        return allowed_idx

    # Hierarchical Clustering
    Z = linkage(D, method=linkage_method)
    labels = fcluster(Z, t=n_representatives, criterion="maxclust")

    representatives = []

    # Medoid per Cluster
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        # Intra-cluster RMSD
        D_cluster = D[np.ix_(cluster_indices, cluster_indices)]

        # Medoid = Minimal Mean Distance
        medoid_local = cluster_indices[np.argmin(D_cluster.mean(axis=1))]

        representatives.append(allowed_idx[medoid_local])

    return np.array(representatives, dtype=int)

def main():
    fastfolders_dirpath = Path("/mnt/raid/mwisniewski/data/molecular_dynamics/fastfolders")
    preproduction_dirpath = fastfolders_dirpath / "preproduction"
    topology_filepath = preproduction_dirpath / "fastfolder_preproduction_init_topology.cif"
    trajectories_dirpath = preproduction_dirpath / "trajectories"
    warmup_trajectory_filepath = trajectories_dirpath / "fastfolder_preproduction_warmup_trajectory.xtc"
    backbone_removal_trajectory_filepath = trajectories_dirpath / "fastfolder_preproduction_backbone_removal_trajectory.xtc"
    nvt_trajectory_filepath = trajectories_dirpath / "fastfolder_preproduction_nvt_trajectory.xtc"

    assert topology_filepath.exists(), f"Topology file {topology_filepath} does not exist"
    assert warmup_trajectory_filepath.exists(), f"Warmup trajectory file {warmup_trajectory_filepath} does not exist"
    assert backbone_removal_trajectory_filepath.exists(), f"Backbone removal trajectory file {backbone_removal_trajectory_filepath} does not exist"
    assert nvt_trajectory_filepath.exists(), f"NVT trajectory file {nvt_trajectory_filepath} does not exist"

    output_dirpath = fastfolders_dirpath / "prepare_to_production"
    rmsd_matrix_filepath = output_dirpath / "rmsd_matrix.npz"

    os.makedirs(output_dirpath, exist_ok=True)

    # Load Reference Structure
    reference_structure = get_reference_structure(reference_pdb_id='2WXC', output_dirpath=output_dirpath)
    reference_structure = reference_structure[reference_structure.hetero == False]
    reference_structure = reference_structure[reference_structure.element != 'H']

    simulation_structure = load_topology(topology_filepath)
    h_mask = simulation_structure.element != 'H'
    simulation_structure = simulation_structure[h_mask]

    trajectory = load_trajectories(
        trajectory_filepaths = [warmup_trajectory_filepath, backbone_removal_trajectory_filepath, nvt_trajectory_filepath],
    )
    trajectory = trajectory[:, h_mask, :]

    trajectory, _ = biotite.structure.superimpose(trajectory[0], trajectory)

    fnat = calculate_fraction_of_native_contacts(
        reference_structure=reference_structure,
        trajectory=trajectory,
    )

    fnat_mask = fnat < 0.75
    
    if rmsd_matrix_filepath.exists():
        rmsd_matrix = np.load(rmsd_matrix_filepath)['rmsd_matrix']
    else:
        rmsd_matrix = calculate_rmsd_matrix(trajectory)
        np.savez(rmsd_matrix_filepath, rmsd_matrix=rmsd_matrix)

    representative_frames = select_representative_frames(
        rmsd_matrix=rmsd_matrix,
        fnat_mask=fnat_mask,
        n_representatives=3000,
    )

    representative_trajectory = trajectory[representative_frames]
    representative_output_dirpath = output_dirpath / 'representatives'
    os.makedirs(representative_output_dirpath, exist_ok=True)
    for frame_index, frame in enumerate(representative_trajectory):
        frame_output_filepath = representative_output_dirpath / f"representative_frame_{frame_index}.cif"
        simulation_structure.coord = frame
        cif_file = pdbx.CIFFile()
        pdbx.set_structure(cif_file, simulation_structure, include_bonds=True)
        cif_file.write(frame_output_filepath)


if __name__ == "__main__":
    main()