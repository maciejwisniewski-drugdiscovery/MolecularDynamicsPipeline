import os
import numpy as np
import logging
from openmm import unit
from molecular_dynamics_pipeline.utils.logger import setup_logger, log_info, log_error, log_warning, log_debug

logger = setup_logger(name="plinder_dynamics", log_level=logging.INFO)

class ForceReporter(object):
    """
    A custom OpenMM reporter that saves atomic forces incrementally
    to a memory-mapped NumPy file (.npy) for efficiency.
    Supports resuming from checkpoints by loading existing force data.
    """
    def __init__(self, file, reportInterval, total_steps, atom_indices, dtype=np.float32):
        """
        Initializes the ForceReporter and creates or loads the memory-mapped file.

        Parameters
        ----------
        file : str
            The path to the output .npz file.
        reportInterval : int
            The interval (in number of steps) at which to report forces.
        total_steps : int
            The total number of steps in the simulation stage.
        atom_indices : list
            List of atom indices to save forces for.
        dtype : np.dtype
            The data type to use for storing forces.
        """
        # Store the final output path (.npz.gz)
        self._final_outpath = file
        # Create temporary .npy file path for memmap
        self._temp_outpath = file.replace('.npz', '.npy')
        self._reportInterval = reportInterval
        self._atom_indices = atom_indices
        self._n_atoms = len(atom_indices)
        self._dtype = dtype
        
        # Calculate the total number of frames that will be saved.
        self._total_frames = int(np.ceil(total_steps / self._reportInterval))
        
        if self._total_frames == 0 or self._n_atoms == 0:
            log_warning(logger, "ForceReporter initialized with 0 total frames or 0 atoms. No forces will be saved.")
            self._memmap = None
            self._report_counter = 0
            return

        # Shape of the final data will be (frames, atoms, 3)
        shape = (self._total_frames, self._n_atoms, 3)

        # Check if temporary file exists and try to load it
        if os.path.exists(self._temp_outpath):
            try:
                existing_memmap = np.lib.format.open_memmap(self._temp_outpath, dtype=self._dtype, mode='r')
                if existing_memmap.shape == shape:
                    # Find the last non-zero frame
                    non_zero_frames = np.any(existing_memmap != 0, axis=(1, 2))
                    last_frame = np.max(np.where(non_zero_frames)[0]) + 1 if np.any(non_zero_frames) else 0
                    
                    # Create new memmap and copy existing data
                    self._memmap = np.lib.format.open_memmap(self._temp_outpath, dtype=self._dtype, mode='r+', shape=shape)
                    self._report_counter = last_frame
                    log_info(logger, f"Resuming ForceReporter from frame {last_frame} of {self._total_frames}")
                    return
                else:
                    log_warning(logger, f"Existing force data has wrong shape. Expected {shape}, found {existing_memmap.shape}. Starting fresh.")
                    os.remove(self._temp_outpath)
            except Exception as e:
                log_warning(logger, f"Error loading existing force data: {str(e)}. Starting fresh.")
                if os.path.exists(self._temp_outpath):
                    os.remove(self._temp_outpath)

        # If we get here, we need to create a new memmap file
        self._memmap = np.lib.format.open_memmap(self._temp_outpath, dtype=self._dtype, mode='w+', shape=shape)
        self._report_counter = 0
        log_info(logger, f"ForceReporter initialized for {self._total_frames} frames on {self._n_atoms} atoms. Will save to {self._final_outpath}")

    def describeNextReport(self, simulation):
        # If no frames are to be saved, don't schedule any reports.
        if not hasattr(self, '_memmap') or self._memmap is None:
            return (0, False, False, False, False, None)
            
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        """
        This method is called by OpenMM at each report interval to write forces to disk.
        """
        if self._memmap is None or self._report_counter >= self._total_frames:
            return

        all_forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometers)
        forces = all_forces[self._atom_indices]
        
        # Write the data directly to the memory-mapped file on disk
        self._memmap[self._report_counter] = forces.astype(self._dtype)
        self._memmap.flush()  # Ensure data is written to disk at this step.
        
        self._report_counter += 1

        # If this is the last frame, compress the file
        if self._report_counter == self._total_frames:
            self._compress_and_cleanup()

    def _compress_and_cleanup(self):
        """
        Compresses the .npy file to .npz.gz format and removes the original file.
        """
        # Close the memmap file
        if hasattr(self, '_memmap') and self._memmap is not None:
            self._memmap.flush()
            del self._memmap

        try:
            # Load the data from the temporary .npy file
            data = np.load(self._temp_outpath)
            # Save as compressed npz
            np.savez_compressed(self._final_outpath, forces=data)
            # Remove the temporary .npy file
            os.remove(self._temp_outpath)
            log_info(logger, f"Successfully compressed forces data to {self._final_outpath}")
        except Exception as e:
            log_error(logger, f"Error compressing forces file: {str(e)}")
            # If compression fails, keep the original file
            if os.path.exists(self._temp_outpath):
                log_warning(logger, f"Keeping original uncompressed file at {self._temp_outpath}")


class HessianReporter(object):
    def __init__(self, file, reportInterval, total_steps, atom_indices, simulation, eps=1e-4, dtype=np.float32):
        """
        A custom OpenMM reporter that saves Hessian matrices incrementally
        to a memory-mapped NumPy file (.npy) for efficiency.
        Supports resuming from checkpoints by loading existing Hessian data.

        Parameters
        ----------
        file : str
            The path to the output .npz file.
        reportInterval : int
            The interval (in number of steps) at which to report Hessian.
        total_steps : int
            The total number of steps in the simulation stage.
        atom_indices : list
            List of atom indices to compute Hessian for.
        simulation : openmm.app.Simulation
            The OpenMM simulation object.
        eps : float
            The finite difference step size.
        dtype : np.dtype
            The data type to use for storing Hessian matrices.
        """
        self._outpath = file.replace('.npz', '.npy')
        self._final_outpath = file
        self._reportInterval = reportInterval
        self._atom_indices = atom_indices
        self._n_atoms = len(atom_indices)
        self._dtype = dtype
        self._eps = eps
        self._simulation = simulation

        self._dof_map = []
        for i in self._atom_indices:
            self._dof_map.extend([3 * i, 3 * i + 1, 3 * i + 2])

        self._total_frames = int(np.ceil(total_steps / self._reportInterval))
        if self._total_frames == 0 or self._n_atoms == 0:
            log_warning(logger, "HessianReporter initialized with 0 frames or 0 atoms. No hessian will be saved.")
            self._memmap = None
            self._report_counter = 0
            return

        shape = (self._total_frames, 3 * self._n_atoms, 3 * self._n_atoms)

        # Check if temporary file exists and try to load it
        if os.path.exists(self._outpath):
            try:
                existing_memmap = np.lib.format.open_memmap(self._outpath, dtype=self._dtype, mode='r')
                if existing_memmap.shape == shape:
                    # Find the last non-zero frame
                    non_zero_frames = np.any(existing_memmap != 0, axis=(1, 2))
                    last_frame = np.max(np.where(non_zero_frames)[0]) + 1 if np.any(non_zero_frames) else 0
                    
                    # Create new memmap and copy existing data
                    self._memmap = np.lib.format.open_memmap(self._outpath, dtype=self._dtype, mode='r+', shape=shape)
                    self._report_counter = last_frame
                    log_info(logger, f"Resuming HessianReporter from frame {last_frame} of {self._total_frames}")
                    return
                else:
                    log_warning(logger, f"Existing Hessian data has wrong shape. Expected {shape}, found {existing_memmap.shape}. Starting fresh.")
                    os.remove(self._outpath)
            except Exception as e:
                log_warning(logger, f"Error loading existing Hessian data: {str(e)}. Starting fresh.")
                if os.path.exists(self._outpath):
                    os.remove(self._outpath)

        # If we get here, we need to create a new memmap file
        self._memmap = np.lib.format.open_memmap(self._outpath, dtype=self._dtype, mode='w+', shape=shape)
        self._report_counter = 0
        log_info(logger, f"HessianReporter initialized for {self._total_frames} frames on {self._n_atoms} atoms. Will save to {self._final_outpath}")

    def describeNextReport(self, simulation):
        if not hasattr(self, '_memmap') or self._memmap is None:
            return (0, False, False, True, False, None)
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        if self._memmap is None or self._report_counter >= self._total_frames:
            return

        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        n_total_atoms = self._simulation.topology.getNumAtoms()
        d_subset = 3 * self._n_atoms
        x_flat = positions.reshape(-1)

        def get_forces(x_flat_perturbed):
            x_perturbed = x_flat_perturbed.reshape(n_total_atoms, 3)
            self._simulation.context.setPositions(x_perturbed * unit.nanometer)
            s = self._simulation.context.getState(getForces=True)
            F = s.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometers)
            return F.reshape(-1)

        H = np.zeros((d_subset, d_subset), dtype=self._dtype)
        for j in range(d_subset):
            dof_index_in_full_system = self._dof_map[j]
            x_plus = x_flat.copy()
            x_minus = x_flat.copy()
            x_plus[dof_index_in_full_system] += self._eps
            x_minus[dof_index_in_full_system] -= self._eps

            F_plus = get_forces(x_plus)
            F_minus = get_forces(x_minus)

            grad_F_col = (F_plus - F_minus) / (2 * self._eps)
            H[:, j] = -grad_F_col[self._dof_map]

        self._memmap[self._report_counter] = H
        self._memmap.flush()
        
        self._simulation.context.setPositions(positions)
        
        self._report_counter += 1

        if self._report_counter == self._total_frames:
            self._compress_and_cleanup()

    def _compress_and_cleanup(self):
        """
        Compresses the .npy file to .npz.gz format and removes the original file.
        """
        # Close the memmap file
        if hasattr(self, '_memmap') and self._memmap is not None:
            self._memmap.flush()
            del self._memmap

        try:
            data = np.load(self._outpath)
            np.savez_compressed(self._final_outpath, hessians=data)
            os.remove(self._outpath)
            log_info(logger, f"Successfully compressed Hessian data to {self._final_outpath}")
        except Exception as e:
            log_error(logger, f"[HessianReporter] Compression error: {e}")
            if os.path.exists(self._outpath):
                log_warning(logger, f"Keeping original uncompressed file at {self._outpath}")


class TrajectoryReporter(object):
    """
    A custom OpenMM reporter that saves atomic positions incrementally
    to a memory-mapped NumPy file (.npy) for efficiency, then compresses to NPZ.
    Supports resuming from checkpoints by loading existing trajectory data.
    """
    def __init__(self, file, reportInterval, total_steps, atom_indices, dtype=np.float32):
        """
        Initializes the TrajectoryReporter and creates or loads the memory-mapped file.

        Parameters
        ----------
        file : str
            The path to the output .npz file.
        reportInterval : int
            The interval (in number of steps) at which to report positions.
        total_steps : int
            The total number of steps in the simulation stage.
        atom_indices : list
            List of atom indices to save positions for.
        dtype : np.dtype
            The data type to use for storing positions.
        """
        # Store the final output path (.npz)
        self._final_outpath = file
        # Create temporary .npy file path for memmap
        self._temp_outpath = file.replace('.npz', '.npy')
        self._reportInterval = reportInterval
        self._atom_indices = atom_indices
        self._n_atoms = len(atom_indices)
        self._dtype = dtype
        
        # Calculate the total number of frames that will be saved.
        self._total_frames = int(np.ceil(total_steps / self._reportInterval))
        
        if self._total_frames == 0 or self._n_atoms == 0:
            log_warning(logger, "TrajectoryReporter initialized with 0 total frames or 0 atoms. No trajectory will be saved.")
            self._memmap = None
            self._report_counter = 0
            return

        # Shape of the final data will be (frames, atoms, 3)
        shape = (self._total_frames, self._n_atoms, 3)

        # Check if temporary file exists and try to load it
        if os.path.exists(self._temp_outpath):
            try:
                existing_memmap = np.lib.format.open_memmap(self._temp_outpath, dtype=self._dtype, mode='r')
                if existing_memmap.shape == shape:
                    # Find the last non-zero frame
                    non_zero_frames = np.any(existing_memmap != 0, axis=(1, 2))
                    last_frame = np.max(np.where(non_zero_frames)[0]) + 1 if np.any(non_zero_frames) else 0
                    
                    # Create new memmap and copy existing data
                    self._memmap = np.lib.format.open_memmap(self._temp_outpath, dtype=self._dtype, mode='r+', shape=shape)
                    self._report_counter = last_frame
                    log_info(logger, f"Resuming TrajectoryReporter from frame {last_frame} of {self._total_frames}")
                    return
                else:
                    log_warning(logger, f"Existing trajectory data has wrong shape. Expected {shape}, found {existing_memmap.shape}. Starting fresh.")
                    os.remove(self._temp_outpath)
            except Exception as e:
                log_warning(logger, f"Error loading existing trajectory data: {str(e)}. Starting fresh.")
                if os.path.exists(self._temp_outpath):
                    os.remove(self._temp_outpath)

        # If we get here, we need to create a new memmap file
        self._memmap = np.lib.format.open_memmap(self._temp_outpath, dtype=self._dtype, mode='w+', shape=shape)
        self._report_counter = 0
        log_info(logger, f"TrajectoryReporter initialized for {self._total_frames} frames on {self._n_atoms} atoms. Will save to {self._final_outpath}")

    def describeNextReport(self, simulation):
        # If no frames are to be saved, don't schedule any reports.
        if not hasattr(self, '_memmap') or self._memmap is None:
            return (0, False, False, False, False, None)
            
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, False, False, False, None)  # Request positions

    def report(self, simulation, state):
        """
        This method is called by OpenMM at each report interval to write positions to disk.
        """
        if self._memmap is None or self._report_counter >= self._total_frames:
            return

        # Get positions and convert to Angstroms
        all_positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        positions = all_positions[self._atom_indices]
        
        # Write the data directly to the memory-mapped file on disk
        self._memmap[self._report_counter] = positions.astype(self._dtype)
        self._memmap.flush()  # Ensure data is written to disk at this step.
        
        self._report_counter += 1

        # If this is the last frame, compress the file
        if self._report_counter == self._total_frames:
            self._compress_and_cleanup()

    def _compress_and_cleanup(self):
        """
        Compresses the .npy file to .npz format and removes the original file.
        """
        # Close the memmap file
        if hasattr(self, '_memmap') and self._memmap is not None:
            self._memmap.flush()
            del self._memmap

        try:
            # Load the data from the temporary .npy file
            data = np.load(self._temp_outpath)
            # Save as compressed npz with metadata
            np.savez_compressed(
                self._final_outpath, 
                positions=data,
                atom_indices=np.array(self._atom_indices),
                n_frames=data.shape[0],
                n_atoms=data.shape[1]
            )
            # Remove the temporary .npy file
            os.remove(self._temp_outpath)
            log_info(logger, f"Successfully compressed trajectory data to {self._final_outpath}")
        except Exception as e:
            log_error(logger, f"Error compressing trajectory file: {str(e)}")
            # If compression fails, keep the original file
            if os.path.exists(self._temp_outpath):
                log_warning(logger, f"Keeping original uncompressed file at {self._temp_outpath}")


# ==================================================================================================
# UTILITY FUNCTIONS FOR LOADING SAVED DATA
# ==================================================================================================

def load_trajectory_data(npz_file):
    """
    Load trajectory data from an NPZ file created by TrajectoryReporter.
    
    Parameters
    ----------
    npz_file : str
        Path to the NPZ trajectory file.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'positions': numpy array of shape (n_frames, n_atoms, 3) in Angstroms
        - 'atom_indices': numpy array of atom indices that were saved
        - 'n_frames': number of frames
        - 'n_atoms': number of atoms
        
    Example
    -------
    >>> data = load_trajectory_data('trajectory.npz')
    >>> positions = data['positions']  # Shape: (n_frames, n_atoms, 3)
    >>> atom_indices = data['atom_indices']
    >>> print(f"Loaded {data['n_frames']} frames for {data['n_atoms']} atoms")
    """
    with np.load(npz_file) as data:
        return {
            'positions': data['positions'],
            'atom_indices': data['atom_indices'],
            'n_frames': int(data['n_frames']),
            'n_atoms': int(data['n_atoms'])
        }


def load_force_data(npz_file):
    """
    Load force data from an NPZ file created by ForceReporter.
    
    Parameters
    ----------
    npz_file : str
        Path to the NPZ force file.
        
    Returns
    -------
    numpy.ndarray
        Array of shape (n_frames, n_atoms, 3) containing forces in kJ/mol/nm.
        
    Example
    -------
    >>> forces = load_force_data('forces.npz')
    >>> print(f"Loaded forces with shape: {forces.shape}")
    >>> # Calculate force magnitudes
    >>> force_magnitudes = np.linalg.norm(forces, axis=2)
    """
    with np.load(npz_file) as data:
        return data['forces']


def load_hessian_data(npz_file):
    """
    Load Hessian data from an NPZ file created by HessianReporter.
    
    Parameters
    ----------
    npz_file : str
        Path to the NPZ Hessian file.
        
    Returns
    -------
    numpy.ndarray
        Array of shape (n_frames, 3*n_atoms, 3*n_atoms) containing Hessian matrices.
        
    Example
    -------
    >>> hessians = load_hessian_data('hessians.npz')
    >>> print(f"Loaded Hessians with shape: {hessians.shape}")
    >>> # Calculate eigenvalues for first frame
    >>> eigenvals = np.linalg.eigvals(hessians[0])
    """
    with np.load(npz_file) as data:
        return data['hessians']


def trajectory_to_xyz(npz_file, xyz_file, atom_symbols=None):
    """
    Convert NPZ trajectory to XYZ format for visualization.
    
    Parameters
    ----------
    npz_file : str
        Path to the NPZ trajectory file.
    xyz_file : str
        Path to output XYZ file.
    atom_symbols : list, optional
        List of atomic symbols. If None, uses 'C' for all atoms.
        
    Example
    -------
    >>> trajectory_to_xyz('trajectory.npz', 'trajectory.xyz', ['C', 'N', 'O', 'H'])
    """
    data = load_trajectory_data(npz_file)
    positions = data['positions']
    n_frames, n_atoms, _ = positions.shape
    
    if atom_symbols is None:
        atom_symbols = ['C'] * n_atoms
    elif len(atom_symbols) != n_atoms:
        raise ValueError(f"Length of atom_symbols ({len(atom_symbols)}) must match n_atoms ({n_atoms})")
    
    with open(xyz_file, 'w') as f:
        for frame in range(n_frames):
            f.write(f"{n_atoms}\n")
            f.write(f"Frame {frame}\n")
            for atom_idx in range(n_atoms):
                x, y, z = positions[frame, atom_idx]
                f.write(f"{atom_symbols[atom_idx]} {x:.6f} {y:.6f} {z:.6f}\n")
    
    log_info(logger, f"Converted trajectory to XYZ format: {xyz_file}")