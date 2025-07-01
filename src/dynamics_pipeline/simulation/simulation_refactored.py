import os
import yaml
import argparse
from pathlib import Path
from copy import deepcopy
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import logging
import pdbfixer
from rdkit import Chem
from openff.toolkit.topology import Molecule
from openbabel import pybel
import pickle

from openmm import openmm
from openmm import app, unit
from openmm import Platform, XmlSerializer, LangevinIntegrator, CustomExternalForce, MonteCarloBarostat
from openmm.app.modeller import Modeller
from openmm.app import ForceField
from openmmforcefields.generators import SystemGenerator, GAFFTemplateGenerator

import biotite.interface.openmm as biotite_openmm 
import biotite.structure.io.pdbx as pdbx

from plinder.core.scores import query_index
from dynamics_pipeline.utils.errors import NoneLigandError, NoneConformerError
from dynamics_pipeline.utils.logger import setup_logger, log_info, log_error, log_warning, log_debug
from dynamics_pipeline.utils.preprocessing import download_nonstandard_residue
from dynamics_pipeline.data.small_molecule import load_molecule_to_openmm
from dynamics_pipeline.data.biomolecules import fix_biomolecule_with_pdb2pqr
from dynamics_pipeline.simulation.reporters import ForceReporter, HessianReporter

logger = setup_logger(name="plinder_dynamics", log_level=logging.INFO)

# ==================================================================================================
# HELPER FUNCTIONS
# ==================================================================================================

def add_backbone_posres(system: openmm.System, positions: unit.Quantity, atoms: list, restraint_force: float) -> openmm.System:
    """
    Adds backbone position restraints to the system.

    Parameters
    ----------
    system : openmm.System
        The system to add restraints to.
    positions : openmm.unit.Quantity
        The reference positions for the restraints.
    atoms : list
        A list of atoms in the topology.
    restraint_force : float
        The force constant for the restraints in kcal/mol/Å².

    Returns
    -------
    openmm.System
        The system with added backbone position restraints.
    """
    force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force_amount = restraint_force * unit.kilocalories_per_mole / unit.angstroms**2
    force.addGlobalParameter("k", force_amount)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    for i, (atom_crd, atom) in enumerate(zip(positions, atoms)):
        if atom.name in ('CA', 'C', 'N'):
            force.addParticle(i, atom_crd.value_in_unit(unit.nanometers))
    posres_sys = deepcopy(system)
    posres_sys.addForce(force)
    return posres_sys


def get_nonstandard_residues_template(nonstandard_residue: str) -> Molecule:
    """
    Downloads and loads a template for a non-standard residue.

    Parameters
    ----------
    nonstandard_residue : str
        The name of the non-standard residue.

    Returns
    -------
    openff.toolkit.topology.Molecule
        The OpenFF molecule object for the non-standard residue.
    """
    nonstandard_residue_file = f"{nonstandard_residue}.sdf"
    nonstandard_residue_filepath = Path(os.getenv('CCD_SDF_DIR', '.')) / nonstandard_residue_file
    if not nonstandard_residue_filepath.exists():
        log_info(logger, f"Downloading non-standard residue template for {nonstandard_residue}")
        if not download_nonstandard_residue(nonstandard_residue, nonstandard_residue_filepath):
            raise FileNotFoundError(f"Nonstandard residue template {nonstandard_residue_file} not found or downloaded.")
    return Molecule.from_file(str(nonstandard_residue_filepath))


# ==================================================================================================
# MDSimulation CLASS
# ==================================================================================================

class MDSimulation:
    """
    A class to set up and run molecular dynamics simulations using OpenMM.
    """
    def __init__(self, config: dict):
        """
        Initializes the MDSimulation object.

        Parameters
        ----------
        config : dict
            A dictionary containing the simulation configuration.
        """
        self.config = config
        self.system = None
        self.system_with_posres = None
        self.model = None
        self.simulation = None
        self.platform = None

        log_info(logger, f"Initializing MDSimulation for system: {self.config['info']['system_id']}")

        self._validate_inputs()
        self._setup_paths()
        self._setup_info()
        self._setup_platform()
        
        output_dir = Path(self.config['paths']['output_dir'])
        output_dir.mkdir(exist_ok=True)

    def _validate_inputs(self):
        """Validate existence of input protein and ligand files."""
        log_debug(logger, "Validating input files.")
        if not all(Path(p).exists() for p in self.config['paths'].get('raw_protein_files', [])):
            error_msg = f"One or more protein files not found: {self.config['paths']['raw_protein_files']}"
            log_error(logger, error_msg)
            raise FileNotFoundError(error_msg)
        
        if self.config['paths'].get('raw_ligand_files'):
            if not all(Path(l).exists() for l in self.config['paths']['raw_ligand_files']):
                error_msg = f"One or more ligand files not found: {self.config['paths']['raw_ligand_files']}"
                log_error(logger, error_msg)
                raise FileNotFoundError(error_msg)

    def _setup_paths(self):
        """Generates and stores all necessary file paths for the simulation."""
        log_debug(logger, "Setting up simulation file paths.")
        output_dir = Path(self.config['paths']['output_dir'])
        system_id = self.config['info']['system_id']

        # Initial structure and system files
        self.config['paths']['init_complex_filepath'] = str(output_dir / f"{system_id}_init_complex.cif")
        self.config['paths']['init_topology_filepath'] = str(output_dir / f"{system_id}_init_topology.cif")
        self.config['paths']['init_system_filepath'] = str(output_dir / f"{system_id}_init_system.xml")
        self.config['paths']['init_system_with_posres_filepath'] = str(output_dir / f"{system_id}_init_system_with_posres.xml")
        self.config['paths']['molecule_forcefield_dirpath'] = str(output_dir / f"forcefields")
        os.makedirs(self.config['paths']['molecule_forcefield_dirpath'], exist_ok=True)

        # Per-stage file paths
        path_configs = {
            'checkpoints': {'dir': 'checkpoints', 'suffix': 'checkpoint', 'ext': 'dcd'},
            'trajectories': {'dir': 'trajectories', 'suffix': 'trajectory', 'ext': 'xtc'},
            'state_reporters': {'dir': 'state_data_reporters', 'suffix': 'state_data', 'ext': 'csv'},
            'states': {'dir': 'states', 'suffix': 'state', 'ext': 'xml'},
            'topologies': {'dir': 'topologies', 'suffix': 'topology', 'ext': 'cif'},
            'forces': {'dir': 'forces', 'suffix': 'forces', 'ext': 'npz'},
            'hessian': {'dir': 'hessian', 'suffix': 'hessian', 'ext': 'npz'}
        }
        stages = ['warmup', 'backbone_removal', 'nvt', 'npt', 'production']

        for key, p_config in path_configs.items():
            self.config['paths'][key] = {}
            p_dir = output_dir / p_config['dir']
            p_dir.mkdir(exist_ok=True)
            for stage in stages:
                if key == 'state_reporters':
                    path_key = f"{stage}_state_reporters_filepath"
                else:
                    path_key = f"{stage}_{p_config['suffix']}_filepath"
                
                filename = f"{system_id}_{stage}_{p_config['suffix']}.{p_config['ext']}"
                self.config['paths'][key][path_key] = str(p_dir / filename)
    
    def _setup_info(self):
        """Sets up protein, ligand, and simulation status information."""
        log_debug(logger, "Setting up simulation info.")
        self._set_protein_info()
        self._set_ligand_info()
        self._set_simulation_info()
        
    def _set_protein_info(self):
        """Sets protein information in the config."""
        if 'protein_info' not in self.config:
            self.config['protein_info'] = {}
        
        protein_files = self.config['paths'].get('raw_protein_files', [])
        self.config['protein_info'].setdefault('n_prot', len(protein_files))
        self.config['protein_info'].setdefault('protein_names', [Path(p).stem for p in protein_files])
        self.config['protein_info'].setdefault('protein_formats', [Path(p).suffix for p in protein_files])

        if 'protein_chain_ids' not in self.config['protein_info']:
            all_chain_ids = []
            for protein_file in protein_files:
                all_chain_ids.extend(self._get_chain_ids_from_file(protein_file))
            self.config['protein_info']['protein_chain_ids'] = list(set(all_chain_ids))

    def _set_ligand_info(self):
        """Sets ligand information in the config."""
        if 'ligand_info' not in self.config:
            self.config['ligand_info'] = {}

        ligand_files = self.config['paths'].get('raw_ligand_files', [])
        ligand_names = [Path(l).stem for l in ligand_files]
        
        self.config['ligand_info'].setdefault('n_lig', len(ligand_files))
        self.config['ligand_info'].setdefault('ligand_names', ligand_names)
        self.config['ligand_info'].setdefault('ligand_formats', [Path(l).suffix for l in ligand_files])
        self.config['ligand_info'].setdefault('ligand_charges', {})

        if self.config['ligand_info'].get('ligand_ccd_codes') is None:
            if self.config['info'].get('use_plinder_index', False):
                plindex = query_index(columns=['system_id', 'ligand_ccd_code', 'ligand_id'], splits=["*"])
                system_index = plindex[plindex['system_id'] == self.config['info']['system_id']]
                
                ccd_codes = []
                for name in ligand_names:
                    entry = system_index[system_index['ligand_id'].str.endswith(name)]
                    ccd_codes.append(entry['ligand_ccd_code'].values[0] if not entry.empty else 'LIG')
                self.config['ligand_info']['ligand_ccd_codes'] = ccd_codes
            else:
                self.config['ligand_info']['ligand_ccd_codes'] = ['LIG'] * len(ligand_names)

    def _set_simulation_info(self):
        """Sets simulation status information."""
        if 'simulation_params' not in self.config:
            raise KeyError("`simulation_params` not found in config file!")
            
        self.config['info']['simulation_status'] = {
            stage: 'Done' if Path(self.config['paths']['topologies'][f'{stage}_topology_filepath']).exists() else 'Not Done'
            for stage in ['warmup', 'backbone_removal', 'nvt', 'npt', 'production']
        }

    def _setup_platform(self):
        """Configures the OpenMM platform for the simulation."""
        log_debug(logger, "Setting up OpenMM platform.")
        platform_name = self.config['simulation_params']['platform']['type']
        self.platform = Platform.getPlatformByName(platform_name)
        if platform_name == 'CUDA':
            self.platform.setPropertyDefaultValue('Precision', 'mixed')
            device_index = self.config['simulation_params']['platform'].get('devices', '0')
            self.platform.setPropertyDefaultValue('CudaDeviceIndex', str(device_index))

    def update_simulation_status(self, stage: str, status: str):
        """Updates the status of a simulation stage."""
        if stage in self.config['info']['simulation_status']:
            self.config['info']['simulation_status'][stage] = status
        else:
            log_warning(logger, f"Attempted to update status for unknown stage: {stage}")
    
    def _get_chain_ids_from_file(self, filepath: str) -> list[str]:
        """
        Extracts a list of unique chain IDs from a PDB or CIF file.
        
        Parameters
        ----------
        filepath : str
            Path to the input file.
            
        Returns
        -------
        list[str]
            A list of unique chain IDs found in the file.
        """
        p = Path(filepath)
        if p.suffix == '.pdb':
            structure = app.PDBFile(filepath)
        elif p.suffix == '.cif':
            structure = app.PDBxFile(filepath)
        else:
            log_warning(logger, f"Unsupported file format for reading chain IDs: {p.suffix}")
            return []
            
        chain_ids = []
        for chain in structure.getTopology().chains():
            if chain.id not in chain_ids:
                chain_ids.append(chain.id)
        return chain_ids

    # ==================================================================================================
    # SYSTEM SETUP AND PREPARATION
    # ==================================================================================================

    def set_system(self):
        """
        Main method to set up the simulation system.
        It loads from files if they exist, otherwise it processes inputs to create them.
        """
        init_system_path = self.config['paths']['init_system_filepath']
        init_posres_path = self.config['paths']['init_system_with_posres_filepath']
        init_topology_path = self.config['paths']['init_topology_filepath']

        if all(Path(p).exists() for p in [init_system_path, init_posres_path, init_topology_path]):
            log_info(logger, "Loading pre-existing system, topology, and position restraint files.")
            self._load_system_from_files(init_topology_path, init_system_path, init_posres_path)
        else:
            log_info(logger, "Processing inputs to generate new system and topology files.")
            self._create_system_from_scratch()

    def _load_system_from_files(self, topology_path, system_path, posres_path):
        """Loads the system, topology, and position restraints from files."""
        if topology_path.endswith('.cif'):
            complex_file = app.PDBxFile(topology_path)
        elif topology_path.endswith('.pdb'):
            complex_file = app.PDBFile(topology_path)
        else:
            raise ValueError(f"Unsupported topology file format: {topology_path}")

        self.model = Modeller(complex_file.topology, complex_file.positions)
        with open(system_path, 'r') as f:
            self.system = XmlSerializer.deserialize(f.read())
        with open(posres_path, 'r') as f:
            self.system_with_posres = XmlSerializer.deserialize(f.read())

    def _create_system_from_scratch(self):
        """Processes protein and ligand inputs to create and save the simulation system."""
        complex_model, openff_molecules = self.process_complex()
        
        nonstandard_templates = []
        if 'nonstandard_residues' in self.config['protein_info']:
            for res in self.config['protein_info']['nonstandard_residues']:
                nonstandard_templates.append(get_nonstandard_residues_template(res))

        generator_molecules = openff_molecules + nonstandard_templates
        
        forcefield_kwargs = self.config['forcefield'].get('forcefield_kwargs', {})
        hydrogen_mass = forcefield_kwargs.get('hydrogenMass')
        if hydrogen_mass:
            forcefield_kwargs['hydrogenMass'] = hydrogen_mass * unit.amu

        gaff = GAFFTemplateGenerator(forcefield=self.config['forcefield']['ligandFF'])
        ffxml_contents = {}
        for generator_molecule in generator_molecules:
            ffxml_contents[generator_molecule.name] = gaff.generate_residue_template(generator_molecule)
            with open(Path(self.config['paths']['molecule_forcefield_dirpath']) / f"{generator_molecule.name}.xml", 'w') as f:
                f.write(ffxml_contents[generator_molecule.name])

        system_generator = SystemGenerator(
            forcefields=[
                self.config['forcefield']['proteinFF'],
                self.config['forcefield']['nucleicFF'],
                self.config['forcefield']['waterFF'],
                gaff.gaff_xml_filename
            ],
            small_molecule_forcefield=self.config['forcefield']['ligandFF'],
            molecules=generator_molecules,
            forcefield_kwargs=forcefield_kwargs
        )

        # Register the GAFF template generator for our custom molecules (ligands and non-standard residues)
        # gaff_gen = GAFFTemplateGenerator(molecules=generator_molecules, forcefield=self.config['forcefield']['ligandFF'])
        # system_generator.forcefield.registerTemplateGenerator(gaff_gen.generator)

        if self.config['preprocessing'].get('add_solvate', True):
            log_info(logger, "Adding solvent to the system.")
            complex_model.addSolvent(
                system_generator.forcefield,
                model=self.config['forcefield']['water_model'],
                padding=self.config['preprocessing']['box_padding'] * unit.nanometers,
                ionicStrength=self.config['preprocessing']['ionic_strength'] * unit.molar,
            )
        
        self.model = complex_model
        
        log_info(logger, "Creating OpenMM system.")
        self.system = system_generator.create_system(self.model.topology, molecules=generator_molecules)

        log_info(logger, "Adding backbone position restraints.")
        self.system_with_posres = add_backbone_posres(
            self.system, self.model.positions, list(self.model.topology.atoms()),
            self.config['simulation_params']['backbone_restraint_force']
        )
        
        self._save_initial_files()

    def _save_initial_files(self):
        """Saves the initial topology, system, and position-restrained system files."""
        log_info(logger, "Saving initial topology and system XML files.")

        biotite_topology = biotite_openmm.from_topology(self.model.topology)
        # Convert OpenMM positions from nanometers to angstroms for biotite
        positions_angstrom = self.model.positions.value_in_unit(unit.angstrom)
        biotite_topology.coord = np.array(positions_angstrom, dtype=np.float32)

        cif_file = pdbx.CIFFile()
        pdbx.set_structure(cif_file, biotite_topology, include_bonds=True)
        cif_file.write(self.config['paths']['init_topology_filepath'])

        # pdb_path = self.config['paths']['init_topology_filepath'].replace('.cif', '.pdb')
        # with open(pdb_path, 'w') as f:
        #     app.PDBFile.writeFile(self.model.topology, self.model.positions, f, keepIds=True)
        
        pdb_path = self.config['paths']['init_topology_filepath'].replace('.cif', '.pdb')
        with open(pdb_path, 'w') as f:
            app.PDBFile.writeFile(self.model.topology, self.model.positions, f, keepIds=True)

        with open(self.config['paths']['init_system_filepath'], 'w') as f:
            f.write(XmlSerializer.serialize(self.system))
            
        with open(self.config['paths']['init_system_with_posres_filepath'], 'w') as f:
            f.write(XmlSerializer.serialize(self.system_with_posres))

    def process_complex(self) -> (Modeller, list):
        """
        Processes protein and ligand files to create a complex.
        
        Returns
        -------
        openmm.app.Modeller
            The Modeller object containing the full complex.
        list
            A list of OpenFF molecule objects for the ligands.
        """
        log_info(logger, "Processing protein and ligand files to build complex.")
        if self.config['preprocessing']['process_protein']:
            all_nonstandard_res = []
            complex_model = None
            for i, protein_file in enumerate(self.config['paths']['raw_protein_files']):
                original_chain_ids = self._get_chain_ids_from_file(protein_file)
                fixer, nonstd = self.process_protein(protein_file)
                all_nonstandard_res.extend(nonstd)
                
                # Remap chain IDs in the fixer topology to match original file
                fixer_chains = list(fixer.topology.chains())
                if len(fixer_chains) == len(original_chain_ids):
                    log_info(logger, f"Remapping chain IDs for {Path(protein_file).name} to: {original_chain_ids}")
                    for chain, original_id in zip(fixer_chains, original_chain_ids):
                        chain.id = original_id
                else:
                    log_warning(logger, f"Chain count mismatch for {Path(protein_file).name}. PDBFixer created {len(fixer_chains)} chains, but original file had {len(original_chain_ids)}. Cannot restore original chain IDs.")

                if i == 0:
                    complex_model = Modeller(fixer.topology, fixer.positions)
                else:
                    complex_model.add(fixer.topology, fixer.positions)

            if all_nonstandard_res:
                self.config['protein_info']['nonstandard_residues'] = list(set(all_nonstandard_res))
        else:
            complex_file = app.PDBxFile(self.config['paths']['init_topology_filepath'])
            complex_model = Modeller(complex_file.topology, complex_file.positions)

        openff_molecules = []
        if self.config['preprocessing']['process_ligand']:
            # Determine which chain IDs to use for ligands
            if 'ligand_names' in self.config['ligand_info']:
                ligand_chain_ids = self.config['ligand_info']['ligand_names']
                if len(ligand_chain_ids) != len(self.config['paths']['raw_ligand_files']):
                    raise ValueError("Length of 'ligand_names' in config must match the number of ligand files.")
                log_info(logger, f"Using custom chain IDs for ligands: {ligand_chain_ids}")
            else:
                ligand_chain_ids = self.config['ligand_info']['ligand_ccd_codes']
                log_info(logger, f"Using ligand CCD codes as chain IDs: {ligand_chain_ids}")

            for i, (lig_file, lig_format, lig_name, lig_ccd) in enumerate(zip(
                self.config['paths']['raw_ligand_files'],
                self.config['ligand_info']['ligand_formats'],
                self.config['ligand_info']['ligand_names'],
                self.config['ligand_info']['ligand_ccd_codes']
            )):
                openmm_mol, openff_mol = self.process_ligand(lig_file, lig_format, lig_name, lig_ccd)
                openff_molecules.append(openff_mol)
                
                # Add ligand to the complex, carefully setting chain and residue info
                current_chain_count = len(list(complex_model.topology.chains()))
                complex_model.add(openmm_mol.topology, openmm_mol.positions)
                new_chain = list(complex_model.topology.chains())[current_chain_count]
                
                chain_id_to_use = ligand_chain_ids[i]
                new_chain.id = chain_id_to_use
                
                for res in new_chain.residues():
                    res.name = lig_ccd # Use CCD code for residue name
                    # Rename atoms to be unique if they are generic
                    atom_counts = defaultdict(int)
                    for atom in res.atoms():
                        if not atom.name:
                            symbol = atom.element.symbol
                            atom_counts[symbol] += 1
                            atom.name = f"{symbol}{atom_counts[symbol]}"
        return complex_model, openff_molecules
        
    def process_protein(self, input_filepath: str, ph: float = 7.4) -> (pdbfixer.PDBFixer, list):
        """
        Processes a protein file with PDBFixer to prepare it for simulation.

        Parameters
        ----------
        input_filepath : str
            Path to the input protein PDB file.
        ph : float
            The pH to use for adding missing hydrogens.

        Returns
        -------
        pdbfixer.PDBFixer
            The PDBFixer object after processing.
        list
            A list of non-standard residue names found.
        """
        log_info(logger, f"Processing protein: {input_filepath} with PDBFixer.")
        fixer = pdbfixer.PDBFixer(filename=input_filepath)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.findMissingAtoms()
        
        nonstandard_residues = [res[0].name for res in fixer.nonstandardResidues]

        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(ph)
        
        output_dir = Path(self.config['paths']['output_dir'])
        output_filepath = output_dir / f"{Path(input_filepath).stem}_fixed.pdb"
        with open(output_filepath, 'w') as f:
            app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
            
        return fixer, list(set(nonstandard_residues))

    def process_ligand(self, input_filepath: str, input_format: str, input_name: str, residue_name: str) -> (Modeller, Molecule):
        """
        Processes a ligand file to prepare it for simulation.

        Parameters
        ----------
        input_filepath : str
            Path to the ligand file.
        input_format : str
            The format of the ligand file (e.g., '.sdf').
        input_name : str
            The name of the ligand.
        residue_name : str
            The 3-letter residue name (CCD code) to assign to the ligand.

        Returns
        -------
        openmm.app.Modeller
            The Modeller object for the ligand.
        openff.toolkit.topology.Molecule
            The OpenFF molecule object for the ligand.
        """
        log_info(logger, f"Processing ligand: {input_name} with residue name {residue_name}")
        openff_mol = load_molecule_to_openmm(input_filepath, input_format)
        openff_mol.name = residue_name
        
        if input_name in self.config['ligand_info']['ligand_charges']:
            charges = self.config['ligand_info']['ligand_charges'][input_name]
            openff_mol.partial_charges = unit.Quantity(np.array(charges), unit.elementary_charge)
        else:
            try:
                openff_mol.assign_partial_charges(partial_charge_method="gasteiger", use_conformers=openff_mol.conformers)
            except Exception:
                openff_mol.assign_partial_charges(partial_charge_method="gasteiger")
            self.config['ligand_info']['ligand_charges'][input_name] = openff_mol.partial_charges.m.tolist()

        openmm_topology = openff_mol.to_topology().to_openmm()
        openmm_positions = openff_mol.conformers[0].to_openmm()
        openmm_model = Modeller(openmm_topology, openmm_positions)
        
        return openmm_model, openff_mol
    
    # ==================================================================================================
    # SIMULATION EXECUTION
    # ==================================================================================================

    def run_pipeline(self):
        """
        Executes the full simulation pipeline according to the config.
        """
        log_info(logger, "Starting simulation pipeline.")
        
        if self.system is None:
            self.set_system()
            
        pipeline_stages = ['warmup', 'backbone_removal', 'nvt', 'npt', 'production']
        for stage in pipeline_stages:
            if self.config['simulation_params'][stage].get('run', False):
                if self.config['info']['simulation_status'][stage] == 'Not Done':
                    log_info(logger, f"Running {stage} stage.")
                    stage_method = getattr(self, stage)
                    stage_method()
                else:
                    log_info(logger, f"Skipping {stage} stage as it is already marked as 'Done'.")
            else:
                log_info(logger, f"Skipping {stage} stage as it is not configured to run.")
        log_info(logger, "Simulation pipeline finished.")

    def _run_simulation_stage(self, stage_name: str, use_posres: bool = False, use_barostat: bool = False):
        """
        A general method to run a simulation stage (NVT, NPT, etc.).
        
        Parameters
        ----------
        stage_name : str
            The name of the simulation stage (e.g., 'nvt').
        use_posres : bool
            Whether to use the system with position restraints.
        use_barostat : bool
            Whether to add a MonteCarloBarostat to the system.
        """
        params = self.config['simulation_params'][stage_name]
        paths = self.config['paths']
        
        log_info(logger, f"--- Starting {stage_name.upper()} Stage ---")
        
        # 1. Setup Integrator
        integrator = LangevinIntegrator(
            params['temp'] * unit.kelvin,
            params['friction'] / unit.picoseconds,
            params['time_step'] * unit.femtoseconds
        )

        # 2. Setup System
        system_to_use = self.system_with_posres if use_posres else deepcopy(self.system)
        if use_barostat:
            barostat = MonteCarloBarostat(params['pressure'] * unit.atmospheres, params['temp'] * unit.kelvin)
            system_to_use.addForce(barostat)

        # 3. Setup Simulation object
        simulation = app.Simulation(self.model.topology, system_to_use, integrator, platform=self.platform)

        # 4. Load state from previous stage if available
        prev_checkpoint = self._get_previous_stage_checkpoint(stage_name)
        checkpoint_path = paths['checkpoints'][f'{stage_name}_checkpoint_filepath']

        if Path(checkpoint_path).exists():
            log_info(logger, f"Loading checkpoint for {stage_name}: {checkpoint_path}")
            simulation.loadCheckpoint(checkpoint_path)
            steps_done = simulation.context.getStepCount()
            steps_to_run = int(params['nsteps']) - steps_done
        elif prev_checkpoint and Path(prev_checkpoint).exists():
            log_info(logger, f"Loading checkpoint from previous stage: {prev_checkpoint}")
            simulation.loadCheckpoint(prev_checkpoint)
            simulation.context.setStepCount(0)
            steps_to_run = int(params['nsteps'])
        else:
            log_info(logger, "No checkpoint found. Starting from initial positions.")
            simulation.context.setPositions(self.model.positions)
            simulation.minimizeEnergy()
            simulation.context.setVelocitiesToTemperature(params['temp'] * unit.kelvin)
            steps_to_run = int(params['nsteps'])
        
        if steps_to_run <= 0:
            log_info(logger, f"Stage {stage_name} already completed. Skipping.")
            self._save_final_topology(simulation, stage_name)
            self.update_simulation_status(stage_name, 'Done')
            return

        # 5. Setup Reporters
        self._set_reporters(
            simulation=simulation,
            stage_name=stage_name,
            total_steps=params['nsteps']
        )
        
        # 6. Run Simulation
        log_info(logger, f"Running {stage_name} for {steps_to_run} steps.")
        simulation.step(steps_to_run)
        
        # 7. Save final state and topology
        simulation.saveState(paths['states'][f'{stage_name}_state_filepath'])
        self._save_final_topology(simulation, stage_name)
        
        self.update_simulation_status(stage_name, 'Done')
        log_info(logger, f"--- {stage_name.upper()} Stage Finished ---")

    def _get_previous_stage_checkpoint(self, stage_name: str) -> str:
        """Gets the checkpoint file path from the preceding stage."""
        stage_order = ['warmup', 'backbone_removal', 'nvt', 'npt', 'production']
        try:
            current_index = stage_order.index(stage_name)
            if current_index == 0:
                return None
            prev_stage = stage_order[current_index - 1]
            return self.config['paths']['checkpoints'][f'{prev_stage}_checkpoint_filepath']
        except (ValueError, KeyError):
            return None

    def warmup(self):
        """
        Performs a warmup simulation with gradual heating and position restraints.
        """
        params = self.config['simulation_params']['warmup']
        paths = self.config['paths']
        log_info(logger, "--- Starting WARMUP Stage (Gradual Heating) ---")

        integrator = LangevinIntegrator(
            params['init_temp'] * unit.kelvin,
            params['friction'] / unit.picoseconds,
            params['time_step'] * unit.femtoseconds
        )
        simulation = app.Simulation(self.model.topology, self.system_with_posres, integrator, platform=self.platform)
        
        heating_steps_per_degree = params['heating_step']
        temp_range = params['final_temp'] - params['init_temp']
        total_heating_steps = temp_range * heating_steps_per_degree
        
        # Load checkpoint or initialize
        checkpoint_path = paths['checkpoints']['warmup_checkpoint_filepath']
        if Path(checkpoint_path).exists():
            simulation.loadCheckpoint(checkpoint_path)
            steps_done = simulation.context.getStepCount()
            degrees_done = steps_done // heating_steps_per_degree
            initial_temp = params['init_temp'] + degrees_done
            log_info(logger, f"Resuming warmup from step {steps_done} at {initial_temp}K.")
        else:
            simulation.context.setPositions(self.model.positions)
            log_info(logger, "Minimizing energy before warmup.")
            simulation.minimizeEnergy()
            initial_temp = params['init_temp']
            log_info(logger, f"Starting fresh warmup from {initial_temp}K.")

        self._set_reporters(simulation, 'warmup', total_heating_steps)
        
        simulation.context.setVelocitiesToTemperature(initial_temp * unit.kelvin)
        integrator.setTemperature(initial_temp * unit.kelvin)

        temp_schedule = np.linspace(initial_temp, params['final_temp'], int(temp_range) + 1)
        
        for temp in tqdm(temp_schedule[1:], desc="Heating", unit="K"):
            integrator.setTemperature(temp * unit.kelvin)
            simulation.step(heating_steps_per_degree)
        
        simulation.saveState(paths['states']['warmup_state_filepath'])
        self._save_final_topology(simulation, 'warmup')
        self.update_simulation_status('warmup', 'Done')
        log_info(logger, f"--- WARMUP Stage Finished at {params['final_temp']}K ---")

    def remove_backbone_constraints(self):
        """
        Performs a simulation to gradually remove backbone position restraints.
        """
        params = self.config['simulation_params']['backbone_removal']
        paths = self.config['paths']
        log_info(logger, "--- Starting BACKBONE CONSTRAINT REMOVAL Stage ---")

        integrator = LangevinIntegrator(
            params['temp'] * unit.kelvin,
            params['friction'] / unit.picoseconds,
            params['time_step'] * unit.femtoseconds
        )
        simulation = app.Simulation(self.model.topology, self.system_with_posres, integrator, platform=self.platform)

        # Load state
        prev_checkpoint = self._get_previous_stage_checkpoint('backbone_removal')
        checkpoint_path = paths['checkpoints']['backbone_removal_checkpoint_filepath']

        if Path(checkpoint_path).exists():
            simulation.loadCheckpoint(checkpoint_path)
        elif Path(prev_checkpoint).exists():
            simulation.loadCheckpoint(prev_checkpoint)
            simulation.context.setStepCount(0)
        else:
            raise FileNotFoundError("Cannot start backbone removal without a warmup checkpoint.")

        n_loops = int(params['nloops'])
        steps_per_loop = int(params['nsteps']) // n_loops
        initial_force = self.config['simulation_params']['backbone_restraint_force']
        force_decrement = initial_force / n_loops

        self._set_reporters(simulation, 'backbone_removal', params['nsteps'])

        for i in tqdm(range(n_loops), desc="Removing Constraints"):
            force_k = (initial_force - (i + 1) * force_decrement) * unit.kilocalories_per_mole / unit.angstroms**2
            simulation.context.setParameter('k', force_k)
            simulation.step(steps_per_loop)
        
        simulation.context.setParameter('k', 0.0) # Ensure it's fully off
        
        simulation.saveState(paths['states']['backbone_removal_state_filepath'])
        self._save_final_topology(simulation, 'backbone_removal')
        self.update_simulation_status('backbone_removal', 'Done')
        log_info(logger, "--- BACKBONE CONSTRAINT REMOVAL Stage Finished ---")
    
    def nvt(self):
        self._run_simulation_stage('nvt')
        
    def npt(self):
        self._run_simulation_stage('npt', use_barostat=True)

    def production(self):
        self._run_simulation_stage('production', use_barostat=True)

    # ==================================================================================================
    # HELPER METHODS
    # ==================================================================================================

    def _set_reporters(self, simulation: app.Simulation, stage_name: str, total_steps: int):
        """
        Sets up reporters for a given simulation stage.
        """
        params = self.config['simulation_params'][stage_name]
        paths = self.config['paths']
        
        checkpoint_path = paths['checkpoints'][f'{stage_name}_checkpoint_filepath']
        trajectory_path = paths['trajectories'][f'{stage_name}_trajectory_filepath']
        state_data_path = paths['state_reporters'][f'{stage_name}_state_reporters_filepath']
        forces_path = paths['forces'][f'{stage_name}_forces_filepath']
        hessian_path = paths['hessian'][f'{stage_name}_hessian_filepath']
        
        # Get atom indices for proteins and ligands, excluding hydrogens, water, and ions
        protein_chains = self.config['protein_info'].get('protein_chain_ids', [])
        ligand_chains = self.config['ligand_info'].get('ligand_names', [])
        allowed_chains = protein_chains + ligand_chains
        
        # Define residues to exclude
        excluded_residues = {'HOH', 'WAT', 'TIP3', 'TIP4', 'TIP5', 'SPC', 'SPCE', 'CL', 'NA'}
        
        atom_indices = []
        excluded_counts = {'hydrogens': 0, 'excluded_residues': 0, 'wrong_chains': 0}
        
        for atom in simulation.topology.atoms():
            # Skip hydrogens
            if atom.element.symbol == 'H':
                excluded_counts['hydrogens'] += 1
                continue
            # Skip excluded residues (water, ions)
            if atom.residue.name in excluded_residues:
                excluded_counts['excluded_residues'] += 1
                continue
            # Only include atoms from allowed chains (protein and ligand chains)
            if atom.residue.chain.id in allowed_chains:
                atom_indices.append(atom.index)
            else:
                excluded_counts['wrong_chains'] += 1
        
        total_atoms = simulation.topology.getNumAtoms()
        log_info(logger, f"Atom filtering summary: Total atoms: {total_atoms}, "
                        f"Selected: {len(atom_indices)}, "
                        f"Excluded - Hydrogens: {excluded_counts['hydrogens']}, "
                        f"Water/Ions: {excluded_counts['excluded_residues']}, "
                        f"Wrong chains: {excluded_counts['wrong_chains']}")
        log_info(logger, f"Allowed chains: {allowed_chains}")
        log_debug(logger, f"Excluded residue types: {excluded_residues}")

        # Checkpoint Reporter
        simulation.reporters.append(app.CheckpointReporter(
            file=checkpoint_path,
            reportInterval=params['checkpoint_interval']
        ))
        
        # Trajectory Reporter (XTC)
        simulation.reporters.append(app.XTCReporter(
            file=trajectory_path,
            reportInterval=params['trajectory_interval'],
            append=Path(trajectory_path).exists()
        ))
        
        # Forces Reporter
        if self.config['simulation_params'].get('save_forces', False):
            simulation.reporters.append(ForceReporter(
                file=forces_path,
                reportInterval=params['state_data_reporter_interval'],
                total_steps=total_steps,
                atom_indices=atom_indices
            ))

        # Hessian Reporter
        if self.config['simulation_params'].get('save_hessian', False):
            simulation.reporters.append(HessianReporter(
                file=hessian_path,
                reportInterval=params['state_data_reporter_interval'],
                total_steps=total_steps,
                atom_indices=atom_indices,
                simulation=simulation
            ))

        # State Data Reporter
        simulation.reporters.append(app.StateDataReporter(
            file=state_data_path,
            reportInterval=params['state_data_reporter_interval'],
            step=True, potentialEnergy=True, kineticEnergy=True, temperature=True,
            volume=True, progress=True, remainingTime=True, speed=True,
            totalSteps=total_steps, separator='\t', append=Path(state_data_path).exists()
        ))
        log_debug(logger, f"Reporters set for stage: {stage_name}")

    def _save_final_topology(self, simulation: app.Simulation, stage_name: str):
        """Saves the final topology of a simulation stage."""
        topology_path = self.config['paths']['topologies'][f'{stage_name}_topology_filepath']
        log_info(logger, f"Saving final topology for stage {stage_name} to {topology_path}")
        positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
        
        topology = simulation.topology
        biotite_topology = biotite_openmm.from_topology(topology)
        # Convert OpenMM positions from nanometers to angstroms for biotite
        positions_angstrom = positions.value_in_unit(unit.angstrom)
        biotite_topology.coord = np.array(positions_angstrom, dtype=np.float32)
        
        cif_file = pdbx.CIFFile()
        pdbx.set_structure(cif_file, biotite_topology, include_bonds=True)
        cif_file.write(topology_path)