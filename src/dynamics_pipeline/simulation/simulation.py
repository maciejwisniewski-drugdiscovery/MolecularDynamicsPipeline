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

from openmm import app, unit
from openmm import Platform, XmlSerializer, LangevinIntegrator, CustomExternalForce, MonteCarloBarostat
from openmm.app.modeller import Modeller
from openmm.app import ForceField
from openmmforcefields.generators import SystemGenerator

from plinder.core.scores import query_index
from dynamics_pipeline.utils.errors import NoneLigandError, NoneConformerError
from dynamics_pipeline.utils.logger import setup_logger, log_info, log_error, log_warning, log_debug
from dynamics_pipeline.data.small_molecule import load_molecule_to_openmm
from dynamics_pipeline.data.biomolecules import fix_biomolecule_with_pdb2pqr

logger = setup_logger(name="plinder_dynamics", log_level=logging.INFO)


def add_backbone_posres(system, positions, atoms, restraint_force):
  force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
  force_amount = restraint_force * unit.kilocalories_per_mole/unit.angstroms**2
  force.addGlobalParameter("k", force_amount)
  force.addPerParticleParameter("x0")
  force.addPerParticleParameter("y0")
  force.addPerParticleParameter("z0")
  for i, (atom_crd, atom) in enumerate(zip(positions, atoms)):
    if atom.name in  ('CA', 'C', 'N'):
      force.addParticle(i, atom_crd.value_in_unit(unit.nanometers))
  posres_sys = deepcopy(system)
  posres_sys.addForce(force)
  return posres_sys


class MDSimulation:
    def __init__(self, config):
        '''
        Class for running molecular dynamics simulations with OpenMM
        '''
        self.config = config
        log_info(logger, f"Initializing MDSimulation with config file: {self.config['info']['simulation_id']}")
        
        if config['info'].get('use_plinder_index', False) == True:
            PLINDEX = query_index(columns = ['system_id','ligand_ccd_code','ligand_id'], splits=["*"])
            self.plindex = PLINDEX[PLINDEX['system_id'] == self.config['info']['system_id']]
        else:
            self.plindex = None

        # Validate input files
        if not any([os.path.exists(protein_file) for protein_file in self.config['paths']['raw_protein_files']]):
            error_msg = f"One or more of the Protein files: {self.config['paths']['raw_protein_files']} dont exist!"
            log_error(logger, error_msg)
            raise FileNotFoundError(error_msg)

        if len(self.config['paths']['raw_ligand_files']) > 0:
            assert any([os.path.exists(ligand_file) for ligand_file in self.config['paths']['raw_ligand_files']]), f"One or more of the Ligand files: {self.config['paths']['raw_ligand_files']} dont exist!"

        self.set_ligand_info()
        self.set_protein_info()
        self.set_files_info()
        self.set_simulation_info()

        # Initialize simulation components
        self.system = None
        self.model = None
        self.simulation = None
        self.get_platform()

        # Create simulation output directory if it doesn't exist
        os.makedirs(self.config['paths']['output_dir'], exist_ok=True)


    def set_files_info(self):
        # Initial files
        self.config['paths']['init_complex_filepath'] = os.path.join(self.config['paths']['output_dir'], f"{self.config['info']['system_id']}_init_complex.cif")
        self.config['paths']['init_topology_filepath'] = os.path.join(self.config['paths']['output_dir'], f"{self.config['info']['system_id']}_init_topology.cif")
        self.config['paths']['init_system_filepath'] = os.path.join(self.config['paths']['output_dir'], f"{self.config['info']['system_id']}_init_system.xml")
        self.config['paths']['init_system_with_posres_filepath'] = os.path.join(self.config['paths']['output_dir'], f"{self.config['info']['system_id']}_init_system_with_posres.xml")
        
        # Checkpoint files
        checkpoint_dir = os.path.join(self.config['paths']['output_dir'], 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.config['paths']['checkpoints'] = {}
        self.config['paths']['checkpoints']['warmup_checkpoint_filepath'] = os.path.join(checkpoint_dir, f"{self.config['info']['system_id']}_warmup_checkpoint.dcd")
        self.config['paths']['checkpoints']['backbone_removal_checkpoint_filepath'] = os.path.join(checkpoint_dir, f"{self.config['info']['system_id']}_backbone_removal_checkpoint.dcd")
        self.config['paths']['checkpoints']['nvt_checkpoint_filepath'] = os.path.join(checkpoint_dir, f"{self.config['info']['system_id']}_nvt_checkpoint.dcd")
        self.config['paths']['checkpoints']['npt_checkpoint_filepath'] = os.path.join(checkpoint_dir, f"{self.config['info']['system_id']}_npt_checkpoint.dcd")
        self.config['paths']['checkpoints']['production_checkpoint_filepath'] = os.path.join(checkpoint_dir, f"{self.config['info']['system_id']}_production_checkpoint.dcd")

        # Trajectory files
        trajectory_dir = os.path.join(self.config['paths']['output_dir'], 'trajectories')
        os.makedirs(trajectory_dir, exist_ok=True)
        self.config['paths']['trajectories'] = {}
        self.config['paths']['trajectories']['warmup_trajectory_filepath'] = os.path.join(trajectory_dir, f"{self.config['info']['system_id']}_warmup_trajectory.xtc")
        self.config['paths']['trajectories']['backbone_removal_trajectory_filepath'] = os.path.join(trajectory_dir, f"{self.config['info']['system_id']}_backbone_removal_trajectory.xtc")
        self.config['paths']['trajectories']['nvt_trajectory_filepath'] = os.path.join(trajectory_dir, f"{self.config['info']['system_id']}_nvt_trajectory.xtc")
        self.config['paths']['trajectories']['npt_trajectory_filepath'] = os.path.join(trajectory_dir, f"{self.config['info']['system_id']}_npt_trajectory.xtc")
        self.config['paths']['trajectories']['production_trajectory_filepath'] = os.path.join(trajectory_dir, f"{self.config['info']['system_id']}_production_trajectory.xtc")

        # State Data Reporter files
        state_data_reporter_dir = os.path.join(self.config['paths']['output_dir'], 'state_data_reporters')
        os.makedirs(state_data_reporter_dir, exist_ok=True)
        self.config['paths']['state_reporters'] = {}
        self.config['paths']['state_reporters']['warmup_state_reporters_filepath'] = os.path.join(state_data_reporter_dir, f"{self.config['info']['system_id']}_warmup_state_data.csv")
        self.config['paths']['state_reporters']['backbone_removal_state_reporters_filepath'] = os.path.join(state_data_reporter_dir, f"{self.config['info']['system_id']}_backbone_removal_state_data.csv")
        self.config['paths']['state_reporters']['nvt_state_reporters_filepath'] = os.path.join(state_data_reporter_dir, f"{self.config['info']['system_id']}_nvt_state_data.csv")
        self.config['paths']['state_reporters']['npt_state_reporters_filepath'] = os.path.join(state_data_reporter_dir, f"{self.config['info']['system_id']}_npt_state_data.csv")
        self.config['paths']['state_reporters']['production_state_reporters_filepath'] = os.path.join(state_data_reporter_dir, f"{self.config['info']['system_id']}_production_state_data.csv")
        
        # State Reporter Files
        state_reporter_dir = os.path.join(self.config['paths']['output_dir'], 'states')
        os.makedirs(state_reporter_dir, exist_ok=True)
        self.config['paths']['states'] = {}
        self.config['paths']['states']['warmup_state_filepath'] = os.path.join(state_reporter_dir, f"{self.config['info']['system_id']}_warmup_state.xml")
        self.config['paths']['states']['backbone_removal_state_filepath'] = os.path.join(state_reporter_dir, f"{self.config['info']['system_id']}_backbone_removal_state.xml")
        self.config['paths']['states']['nvt_state_filepath'] = os.path.join(state_reporter_dir, f"{self.config['info']['system_id']}_nvt_state.xml")
        self.config['paths']['states']['npt_state_filepath'] = os.path.join(state_reporter_dir, f"{self.config['info']['system_id']}_npt_state.xml")
        self.config['paths']['states']['production_state_filepath'] = os.path.join(state_reporter_dir, f"{self.config['info']['system_id']}_production_state.xml")

        # Topology Files
        topology_dir = os.path.join(self.config['paths']['output_dir'], 'topologies')
        os.makedirs(topology_dir, exist_ok=True)
        self.config['paths']['topologies'] = {}
        self.config['paths']['topologies']['warmup_topology_filepath'] = os.path.join(topology_dir, f"{self.config['info']['system_id']}_warmup_topology.cif")
        self.config['paths']['topologies']['backbone_removal_topology_filepath'] = os.path.join(topology_dir, f"{self.config['info']['system_id']}_backbone_removal_topology.cif")
        self.config['paths']['topologies']['nvt_topology_filepath'] = os.path.join(topology_dir, f"{self.config['info']['system_id']}_nvt_topology.cif")
        self.config['paths']['topologies']['npt_topology_filepath'] = os.path.join(topology_dir, f"{self.config['info']['system_id']}_npt_topology.cif")
        self.config['paths']['topologies']['production_topology_filepath'] = os.path.join(topology_dir, f"{self.config['info']['system_id']}_production_topology.cif")


    def set_ligand_info(self):
        """Set ligand information to config"""

        if self.config.get('ligand_info', None) is None:
            self.config['ligand_info'] = {}

        if self.config['ligand_info'].get('n_lig', None) is None:
            self.config['ligand_info']['n_lig'] = len(self.config['paths']['raw_ligand_files'])
        
        if self.config['ligand_info'].get('ligand_names', None) is None:
            self.config['ligand_info']['ligand_names'] = [os.path.splitext(os.path.basename(ligand_file))[0] for ligand_file in self.config['paths']['raw_ligand_files']]
        
        if self.config['ligand_info'].get('ligand_formats', None) is None:
            self.config['ligand_info']['ligand_formats'] = [os.path.splitext(os.path.basename(ligand_file))[1] for ligand_file in self.config['paths']['raw_ligand_files']]
        
        if self.config['ligand_info'].get('ligand_ccd_codes', None) is None:
            if self.plindex is not None:
                self.config['ligand_info']['ligand_ccd_codes'] = [self.plindex[self.plindex['ligand_id'].str.endswith(ligand_name)]['ligand_ccd_code'].values[0] for ligand_name in self.config['ligand_info']['ligand_names']]
            else:
                self.config['ligand_info']['ligand_ccd_codes'] = ['LIG' for ligand_name in self.config['ligand_info']['ligand_names']]
        
        if self.config['ligand_info'].get('ligand_charges', None) is None:
            self.config['ligand_info']['ligand_charges'] = {}


    def set_protein_info(self):
        """Set protein information to config"""
        if self.config.get('protein_info', None) is None:
            self.config['protein_info'] = {}
        
        if self.config['protein_info'].get('n_prot', None) is None:
            self.config['protein_info']['n_prot'] = len(self.config['paths']['raw_protein_files'])
        
        if self.config['protein_info'].get('protein_names', None) is None:
            self.config['protein_info']['protein_names'] = [os.path.splitext(os.path.basename(protein_file))[0] for protein_file in self.config['paths']['raw_protein_files']]
        
        if self.config['protein_info'].get('protein_formats', None) is None:
            self.config['protein_info']['protein_formats'] = [os.path.splitext(os.path.basename(protein_file))[1] for protein_file in self.config['paths']['raw_protein_files']]


    def set_simulation_info(self):
        """Set simulation information to config"""
        if self.config.get('simulation_params', None) is None:
            raise KeyError("Simulation information not found in config file!") 
        
        # Check Status of Simulation
        self.config['info']['simulation_status'] = {
            'warmup': 'Done' if os.path.exists(self.config['paths']['topologies']['warmup_topology_filepath']) else 'Not Done',
            'backbone_removal': 'Done' if os.path.exists(self.config['paths']['topologies']['backbone_removal_topology_filepath']) else 'Not Done',
            'nvt': 'Done' if os.path.exists(self.config['paths']['topologies']['nvt_topology_filepath']) else 'Not Done',
            'npt': 'Done' if os.path.exists(self.config['paths']['topologies']['npt_topology_filepath']) else 'Not Done',
            'production': 'Done' if os.path.exists(self.config['paths']['topologies']['production_topology_filepath']) else 'Not Done'
        }


    def update_simulation_status(self, stage: str, status: str):
        assert stage in self.config['info']['simulation_status'], f"Stage {stage} not found in config!"
        self.config['info']['simulation_status'][stage] = status if os.path.exists(self.config['paths']['topologies'][f'{stage}_topology_filepath']) else 'Not Done'


    def get_platform(self):
        self.platform = Platform.getPlatformByName(self.config['simulation_params']['platform']['type'])
        if self.platform.getName() == 'CUDA':
            self.platform.setPropertyDefaultValue('Precision', 'mixed')
            self.platform.setPropertyDefaultValue('CudaDeviceIndex', self.config['simulation_params']['platform']['devices'])


    def process_protein_with_pdbfixer(self, input_filepath: str, ph: float = 7.4):
        """Process the protein with PDBFixer"""
        fixer = pdbfixer.PDBFixer(input_filepath)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()  # find non-standard residue
        fixer.replaceNonstandardResidues()  # replace non-standard residues with standard one
        fixer.findMissingAtoms()  # find missing heavy atoms
        fixer.addMissingHydrogens(ph)
        fixer.addMissingAtoms()  # add missing atoms and residues
        
        output_filepath = os.path.join(self.config['paths']['output_dir'], os.path.basename(input_filepath).replace('.pdb', '_pdbfixer.pdb'))
        app.PDBFile.writeFile(fixer.topology, fixer.positions, open(output_filepath, 'w'))
        return fixer, output_filepath
        

    def process_protein_with_pdb2pqr(self, input_filepath: str, output_pdb_filepath: str, ph: float = 7.4):
        fix_biomolecule_with_pdb2pqr(receptor_pdb_filepath = input_filepath,
                                     output_pdb_filepath = output_pdb_filepath,
                                     ph = ph)
        return output_pdb_filepath


    def process_protein(self, input_filepath: str, ph: float = 7.4):
        output_filepath = os.path.join(self.config['paths']['output_dir'], os.path.basename(input_filepath).replace('.pdb', '_fixed.pdb'))
        if not os.path.exists(output_filepath):
            fixer, fixer_filepath = self.process_protein_with_pdbfixer(input_filepath, ph)
            output_filepath = self.process_protein_with_pdb2pqr(input_filepath=fixer_filepath, output_pdb_filepath=output_filepath, ph=ph)
        fixer = pdbfixer.PDBFixer(output_filepath)
        return fixer


    def process_ligand(self, input_filepath: str, input_format: str, input_name: str):
        # At this moment we only support SDF files
                   
        # Create OpenFF Ligand Model
        openff_molecule = load_molecule_to_openmm(input_filepath, input_format)

        if self.config['ligand_info']['ligand_charges'].get(input_name, None):
            openff_molecule.assign_partial_charges(partial_charge_method='gasteiger')
            openff_molecule.partial_charges.magnitude = np.array(self.config['ligand_info']['ligand_charges'][input_name])
        else:
            try:
                openff_molecule.assign_partial_charges(partial_charge_method="gasteiger", use_conformers=openff_molecule.conformers)
                self.config['ligand_info']['ligand_charges'][input_name] = {'charges': openff_molecule.partial_charges.magnitude.tolist(), 'method': 'gasteiger'}
            except:
                openff_molecule.assign_partial_charges(partial_charge_method="gasteiger")
                self.config['ligand_info']['ligand_charges'][input_name] = {'charges': openff_molecule.partial_charges.magnitude.tolist(), 'method': 'gasteiger'}

        openff_molecule_topology = openff_molecule.to_topology()
        openff_molecule_positions = openff_molecule.conformers[0].to("nanometers")

        # Create OpenMM Ligand Model
        openmm_molecule_topology = openff_molecule_topology.to_openmm()
        openmm_molecule_positions = openff_molecule_positions.to_openmm()
        openmm_molecule = Modeller(openmm_molecule_topology, openmm_molecule_positions)
            
        return openmm_molecule, openff_molecule


    def process_complex(self):
        # Load the complex structure        
        if self.config['preprocessing']['process_protein'] == True:
            for idx, protein_filepath in enumerate(self.config['paths']['raw_protein_files']):
                protein_pdbfixer = self.process_protein(protein_filepath)
                if idx == 0:
                    complex = Modeller(protein_pdbfixer.topology, protein_pdbfixer.positions)
                else:
                    complex.add(protein_pdbfixer.topology, protein_pdbfixer.positions) 
        else:
            complex = app.pdbxfile.PDBxFile(self.config['paths']['init_topology_filepath'])
        
        openff_molecules = []        
        if self.config['preprocessing']['process_ligand'] == True:
            chain_idx = len(complex.topology._chains)
            for ligand_filepath, ligand_format, ligand_name, ligand_ccd in zip(self.config['paths']['raw_ligand_files'], 
                                                                  self.config['ligand_info']['ligand_formats'],
                                                                  self.config['ligand_info']['ligand_names'], 
                                                                  self.config['ligand_info']['ligand_ccd_codes']):
                openmm_molecule, openff_molecule = self.process_ligand(ligand_filepath, ligand_format, ligand_name)
                openmm_molecule.name = ligand_ccd
                openff_molecule.name = ligand_ccd
                openff_molecules.append(openff_molecule)
                openmm_molecule.topology.id = ligand_ccd
                complex.add(openmm_molecule.topology, openmm_molecule.positions)
                complex.topology._chains[chain_idx].id = ligand_ccd
                for res_idx in range(len(complex.topology._chains[chain_idx]._residues)):
                    complex.topology._chains[chain_idx]._residues[res_idx].name = ligand_ccd
                    
                    new_atom_values = defaultdict(int)
                    for atom_idx, atom in enumerate(complex.topology._chains[chain_idx]._residues[res_idx]._atoms):
                        if len(atom.name) == 0:
                            if atom.element.symbol not in new_atom_values:
                                new_atom_values[atom.element.symbol] = 1
                            else:
                                new_atom_values[atom.element.symbol] += 1
                            
                            complex.topology._chains[chain_idx]._residues[res_idx]._atoms[atom_idx].name = atom.element.symbol + str(new_atom_values[atom.element.symbol])
                            a=1
                chain_idx += 1

        return complex, openff_molecules


    def set_system(self):
        if all([os.path.exists(self.config['paths']['init_system_filepath']),
                os.path.exists(self.config['paths']['init_system_with_posres_filepath']),
                os.path.exists(self.config['paths']['init_topology_filepath'])]):
            if self.config['paths']['init_topology_filepath'].endswith('.cif'):
                complex = app.pdbxfile.PDBxFile(self.config['paths']['init_topology_filepath'])
            elif self.config['paths']['init_topology_filepath'].endswith('.pdb'):
                complex = app.pdbfile.PDBFile(self.config['paths']['init_topology_filepath'])
            complex = Modeller(complex.topology, complex.positions)
            system = XmlSerializer.deserialize(open(self.config['paths']['init_system_filepath']).read())
            system_with_posres = XmlSerializer.deserialize(open(self.config['paths']['init_system_with_posres_filepath']).read())
        else:
            complex, openff_molecules = self.process_complex()
            system_generator = SystemGenerator(
                forcefields = [self.config['forcefield']['proteinFF'], 
                               self.config['forcefield']['nucleicFF'], 
                               self.config['forcefield']['waterFF']],
                small_molecule_forcefield = self.config['forcefield']['ligandFF'],
                molecules = openff_molecules,
                forcefield_kwargs = {
                    'constraints': self.config['forcefield']['forcefield_kwargs'].get('constraints', None),
                    'rigidWater': self.config['forcefield']['forcefield_kwargs'].get('rigidWater', True),
                    'removeCMMotion': self.config['forcefield']['forcefield_kwargs'].get('removeCMMotion', False),
                    'hydrogenMass': self.config['forcefield']['forcefield_kwargs'].get('hydrogenMass') * unit.amu
                }
                )
            if self.config['preprocessing'].get('add_solvate', True):
                complex.addSolvent(
                    system_generator.forcefield,
                    model = self.config['forcefield']['water_model'],
                    padding = self.config['preprocessing']['box_padding'] * unit.nanometers,
                    ionicStrength = self.config['preprocessing']['ionic_strength'] * unit.molar,
                )
                a=1
                with open(self.config['paths']['init_topology_filepath'], 'w') as outfile:
                    app.PDBxFile.writeFile(complex.topology, complex.positions, outfile)
            
                system = system_generator.create_system(complex.topology, molecules = openff_molecules)
                with open(self.config['paths']['init_system_filepath'], 'w') as outfile:
                    outfile.write(XmlSerializer.serialize(system))

                system_with_posres = add_backbone_posres(system, complex.positions, complex.topology.atoms(), self.config['simulation_params']['backbone_restraint_force'])
                with open(self.config['paths']['init_system_with_posres_filepath'], 'w') as outfile:
                    outfile.write(XmlSerializer.serialize(system_with_posres))

        self.model = complex
        self.system = system
        self.system_with_posres = system_with_posres


    def set_reporters(self, simulation, 
                      checkpoint_filepath: str,
                      trajectory_filepath: str,
                      state_data_reporter_filepath: str,
                      checkpoint_interval: str,
                      trajectory_interval: str,
                      state_data_reporter_interval: str,
                      total_steps: str,
                      ):        
        # Set CheckPoint Reporter
        simulation.reporters.append(
            app.checkpointreporter.CheckpointReporter(
                file=checkpoint_filepath,
                reportInterval = checkpoint_interval
            )
        )

        # Set Trajectory Reporter
        simulation.reporters.append(
            app.xtcreporter.XTCReporter(
                file = trajectory_filepath,
                reportInterval = trajectory_interval,
                enforcePeriodicBox = False,
                append = os.path.exists(trajectory_filepath)
            )
        )

        # Add State Reporter
        simulation.reporters.append(
            app.StateDataReporter(
                file = state_data_reporter_filepath,
                reportInterval = state_data_reporter_interval,
                step=True,
                potentialEnergy=True,
                kineticEnergy=True,
                temperature=True,
                volume=True,
                progress=True,
                remainingTime=True,
                speed=True,
                totalSteps = total_steps,
                separator="\t",
                append=os.path.exists(state_data_reporter_filepath)
            )
        )
        return simulation


    def warmup(self):
        """Run warmup simulation with gradual temperature increase.
        
        This function performs a warmup simulation with backbone position restraints,
        gradually increasing the temperature from initial to final temperature.
        The simulation includes:
        - Energy minimization
        - Gradual heating in small temperature increments
        - Regular state and checkpoint saving
        
        The simulation uses position restraints on the protein backbone to prevent
        large conformational changes during the heating process.
        """

        log_info(logger, "Starting warmup simulation with gradual heating")
        # Log all warmup parameters
        warmup_params = {
            'initial_temperature': f"{self.config['simulation_params']['warmup']['init_temp']}K",
            'final_temperature': f"{self.config['simulation_params']['warmup']['final_temp']}K",
            'friction': f"{self.config['simulation_params']['warmup']['friction']}/ps",
            'timestep': f"{self.config['simulation_params']['warmup']['time_step']}fs",
            'heating_step': self.config['simulation_params']['warmup']['heating_step'],
            'checkpoint_interval': self.config['simulation_params']['warmup']['checkpoint_interval'],
            'trajectory_interval': self.config['simulation_params']['warmup']['trajectory_interval'],
            'state_data_interval': self.config['simulation_params']['warmup']['state_data_reporter_interval']
        }
        log_info(logger, "WarmUp simulation parameters:")
        for param, value in warmup_params.items():
            log_info(logger, f"  {param}: {value}")
        
        try:
            # Initialize Integrator and Simulation
            integrator = LangevinIntegrator(self.config['simulation_params']['warmup']['init_temp'] * unit.kelvin,
                                            self.config['simulation_params']['warmup']['friction'] / unit.picoseconds,
                                            self.config['simulation_params']['warmup']['time_step'] * unit.femtoseconds) 
            log_debug(logger, f"Initialized Langevin integrator with initial temperature: {self.config['simulation_params']['warmup']['init_temp']}K")
                    
            warmup_simulation = app.Simulation(self.model.topology, self.system_with_posres, integrator, platform=self.platform)
            log_debug(logger, "Created warmup simulation context with position restraints")
            
            heating_steps = (self.config['simulation_params']['warmup']['final_temp'] - self.config['simulation_params']['warmup']['init_temp']) * self.config['simulation_params']['warmup']['heating_step']
            log_debug(logger, f"Calculated total heating steps: {heating_steps}")
            
            # Load Checkpoint if it exists
            if os.path.exists(self.config['paths']['checkpoints']['warmup_checkpoint_filepath']):
                warmup_simulation.loadCheckpoint(self.config['paths']['checkpoints']['warmup_checkpoint_filepath'])
                nsteps_done = warmup_simulation.context.getStepCount() 
                heatup_loops = self.config['simulation_params']['warmup']['final_temp'] - self.config['simulation_params']['warmup']['init_temp']
                heatup_loops_done = nsteps_done / self.config['simulation_params']['warmup']['heating_step'] 
                heatup_loops_to_run = heatup_loops - heatup_loops_done
                initial_temperature = heatup_loops_done + self.config['simulation_params']['warmup']['init_temp']
                log_info(logger, f"Loaded warmup checkpoint, continuing from step {nsteps_done} at temperature {initial_temperature}K")
            else:
                warmup_simulation.context.setPositions(self.model.positions)
                log_debug(logger, "Starting energy minimization")
                warmup_simulation.minimizeEnergy()
                log_info(logger, "Completed energy minimization")
                heatup_loops = self.config['simulation_params']['warmup']['final_temp'] - self.config['simulation_params']['warmup']['init_temp']
                heatup_loops_done = 0
                heatup_loops_to_run = heatup_loops
                initial_temperature = self.config['simulation_params']['warmup']['init_temp']
                log_info(logger, f"Starting fresh warmup from {initial_temperature}K")
            
            # Set Reporters
            warmup_simulation = self.set_reporters(
                simulation = warmup_simulation,
                checkpoint_filepath = self.config['paths']['checkpoints']['warmup_checkpoint_filepath'],
                trajectory_filepath = self.config['paths']['trajectories']['warmup_trajectory_filepath'],
                state_data_reporter_filepath = self.config['paths']['state_reporters']['warmup_state_reporters_filepath'],
                checkpoint_interval = self.config['simulation_params']['warmup']['checkpoint_interval'],
                trajectory_interval = self.config['simulation_params']['warmup']['trajectory_interval'],
                state_data_reporter_interval = self.config['simulation_params']['warmup']['state_data_reporter_interval'],
                total_steps = heating_steps,
            )
            log_debug(logger, "Set up warmup simulation reporters")
            
            # Heating Loop
            warmup_simulation.context.setVelocitiesToTemperature(initial_temperature * unit.kelvin)
            log_info(logger, f"Starting heating loop with {heatup_loops_to_run} temperature increments")
            
            for i in tqdm(range(int(heatup_loops_to_run)), desc="Heating...", total=int(heatup_loops_to_run)):
                warmup_simulation.step(self.config['simulation_params']['warmup']['heating_step'])
                current_temperature = (initial_temperature + (i + int(heatup_loops_done))) * unit.kelvin
                integrator.setTemperature(current_temperature)
                log_debug(logger, f"Temperature increased to {current_temperature}")
                
                warmup_simulation.saveState(self.config['paths']['states']['warmup_state_filepath'])
                warmup_simulation.saveCheckpoint(self.config['paths']['checkpoints']['warmup_checkpoint_filepath'])
                log_debug(logger, f"Saved state and checkpoint at temperature {current_temperature}")
            
            with open(self.config['paths']['topologies']['warmup_topology_filepath'], "w") as cif_file:
                app.PDBxFile.writeFile(
                    warmup_simulation.topology,
                    warmup_simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(),
                    file=cif_file,
                    keepIds=True,
                )
            log_debug(logger, f"Saved final warmup topology to {self.config['paths']['topologies']['warmup_topology_filepath']}")
            
            self.update_simulation_status('warmup', 'Done')
            log_info(logger, f"Warmup phase completed successfully, reached final temperature {self.config['simulation_params']['warmup']['final_temp']}K")
            
        except Exception as e:
            error_msg = f"Error during warmup simulation: {str(e)}"
            log_error(logger, error_msg)
            raise

    def remove_backbone_constraints(self):
        """Gradually remove backbone position restraints from the system.
        
        This function performs a simulation where the backbone position restraints
        are gradually reduced to zero. The process involves:
        - Starting with strong position restraints (99.02 kcal/mol/Å²)
        - Gradually reducing the restraint force in small increments (0.98 kcal/mol/Å² per loop)
        - Running multiple loops until restraints are completely removed
        - Regular state and checkpoint saving
        
        This gradual removal helps the system adjust smoothly from the restrained
        to the fully flexible state, preventing sudden structural changes.
        """
        log_info(logger, "Starting backbone restraints removal phase")
        
        # Log all backbone removal parameters
        backbone_params = {
            'temperature': f"{self.config['simulation_params']['backbone_removal']['temp']}K",
            'friction': f"{self.config['simulation_params']['backbone_removal']['friction']}/ps",
            'timestep': f"{self.config['simulation_params']['backbone_removal']['time_step']}fs",
            'total_steps': self.config['simulation_params']['backbone_removal']['nsteps'],
            'number_of_loops': self.config['simulation_params']['backbone_removal']['nloops'],
            'initial_restraint': "99.02 kcal/mol/Å²",
            'restraint_decrement': "0.98 kcal/mol/Å²",
            'checkpoint_interval': self.config['simulation_params']['backbone_removal']['checkpoint_interval'],
            'trajectory_interval': self.config['simulation_params']['backbone_removal']['trajectory_interval'],
            'state_data_interval': self.config['simulation_params']['backbone_removal']['state_data_reporter_interval']
        }
        
        log_info(logger, "Backbone Restraints Removal parameters:")
        for param, value in backbone_params.items():
            log_info(logger, f"  {param}: {value}")

        try:
            # Initialize Integrator and Simulation
            integrator = LangevinIntegrator(self.config['simulation_params']['backbone_removal']['temp'] * unit.kelvin,
                                            self.config['simulation_params']['backbone_removal']['friction'] / unit.picoseconds,
                                            self.config['simulation_params']['backbone_removal']['time_step'] * unit.femtoseconds)
            log_debug(logger, f"Initialized Langevin integrator with temperature: {self.config['simulation_params']['backbone_removal']['temp']}K")
            
            backbone_simulation = app.Simulation(self.model.topology,
                                                 self.system_with_posres,
                                                 integrator,
                                                 platform=self.platform)
            log_debug(logger, "Created backbone restraints removal simulation context")
            
            if os.path.exists(self.config['paths']['checkpoints']['backbone_removal_checkpoint_filepath']):
                backbone_simulation.loadCheckpoint(self.config['paths']['checkpoints']['backbone_removal_checkpoint_filepath'])
                nsteps_done = backbone_simulation.context.getStepCount()
                br_nsteps_to_run = self.config['simulation_params']['backbone_removal']['nsteps'] - nsteps_done
                log_info(logger, f"Loaded backbone removal checkpoint, continuing from step {nsteps_done}")
            elif os.path.exists(self.config['paths']['checkpoints']['warmup_checkpoint_filepath']):
                backbone_simulation.loadCheckpoint(self.config['paths']['checkpoints']['warmup_checkpoint_filepath'])
                backbone_simulation.context.setStepCount(0)
                br_nsteps_done = 0
                br_nsteps_to_run = self.config['simulation_params']['backbone_removal']['nsteps']
                log_info(logger, "Loaded warmup checkpoint for backbone removal phase")
            else:
                backbone_simulation.context.setStepCount(0)
                br_nsteps_done = 0
                br_nsteps_to_run = self.config['simulation_params']['backbone_removal']['nsteps']
                log_info(logger, "Starting backbone removal phase from scratch")

            br_steps_in_loop = self.config['simulation_params']['backbone_removal']['nsteps'] / self.config['simulation_params']['backbone_removal']['nloops']
            br_loops_to_run = self.config['simulation_params']['backbone_removal']['nloops'] - (br_nsteps_done / br_steps_in_loop)
            br_loops_done = self.config['simulation_params']['backbone_removal']['nloops'] - br_loops_to_run
            log_debug(logger, f"Calculated {br_loops_to_run} loops remaining, {br_steps_in_loop} steps per loop")

            backbone_simulation = self.set_reporters(
                simulation = backbone_simulation,
                checkpoint_filepath = self.config['paths']['checkpoints']['backbone_removal_checkpoint_filepath'],
                trajectory_filepath = self.config['paths']['trajectories']['backbone_removal_trajectory_filepath'],
                state_data_reporter_filepath = self.config['paths']['state_reporters']['backbone_removal_state_reporters_filepath'],
                checkpoint_interval = self.config['simulation_params']['backbone_removal']['checkpoint_interval'],
                trajectory_interval = self.config['simulation_params']['backbone_removal']['trajectory_interval'],
                state_data_reporter_interval = self.config['simulation_params']['backbone_removal']['state_data_reporter_interval'],
                total_steps = self.config['simulation_params']['backbone_removal']['nsteps'],
            )
            log_debug(logger, "Set up backbone removal simulation reporters")

            state = backbone_simulation.context.getState(getVelocities=True, getPositions=True)
            positions = state.getPositions()
            velocities = state.getVelocities()
            backbone_simulation.context.setPositions(positions)
            backbone_simulation.context.setVelocities(velocities)
            backbone_simulation.context.setVelocitiesToTemperature(self.config['simulation_params']['backbone_removal']['temp'] * unit.kelvin)
            log_debug(logger, "Set positions and velocities")

            initial_restraint = float(99.02 - (int(br_loops_done) * 0.98))
            backbone_simulation.context.setParameter('k', (initial_restraint * unit.kilocalories_per_mole / unit.angstroms ** 2))
            log_info(logger, f"Starting restraint removal loops with initial force constant: {initial_restraint} kcal/mol/Å²")

            for i in tqdm(range(int(br_loops_to_run)), desc="Removing Backbone Constraints...", total=int(br_loops_to_run)):
                backbone_simulation.step(br_steps_in_loop)
                current_restraint = float(99.02 - ((i + br_loops_done) * 0.98))
                backbone_simulation.context.setParameter('k', (current_restraint * unit.kilocalories_per_mole / unit.angstroms ** 2))
                log_debug(logger, f"Reduced restraint force constant to {current_restraint} kcal/mol/Å²")

            backbone_simulation.context.setParameter('k', 0)
            log_info(logger, "Restraints completely removed (force constant = 0)")
            
            backbone_simulation.saveState(self.config['paths']['states']['backbone_removal_state_filepath'])
            log_debug(logger, f"Saved final state to {self.config['paths']['states']['backbone_removal_state_filepath']}")
            
            with open(self.config['paths']['topologies']['backbone_removal_topology_filepath'], "w") as cif_file:
                app.PDBxFile.writeFile(
                    backbone_simulation.topology,
                    backbone_simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(),
                    file=cif_file,
                    keepIds=True,
                )
            log_debug(logger, f"Saved final topology to {self.config['paths']['topologies']['backbone_removal_topology_filepath']}")
            
            self.update_simulation_status('backbone_removal', 'Done')
            log_info(logger, "Backbone restraints removal phase completed successfully")

        except Exception as e:
            error_msg = f"Error during backbone restraints removal: {str(e)}"
            log_error(logger, error_msg)
            raise

    def nvt(self):
        integrator = LangevinIntegrator(
            self.config['simulation_params']['nvt']['temp'] * unit.kelvin, 
            self.config['simulation_params']['nvt']['friction'] / unit.picoseconds,
            self.config['simulation_params']['nvt']['time_step'] * unit.femtoseconds
            )

        nvt_simulation = app.Simulation(
            self.model.topology,
            self.system,
            integrator,
            platform=self.platform)
        
        if os.path.exists(self.config['paths']['checkpoints']['nvt_checkpoint_filepath']):
            # Load Backbone Checkpoint if possible
            nvt_simulation.loadCheckpoint(self.config['paths']['checkpoints']['nvt_checkpoint_filepath'])
            steps_done = nvt_simulation.context.getStepCount()
            steps_to_run = self.config['simulation_params']['nvt']['nsteps'] - steps_done
        elif os.path.exists(self.config['paths']['checkpoints']['warmup_checkpoint_filepath']):
            # Load Warmed Up Checkpoint if possible
            nvt_simulation.loadCheckpoint(self.config['paths']['checkpoints']['warmup_checkpoint_filepath'])
            nvt_simulation.context.setStepCount(0)
            steps_to_run = self.config['simulation_params']['nvt']['nsteps']
        else:
            nvt_simulation.context.setStepCount(0)
            steps_to_run = self.config['simulation_params']['nvt']['nsteps']
        
                # Set Reporters
        nvt_simulation = self.set_reporters(
            simulation = nvt_simulation,
            checkpoint_filepath = self.config['paths']['checkpoints']['nvt_checkpoint_filepath'],
            trajectory_filepath = self.config['paths']['trajectories']['nvt_trajectory_filepath'],
            state_data_reporter_filepath = self.config['paths']['state_reporters']['nvt_state_reporters_filepath'],
            checkpoint_interval = self.config['simulation_params']['nvt']['checkpoint_interval'],
            trajectory_interval = self.config['simulation_params']['nvt']['trajectory_interval'],
            state_data_reporter_interval = self.config['simulation_params']['nvt']['state_data_reporter_interval'],
            total_steps = self.config['simulation_params']['nvt']['nsteps'],
        )

        state = nvt_simulation.context.getState(getVelocities=True, getPositions=True)

        positions = state.getPositions()
        velocities = state.getVelocities()
        nvt_simulation.context.setPositions(positions)
        nvt_simulation.context.setVelocities(velocities)

        nvt_simulation.context.setVelocitiesToTemperature(self.config['simulation_params']['nvt']['temp'] * unit.kelvin)
        nvt_simulation.context.setParameter('k', 0)
        nvt_simulation.step(steps_to_run)
        
        nvt_simulation.saveState(self.config['paths']['states']['nvt_state_filepath'])
        
        with open(self.config['paths']['topologies']['nvt_topology_filepath'], "w") as cif_file:
            app.PDBxFile.writeFile(
                nvt_simulation.topology,
                nvt_simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(),
                file=cif_file,
                keepIds=True,
            )

    def npt(self):
        integrator = LangevinIntegrator(
            self.config['simulation_params']['npt']['temp'] * unit.kelvin,
            self.config['simulation_params']['npt']['friction'] / unit.picoseconds,
            self.config['simulation_params']['npt']['time_step'] * unit.femtoseconds
        )
        
        self.system.addForce(MonteCarloBarostat(self.config['simulation_params']['npt']['pressure'] * unit.atmospheres, self.config['simulation_params']['npt']['temp'] * unit.kelvin))
        npt_simulation = app.Simulation(
            self.model.topology,
            self.system,
            integrator,
            platform=self.platform)
        
        if os.path.exists(self.config['paths']['checkpoints']['npt_checkpoint_filepath']):
            npt_simulation.loadCheckpoint(self.config['paths']['checkpoints']['npt_checkpoint_filepath'])
            steps_done = npt_simulation.context.getStepCount()
            steps_to_run = self.config['simulation_params']['npt']['nsteps'] - steps_done
        elif os.path.exists(self.config['paths']['checkpoints']['nvt_checkpoint_filepath']):
            npt_simulation.loadCheckpoint(self.config['paths']['checkpoints']['nvt_checkpoint_filepath'])
            npt_simulation.context.setStepCount(0)
            steps_to_run = self.config['simulation_params']['npt']['nsteps']
        else:
            npt_simulation.context.setStepCount(0)
            steps_to_run = self.config['simulation_params']['npt']['nsteps']
        
        npt_simulation = self.set_reporters(
            simulation = npt_simulation,
            checkpoint_filepath = self.config['paths']['checkpoints']['npt_checkpoint_filepath'],
            trajectory_filepath = self.config['paths']['trajectories']['npt_trajectory_filepath'],
            state_data_reporter_filepath = self.config['paths']['state_reporters']['npt_state_reporters_filepath'],
            checkpoint_interval = self.config['simulation_params']['npt']['checkpoint_interval'],
            trajectory_interval = self.config['simulation_params']['npt']['trajectory_interval'],
            state_data_reporter_interval = self.config['simulation_params']['npt']['state_data_reporter_interval'],
            total_steps = self.config['simulation_params']['npt']['nsteps'],
        )
        
        state = npt_simulation.context.getState(getVelocities=True, getPositions=True)
        positions = state.getPositions()
        velocities = state.getVelocities()
        npt_simulation.context.setPositions(positions)
        npt_simulation.context.setVelocities(velocities)
        npt_simulation.context.setVelocitiesToTemperature(self.config['simulation_params']['npt']['temp'] * unit.kelvin)
        npt_simulation.context.setParameter('k', 0)
        npt_simulation.step(steps_to_run)

        npt_simulation.saveState(self.config['paths']['states']['npt_state_filepath'])   
        
        with open(self.config['paths']['topologies']['npt_topology_filepath'], "w") as cif_file:
            app.PDBxFile.writeFile(
                npt_simulation.topology,
                npt_simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(),
                file=cif_file,
                keepIds=True,
            )
    
    def production(self):
        integrator = LangevinIntegrator(
            self.config['simulation_params']['production']['temp'] * unit.kelvin,
            self.config['simulation_params']['production']['friction'] / unit.picoseconds,
            self.config['simulation_params']['production']['time_step'] * unit.femtoseconds
        )

        self.system.addForce(MonteCarloBarostat(self.config['simulation_params']['production']['pressure'] * unit.atmospheres, self.config['simulation_params']['production']['temp'] * unit.kelvin))
        
        production_simulation = app.Simulation(
            self.model.topology,
            self.system,
            integrator,
            platform=self.platform
        )

        if os.path.exists(self.config['paths']['checkpoints']['production_checkpoint_filepath']):
            production_simulation.loadCheckpoint(self.config['paths']['checkpoints']['production_checkpoint_filepath'])
            steps_done = production_simulation.context.getStepCount()
            steps_to_run = self.config['simulation_params']['production']['nsteps'] - steps_done
        elif os.path.exists(self.config['paths']['checkpoints']['npt_checkpoint_filepath']):
            production_simulation.loadCheckpoint(self.config['paths']['checkpoints']['npt_checkpoint_filepath'])
            production_simulation.context.setStepCount(0)
            steps_to_run = self.config['simulation_params']['production']['nsteps']
        else:
            production_simulation.context.setStepCount(0)
            steps_to_run = self.config['simulation_params']['production']['nsteps']

        production_simulation = self.set_reporters(
            simulation = production_simulation,
            checkpoint_filepath = self.config['paths']['checkpoints']['production_checkpoint_filepath'],
            trajectory_filepath = self.config['paths']['trajectories']['production_trajectory_filepath'],
            state_data_reporter_filepath = self.config['paths']['state_reporters']['production_state_reporters_filepath'],
            checkpoint_interval = self.config['simulation_params']['production']['checkpoint_interval'],
            trajectory_interval = self.config['simulation_params']['production']['trajectory_interval'],
            state_data_reporter_interval = self.config['simulation_params']['production']['state_data_reporter_interval'],
            total_steps = self.config['simulation_params']['production']['nsteps'],
        )
        
        state = production_simulation.context.getState(getVelocities=True, getPositions=True)
        positions = state.getPositions()
        velocities = state.getVelocities()
        production_simulation.context.setPositions(positions)
        production_simulation.context.setVelocities(velocities)
        production_simulation.context.setVelocitiesToTemperature(self.config['simulation_params']['production']['temp'] * unit.kelvin)
        production_simulation.context.setParameter('k', 0)
        production_simulation.step(steps_to_run)
        
        production_simulation.saveState(self.config['paths']['states']['production_state_filepath'])
        
        with open(self.config['paths']['topologies']['production_topology_filepath'], "w") as cif_file:
            app.PDBxFile.writeFile(
                production_simulation.topology,
                production_simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(),
                file=cif_file,
                keepIds=True,
            )