import os
from openmmplumed import PlumedForce
from molecular_dynamics_pipeline.simulation.simulation import MDSimulation
from openmm import XmlSerializer, app, unit
from openmm.app.modeller import Modeller
from openmmforcefields.generators import SystemGenerator
from biotite.structure.io.pdbx import CIFFile, get_structure
from molecular_dynamics_pipeline.simulation.simulation import add_backbone_posres
import numpy as np
from scipy.spatial import cKDTree

class MetaMDSimulation(MDSimulation):
    def __init__(self, config):
        super().__init__(config)

    def set_metadynamics_params(self):

        if self.config['metadynamics'].get('pocket_distance') == None:
            self.conifg['metadynamics']['pokcet_distance'] = 4.0
            
        if self.config['metadynamics'].get('sigma') == None:
            self.config['metadynamics']['sigma'] = 0.2

        if self.config['metadynamics'].get('height') == None:
            self.config['metadynamics']['height'] = 0.3
            
        if self.config['metadynamics'].get('pace') == None:
            self.config['metadynamics']['pace'] = 500

    def get_protein_ligand_atom_pairs(self):
        """
        This function gets the protein and ligand atom pairs from the complex.
        It is used to create a metadynamics script for the simulation.
        """

        complex_file = CIFFile.read(self.config['paths']['init_topology_filepath'])
        complex_structure = get_structure(complex_file)

        complex_coords = complex_structure.coord
        pocket_distance = self.config['metadynamics']['pocket_distance']

        ligand_indices = np.where(np.isin(complex_structure.res_name, self.config['ligand_info']['ligand_names']))[0]
        
        tree = cKDTree(complex_coords[0])
        pockets = tree.query_ball_point(coords[ligand_indices], r = pocket_distance)
        pocket_indices = set([i for sublist in pockets for i in sublist])
        pocket_indices.difference_update(ligand_indices)
        pocket_indices = np.array(sorted(pocket_indices))
        
        pairs = ""
        pair_index = 1
        for ligand_index in ligand_indices:
            for pocket_index in pocket_indices:
                pairs += f"d{str(pair_index)}: DISTANCE ATOMS={str(ligand_index+1)},{str(pocket_index+1)}\n"
        return pairs
        
    def create_metadynamics_script(self, complex):
        """""
        This script creates a metadynamics script for the simulation.
        It is used to create a metadynamics potential for the simulation.
        The script is created based on the following parameters:
        - DISTANCE ATOMS: The atoms to calculate the distance between.
        - METAD ARG: The argument for the metadynamics potential.
        - SIGMA: The sigma for the metadynamics potential.
        """

        protein_ligand_atom_pairs = self.get_protein_ligand_atom_pairs()
        
        sigma = self.config['metadynamics']['sigma']
        height = self.config['metadynamics']['height']
        pace = self.config['metadynamics']['pace']

        metadynamics_script = pairs + f"""
        METAD ARG=d SIGMA={sigma} HEIGHT={height} PACE={pace}"""

        return metadynamics_script

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
                with open(self.config['paths']['init_topology_filepath'].replace('.cif', '.pdb'), 'w') as outfile:
                    app.PDBFile.writeFile(complex.topology, complex.positions, outfile)
            
                system = system_generator.create_system(complex.topology, molecules = openff_molecules)
                with open(self.config['paths']['init_system_filepath'], 'w') as outfile:
                    outfile.write(XmlSerializer.serialize(system))

                system_with_posres = add_backbone_posres(system, complex.positions, complex.topology.atoms(), self.config['simulation_params']['backbone_restraint_force'])
                with open(self.config['paths']['init_system_with_posres_filepath'], 'w') as outfile:
                    outfile.write(XmlSerializer.serialize(system_with_posres))

        self.model = complex

        metadynamics_script = self.create_metadynamics_script(complex)

        self.system = system
        self.system_with_posres = system_with_posres
        