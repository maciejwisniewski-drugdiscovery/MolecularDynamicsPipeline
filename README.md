# Molecular Dynamics Simulation Pipeline

A comprehensive molecular dynamics simulation pipeline built with OpenMM, supporting protein-ligand systems with multiple stages of simulation including warmup, backbone constraint removal, NVT, NPT, and production runs.

## Features

- Multi-stage molecular dynamics simulation pipeline
- Support for protein-ligand systems
- Automatic system preparation and solvation
- Configurable force fields and simulation parameters
- Checkpoint-based simulation recovery
- Extensive logging and monitoring
- Support for multiple GPUs

## Prerequisites

- Python 3.7+
- OpenMM
- OpenFF Toolkit
- RDKit
- OpenBabel
- pdb2pqr
- PDBFixer

## Input Requirements

### Structure Files

1. **Protein Files**:
   - Supported formats: PDB, CIF
   - Can handle multiple protein chains
   - Files should be specified in the configuration under `raw_protein_files`

2. **Ligand Files**:
   - Supported formats: SDF, MOL2
   - Can handle multiple ligands
   - Files should be specified in the configuration under `raw_ligand_files`

### Configuration Files

The pipeline requires YAML configuration files with the following structure:

```yaml
info:
  simulation_id: "unique_simulation_id"
  system_id: "system_identifier"
  use_plinder_index: false  # Optional

paths:
  raw_protein_files: ["path/to/protein.pdb"]
  raw_ligand_files: ["path/to/ligand.sdf"]
  output_dir: "path/to/output"

preprocessing:
  process_protein: true
  process_ligand: true
  add_solvate: true
  box_padding: 1.0  # nm
  ionic_strength: 0.15  # M

forcefield:
  proteinFF: "amber14/protein.ff14SB.xml"
  nucleicFF: "amber14/DNA.OL15.xml"
  waterFF: "amber14/tip3p.xml"
  ligandFF: "openff-2.0.0"
  water_model: "tip3p"
  forcefield_kwargs:
    constraints: "HBonds"
    rigidWater: true
    removeCMMotion: false
    hydrogenMass: 1.5

simulation_params:
  platform:
    type: "CUDA"
    devices: "0"
  backbone_restraint_force: 100.0  # kcal/mol/Å²

  warmup:
    init_temp: 0
    final_temp: 300
    friction: 1.0
    time_step: 2.0
    heating_step: 1000
    checkpoint_interval: 1000
    trajectory_interval: 1000
    state_data_reporter_interval: 1000

  backbone_removal:
    temp: 300
    friction: 1.0
    time_step: 2.0
    nsteps: 100000
    nloops: 100
    checkpoint_interval: 1000
    trajectory_interval: 1000
    state_data_reporter_interval: 1000

  nvt:
    temp: 300
    friction: 1.0
    time_step: 2.0
    nsteps: 100000
    checkpoint_interval: 1000
    trajectory_interval: 1000
    state_data_reporter_interval: 1000

  npt:
    temp: 300
    friction: 1.0
    time_step: 2.0
    pressure: 1.0
    nsteps: 100000
    checkpoint_interval: 1000
    trajectory_interval: 1000
    state_data_reporter_interval: 1000

  production:
    temp: 300
    friction: 1.0
    time_step: 2.0
    pressure: 1.0
    nsteps: 1000000
    checkpoint_interval: 1000
    trajectory_interval: 1000
    state_data_reporter_interval: 1000
```

## Output Directory Structure

```
output_dir/
├── checkpoints/
│   ├── {system_id}_warmup_checkpoint.dcd
│   ├── {system_id}_backbone_removal_checkpoint.dcd
│   ├── {system_id}_nvt_checkpoint.dcd
│   ├── {system_id}_npt_checkpoint.dcd
│   └── {system_id}_production_checkpoint.dcd
├── trajectories/
│   ├── {system_id}_warmup_trajectory.xtc
│   ├── {system_id}_backbone_removal_trajectory.xtc
│   ├── {system_id}_nvt_trajectory.xtc
│   ├── {system_id}_npt_trajectory.xtc
│   └── {system_id}_production_trajectory.xtc
├── state_data_reporters/
│   ├── {system_id}_warmup_state_data.csv
│   ├── {system_id}_backbone_removal_state_data.csv
│   ├── {system_id}_nvt_state_data.csv
│   ├── {system_id}_npt_state_data.csv
│   └── {system_id}_production_state_data.csv
├── states/
│   ├── {system_id}_warmup_state.xml
│   ├── {system_id}_backbone_removal_state.xml
│   ├── {system_id}_nvt_state.xml
│   ├── {system_id}_npt_state.xml
│   └── {system_id}_production_state.xml
├── topologies/
│   ├── {system_id}_warmup_topology.cif
│   ├── {system_id}_backbone_removal_topology.cif
│   ├── {system_id}_nvt_topology.cif
│   ├── {system_id}_npt_topology.cif
│   └── {system_id}_production_topology.cif
├── {system_id}_init_complex.cif
├── {system_id}_init_topology.cif
├── {system_id}_init_system.xml
└── {system_id}_init_system_with_posres.xml
```

## Usage

1. **Create Configuration File**:
   ```python
   import yaml
   
   config = {
       # Add configuration parameters as shown above
   }
   
   with open('config.yaml', 'w') as f:
       yaml.dump(config, f)
   ```

2. **Run Simulation**:
   ```python
   from dynamics_pipeline.simulation.simulation import MDSimulation
   
   # Load configuration
   with open('config.yaml', 'r') as f:
       config = yaml.safe_load(f)
   
   # Initialize simulation
   simulation = MDSimulation(config)
   
   # Set up system
   simulation.set_system()
   
   # Run simulation stages
   simulation.warmup()
   simulation.remove_backbone_constraints()
   simulation.nvt()
   simulation.npt()
   simulation.production()
   ```

## Simulation Stages

1. **Warmup**:
   - Gradually heats the system from initial to final temperature
   - Maintains backbone position restraints
   - Energy minimization at the start
   - Outputs: Trajectory, checkpoint, state data, topology

2. **Backbone Constraint Removal**:
   - Gradually removes position restraints from protein backbone
   - Maintains constant temperature
   - Outputs: Trajectory, checkpoint, state data, topology

3. **NVT Equilibration**:
   - Constant volume and temperature
   - No position restraints
   - Outputs: Trajectory, checkpoint, state data, topology

4. **NPT Equilibration**:
   - Constant pressure and temperature
   - System volume adjustment
   - Outputs: Trajectory, checkpoint, state data, topology

5. **Production**:
   - Final simulation phase
   - Constant pressure and temperature
   - Longest duration
   - Outputs: Trajectory, checkpoint, state data, topology

## Monitoring and Analysis

Each simulation stage produces:
- Trajectory files (.xtc format)
- Checkpoint files for simulation recovery
- State data reports (CSV) containing:
  - Step number
  - Potential energy
  - Kinetic energy
  - Temperature
  - Volume
  - Simulation speed
  - Remaining time
- System state files (XML)
- Topology files (CIF)

## Error Handling and Recovery

The pipeline supports automatic recovery from checkpoints if a simulation is interrupted. Each stage checks for existing checkpoints and resumes from the last saved state if available.

## Logging

The pipeline uses a comprehensive logging system that records:
- Simulation progress
- Parameter changes
- Error messages
- Warning messages
- Debug information

Logs are available at different verbosity levels (INFO, DEBUG, WARNING, ERROR).
