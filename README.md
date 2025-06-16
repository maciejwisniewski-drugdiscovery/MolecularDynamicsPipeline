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

- Python 3.11
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

## Directory Structure

```
plinder_dynamics/
├── config/
│   ├── plinder_parameters_bound.yaml     # Configuration for bound state simulations
│   ├── plinder_parameters_unbound.yaml   # Configuration for unbound state simulations
│   ├── misato_parameters.yaml            # Configuration for MISATO simulations
│   └── simulation_parameters.yaml        # Base simulation parameters
├── scripts/
│   ├── filters/
│   │   ├── train_plinder.yaml           # Filter for training set
│   │   └── test_plinder.yaml            # Filter for test set
│   ├── run_single_plinder_simulation.py  # Run single system simulation
│   ├── run_single_misato_simulation.py   # Run single MISATO simulation
│   ├── run_simulation.py                 # Generic simulation runner
│   └── fix_autodock_molecules.py         # Utility for fixing AutoDock output
└── src/
    └── dynamics_pipeline/
        ├── simulation/
        ├── data/
        └── utils/
```

## Scripts Usage

### Running Simulations

1. **Single System Simulation**:
   ```bash
   python scripts/run_simulation.py \
     --config path/to/config.yaml \
   ```

   Arguments:
   - `--config_template`: Path to configuration file

2. **Plinder System Simulation**:
   ```bash
   python scripts/run_single_plinder_simulation.py \
     --config_template config/plinder_parameters_bound.yaml \
     --filters scripts/filters/train_plinder.yaml \
     --output_dir /path/to/output \
     --parallel 4
   ```

   Arguments:
   - `--config_template`: Path to configuration template
   - `--filters`: Path to system filters file
   - `--output_dir`: Output directory for simulations
   - `--parallel`: Number of parallel processes (default: 1)
   - `--overwrite`: Overwrite existing simulations (default: False)

3. **MISATO Simulation**:
   ```bash
   python scripts/run_single_misato_simulation.py \
     --config config/misato_parameters.yaml \
     --output_dir /path/to/output
   ```

### Environment Setup

Required environment variables:
```bash
export PLINDER_MOUNT='/path/to/data'
export PLINDER_RELEASE='2024-06'
export PLINDER_ITERATION='v2'
export PLINDER_OFFLINE='true'
```

## Configuration System

### Bound vs Unbound States

The pipeline supports both bound and unbound state simulations:

1. **Bound State**:
   - The name can be misleading but by dyfeault `bound_state:` in config file is set `True`
   - It's classical MD Simualtion

2. **Unbound State**:
   - Simulates protein and ligand separately
   - When `bound_states` is set `False`, ligands are moved by random vector on sphere to simulate unbound state of protein - ligand complex 

## Preprocessing Mechanisms

### Protein Preprocessing

1. **PDBFixer Processing**:
   - Finds and adds missing residues
   - Replaces non-standard residues
   - Adds missing heavy atoms
   - Adds missing hydrogens at specified pH
   - Outputs standardized PDB file

2. **PDB2PQR Processing**:
   - Assigns protonation states
   - Optimizes hydrogen bonding network
   - Assigns atomic charges and radii
   - Repairs broken side chains

### Ligand Preprocessing

1. **Structure Preparation**:
   - Converts input format to OpenFF molecule
   - Assigns atom types and parameters
   - Generates 3D conformers if needed
   - Validates molecular structure

2. **Charge Assignment**:
   - Uses Gasteiger charge method by default
   - Stores charges in configuration
   - Supports custom charge assignments
   - Validates total molecular charge

3. **Force Field Assignment**:
   - Uses OpenFF force field
   - Generates parameters for all atom types
   - Validates parameter coverage
   - Handles special atom types

### System Setup

1. **Complex Building**:
   - Combines processed protein and ligand
   - Assigns chain IDs and residue names
   - Validates structure integrity
   - Creates initial topology

2. **Solvation**:
   - Adds water box with specified padding
   - Adds ions to neutralize system
   - Sets ionic strength
   - Validates system composition

3. **Force Field Setup**:
   - Combines protein and ligand parameters
   - Adds solvent parameters
   - Sets up periodic boundary conditions
   - Validates parameter completeness

## Troubleshooting

Common issues and solutions:

1. **CUDA Errors**:
   - Ensure CUDA drivers are installed
   - Check GPU visibility with `nvidia-smi`
   - Verify OpenMM CUDA support
   - Try running with `--force_cpu`

2. **Memory Issues**:
   - Reduce system size or padding
   - Decrease parallel processes
   - Monitor GPU memory usage
   - Use smaller trajectory save intervals

3. **Simulation Instability**:
   - Check input structure quality
   - Validate force field parameters
   - Adjust time step and constraints
   - Monitor energy conservation

4. **Checkpoint Recovery**:
   - Verify checkpoint file integrity
   - Check file permissions
   - Ensure consistent configuration
   - Monitor disk space

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

## Logging System

The pipeline implements a comprehensive logging system with the following features:

1. **Global Logger**:
   - Tracks overall simulation progress
   - Located in the main output directory under `logs/`
   - Named `plinder_dynamics_TIMESTAMP.log`

2. **System-Specific Loggers**:
   - Each simulation gets its own logger
   - Located in `output_dir/SYSTEM_ID/logs/`
   - Named `plinder_dynamics_SYSTEM_ID_TIMESTAMP.log`
   - Tracks detailed progress of individual simulation stages

3. **Log Structure**:
   ```
   output_dir/
   ├── logs/
   │   └── plinder_dynamics_20240315_123456.log  # Global logger
   ├── system1_simulation/
   │   ├── logs/
   │   │   └── plinder_dynamics_system1_20240315_123456.log
   │   └── ...
   ├── system2_simulation/
   │   ├── logs/
   │   │   └── plinder_dynamics_system2_20240315_123457.log
   │   └── ...
   └── ...
   ```

4. **Log Levels**:
   - ERROR: Critical issues that prevent simulation completion
   - WARNING: Non-critical issues that might affect results
   - INFO: Progress updates and stage completion
   - DEBUG: Detailed technical information

5. **Log Format**:
   ```
   2024-03-15 12:34:56 - plinder_dynamics_system1 - INFO - Starting warmup phase
   ```
   Each log entry includes:
   - Timestamp
   - Logger name
   - Log level
   - Message

6. **Parallel Processing**:
   - In parallel mode, each process maintains its own system-specific logger
   - Prevents log file conflicts between parallel simulations
   - Global logger tracks overall progress

7. **Error Handling**:
   - Failed simulations are logged with full stack traces
   - Errors are captured in both global and system-specific logs
   - System-specific logs provide detailed context for debugging
