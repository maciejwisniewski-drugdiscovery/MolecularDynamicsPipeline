# Molecular Dynamics Simulation Pipeline

A comprehensive molecular dynamics simulation pipeline for protein-ligand systems, built with OpenMM and designed for high-throughput screening and detailed biomolecular analysis. This pipeline supports multi-stage MD simulations with advanced bond preservation, checkpoint recovery, and SLURM cluster integration.

## 🚀 Features

- **Multi-stage MD simulation pipeline**: Warmup → Backbone restraint removal → NVT → NPT → Production
- **Checkpoint-based recovery**: Resume interrupted simulations from any stage
- **SLURM cluster integration**: High-throughput batch processing capabilities
- **Multiple file format support**: PDB, CIF, SDF, MOL2 with proper bond handling
- **Comprehensive reporting**: Forces, trajectories, thermodynamic data, and Hessians
- **GPU acceleration**: CUDA and OpenCL platform support
- **Flexible force fields**: AMBER, GAFF, OpenFF with customizable parameters

## 📋 Prerequisites

- **Python**: 3.7-3.12 (recommended: 3.11)
- **CUDA**: For GPU acceleration (optional but recommended)
- **Git**: For installation from source
- **Conda/Mamba**: For environment management

## 🔧 Installation

### Method 1: Installation with Conda Environment (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/maciejwisniewski-drugdiscovery/MolecularDynamicsPipeline.git
   cd MolecularDynamicsPipeline
   ```

2. **Create conda environment from YAML**:
   ```bash
   conda env create -f environment.yml
   conda activate molecular_dynamics_pipeline
   ```

3. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

### Verification

Test your installation:
```bash
# Quick test
python -c "import molecular_dynamics_pipeline; print('Installation successful!')"
```

The validation script will check:
- Python version compatibility
- All required dependencies
- GPU/CUDA support
- Basic OpenMM functionality

## 📁 Project Structure

```
plinder_dynamics/
├── config/                           # Configuration templates
│   ├── plinder_parameters_bound.yaml      # Bound state simulations
│   ├── plinder_parameters_unbound.yaml    # Unbound state simulations  
│   ├── plinder_parameters_metadynamics.yaml # Enhanced sampling
│   ├── misato_parameters.yaml             # MISATO dataset configs
│   └── simulation_parameters.yaml         # Base parameters
├── scripts/                          # Execution scripts
│   ├── run_simulation.py                 # Main simulation runner
│   ├── plinder_scripts/                  # PLINDER-specific scripts
│   └── misato_scripts/                   # MISATO-specific scripts
├── src/dynamics_pipeline/            # Core pipeline modules
│   ├── simulation/                       # MD simulation engine
│   ├── data/                            # Data handling and processing
│   └── utils/                           # Utilities and helpers
├── environment.yml                   # Conda environment specification
└── setup.py                        # Package installation
```

## ⚙️ Configuration

### Configuration File Structure

The pipeline uses YAML configuration files with the following sections:

#### 1. System Information (`info`)
```yaml
info:
  system_id: "1abc_ligand_123"        # Unique system identifier
  simulation_id: "bound_state_md"      # Simulation identifier
  use_plinder_index: true             # Use PLINDER database integration
  bound_state: true                   # Bound vs unbound simulation
```

#### 2. File Paths (`paths`)
```yaml
paths:
  raw_protein_files: 
    - "path/to/protein.pdb"           # Protein structure files
  raw_ligand_files:
    - "path/to/ligand.sdf"            # Ligand structure files  
  output_dir: "path/to/output"        # Output directory
```

#### 3. Preprocessing Parameters (`preprocessing`)
```yaml
preprocessing:
  process_protein: true               # Clean protein with PDBFixer
  process_ligand: true                # Process ligand with OpenFF
  add_solvent: true                   # Add explicit solvent
  ionic_strength: 0.15                # Salt concentration (M)
  box_padding: 1.0                    # Solvent box padding (nm)
```

#### 4. Force Field Configuration (`forcefield`)
```yaml
forcefield:
  proteinFF: "amber14-all.xml"        # Protein force field
  nucleicFF: "amber14/DNA.OL15.xml"   # Nucleic acid force field
  ligandFF: "gaff-2.11"               # Ligand force field (gaff-2.11, openff-2.0.0)
  waterFF: "amber14/tip3pfb.xml"      # Water model
  water_model: "tip3p"                # Water model name
  forcefield_kwargs:                  # Additional FF parameters
    rigidWater: true
    removeCMMotion: false
    hydrogenMass: 1.5                 # Hydrogen mass repartitioning
```

**Force Field Options**:
- **Protein**: `amber14-all.xml`, `amber14/protein.ff14SB.xml`, `amber99sbildn.xml`
- **Ligand**: `gaff-2.11`, `openff-2.0.0`, `openff-2.1.0`
- **Water**: `tip3p`, `tip4pew`, `spce`

#### 5. Simulation Parameters (`simulation_params`)

**Platform Configuration**:
```yaml
simulation_params:
  platform:
    type: "CUDA"                      # Platform: CUDA, OpenCL, CPU
    devices: "0"                      # GPU device indices
  backbone_restraint_force: 100.0     # Backbone restraint (kcal/mol/Å²)
  save_forces: true                   # Save force data
  save_hessian: false                 # Save Hessian matrices
```

**Stage-Specific Parameters**:

Each simulation stage (warmup, backbone_removal, nvt, npt, production) supports:

```yaml
  warmup:
    init_temp: 50.0                   # Initial temperature (K)
    final_temp: 300.0                 # Final temperature (K)
    friction: 1.0                     # Langevin friction (ps⁻¹)
    time_step: 2.0                    # Integration timestep (fs)
    heating_step: 100                 # Steps per 1K temperature increase
    checkpoint_interval: 1000         # Checkpoint frequency
    trajectory_interval: 1000         # Trajectory save frequency
    state_data_reporter_interval: 1000 # State data frequency
```

### Creating Configuration Files

1. **Copy a template**:
   ```bash
   cp config/plinder_parameters_bound.yaml my_simulation.yaml
   ```

2. **Edit required fields**:
   - Set `system_id` and `simulation_id`
   - Update file paths in `paths` section
   - Adjust simulation parameters as needed

3. **Validate configuration**:
   ```bash
   python scripts/run_simulation.py --config my_simulation.yaml --validate-only
   ```

## 🏃 Usage

### Basic Simulation Execution

**Single simulation**:
```bash
python scripts/run_simulation.py --config config/my_simulation.yaml
```

**With custom output directory**:
```bash
python scripts/run_simulation.py \
  --config config/my_simulation.yaml \
  --output-dir /path/to/output
```

**Resume from checkpoint**:
```bash
python scripts/run_simulation.py \
  --config config/my_simulation.yaml \
  --resume
```

### Advanced Options

**Run specific stages only**:
```bash
python scripts/run_simulation.py \
  --config config/my_simulation.yaml \
  --stages warmup,nvt,production
```

**Validation mode** (check config without running):
```bash
python scripts/run_simulation.py \
  --config config/my_simulation.yaml \
  --validate-only
```

**Verbose logging**:
```bash
python scripts/run_simulation.py \
  --config config/my_simulation.yaml \
  --log-level DEBUG
```

### PLINDER Integration

For PLINDER database systems:
```bash
python scripts/plinder_scripts/run_single_plinder_simulation.py \
  --plinder_id "1abc__1.00__ligand_113" \
  --config config/plinder_parameters_bound.yaml \
  --output-dir /path/to/output
```

## 📊 Output Structure

```
output_directory/
├── forcefields/                      # Ligand topology with bonds
│   ├── {ligand_name}_topology.sdf         # SDF format with bonds
│   ├── {ligand_name}_topology.mol2        # MOL2 format with bonds
│   └── {ligand_name}_info.yaml            # Ligand metadata
├── trajectories/                     # Simulation trajectories
│   ├── {system_id}_warmup_trajectory.npz       # NPZ trajectory data
│   ├── {system_id}_nvt_trajectory.npz          # NPZ trajectory data
│   └── {system_id}_production_trajectory.npz   # NPZ trajectory data
├── checkpoints/                      # Checkpoint files for recovery
│   ├── {system_id}_warmup_checkpoint.dcd
│   └── {system_id}_production_checkpoint.dcd
├── state_data_reporters/             # Thermodynamic data
│   ├── {system_id}_warmup_state_data.csv
│   └── {system_id}_production_state_data.csv
├── states/                          # XML state files
├── topologies/                      # Structure files with bonds
│   ├── {system_id}_warmup_topology.cif
│   └── {system_id}_production_topology.cif
├── forces/                          # Force data (if enabled)
│   └── {system_id}_production_forces.npy
├── hessians/                        # Hessian matrices (if enabled)
│   └── {system_id}_production_hessian.npy
└── {system_id}_init_complex.cif     # Initial system structure
```

## 🧬 Bond Preservation

This pipeline preserves ligand bond connectivity throughout simulations:

- **Input processing**: Loads ligands with OpenFF/RDKit maintaining bond orders
- **Topology saving**: Exports CIF files with explicit bond information  
- **Multi-format output**: SDF and MOL2 files preserve bond connectivity
- **Metadata tracking**: YAML files contain bond and charge information


## 🖥️ SLURM Cluster Usage

For high-throughput simulations on SLURM clusters, see the provided SLURM integration scripts in the `scripts/` directory.

## 🐛 Troubleshooting

**Common Issues**:

1. **CUDA errors**: Ensure CUDA toolkit matches OpenMM version
2. **Memory issues**: Reduce trajectory saving frequency or use smaller systems
3. **Bond connectivity lost**: Verify input files have explicit bonds (SDF/MOL2)
4. **Checkpoint corruption**: Delete checkpoint files to restart from beginning

**Debugging**:
```bash
python scripts/run_simulation.py --config config.yaml --log-level DEBUG
```

## 📚 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{MolecularDynamicsPipeline,
  title={Molecular Dynamics Pipeline},
  author={Maciej Wisniewski},
  year={2024},
  url={https://github.com/maciejwisniewski-drugdiscovery/MolecularDynamicsPipeline}
}
```

## 🤝 Contributing

Contributions are welcome! Please see our contribution guidelines and submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions and support:
- **Issues**: GitHub Issues
- **Email**: m.wisniewski@datascience.edu.pl
- **Documentation**: See `docs/` directory for detailed API documentation
