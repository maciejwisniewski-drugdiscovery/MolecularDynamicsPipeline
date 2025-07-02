# SLURM Scripts for PLINDER Simulations

This directory contains scripts for running PLINDER molecular dynamics simulations on SLURM clusters.

## Scripts

### 1. `run_single_plinder.sh`
Runs a single PLINDER simulation with comprehensive logging and resource monitoring.

**Usage:**
```bash
./scripts/run_single_plinder.sh <PLINDER_ID> [CONFIG_TEMPLATE] [OUTPUT_DIR]
```

**Examples:**
```bash
# Basic usage (uses environment variables for output dir)
./scripts/run_single_plinder.sh 1a0o__A_1_A.bfactor-90_01

# With custom config and output directory
./scripts/run_single_plinder.sh 1a0o__A_1_A.bfactor-90_01 config/plinder_parameters_bound.yaml /path/to/output

# Run directly (requires OUTPUT_DIR environment variable)
export OUTPUT_DIR=/mnt/raid/output
./scripts/run_single_plinder.sh 1a0o__A_1_A.bfactor-90_01
```

### 2. `submit_plinder_jobs.sh`
Submits multiple PLINDER simulations to SLURM queue based on a list of IDs.

**Usage:**
```bash
./scripts/submit_plinder_jobs.sh <plinder_id_list.txt> [CONFIG_TEMPLATE] [OUTPUT_DIR]
```

**Examples:**
```bash
# Submit jobs from ID list
./scripts/submit_plinder_jobs.sh scripts/example_plinder_ids.txt

# With custom parameters
./scripts/submit_plinder_jobs.sh my_plinder_ids.txt config/plinder_parameters_bound.yaml /path/to/output
```

## Setup

### 1. Environment Variables
Set these in your `plinder.env` file or shell:
```bash
export OUTPUT_DIR="/path/to/your/output/directory"
```

### 2. SLURM Configuration
Edit the default SLURM parameters in `submit_plinder_jobs.sh`:
```bash
DEFAULT_PARTITION="short"         # Partition name
DEFAULT_NODES="1"                 # Number of nodes
DEFAULT_ACCOUNT="sfglab"          # Account name
DEFAULT_GPUS="1"                  # GPU resource request
DEFAULT_CPUS_PER_TASK="10"        # Number of CPU cores
DEFAULT_MEM="100G"                # Memory request
DEFAULT_TIME="24:00:00"           # Wall time limit
```

### 3. Environment Setup
Edit the environment setup section in the generated sbatch scripts:
```bash
# Activate virtual environment (EDIT FOR YOUR SETUP)
# Replace with your actual virtual environment path
source /path/to/your/venv/bin/activate
```

### 4. PLINDER ID List Format
Create a text file with one PLINDER ID per line:
```
# Comments start with #
1a0o__A_1_A.bfactor-90_01
1a1e__A_1_A.bfactor-90_01
1a28__A_1_A.bfactor-90_01
```

## SLURM Environment Variables

Override default SLURM parameters using environment variables:
```bash
export SLURM_PARTITION="long"
export SLURM_NODES="1"
export SLURM_ACCOUNT="your_account"
export SLURM_GPUS="2"
export SLURM_CPUS_PER_TASK="20"
export SLURM_MEM="200G"
export SLURM_TIME="48:00:00"

# Then submit jobs
./scripts/submit_plinder_jobs.sh my_plinder_ids.txt
```

## File Organization

When you run the scripts, the following directory structure is created:
```
$OUTPUT_DIR/
├── logs/                                    # Global logs from submit script
├── sbatch_scripts/                         # Generated sbatch files
│   ├── plinder_1a0o__A_1_A.bfactor-90_01.sbatch
│   └── ...
├── slurm_logs/                             # SLURM stdout/stderr logs
│   ├── plinder_1a0o__A_1_A.bfactor-90_01_12345.out
│   ├── plinder_1a0o__A_1_A.bfactor-90_01_12345.err
│   └── ...
└── 1a0o__A_1_A.bfactor-90_01_simulation_bound_state/  # Simulation output
    ├── config.yaml
    ├── logs/
    ├── checkpoints/
    ├── trajectories/
    └── ...
```

## Monitoring Jobs

```bash
# Monitor your jobs
squeue -u $USER

# Check specific job details
scontrol show job <job_id>

# Cancel all your jobs
scancel -u $USER

# Cancel specific job
scancel <job_id>

# Check job efficiency after completion
seff <job_id>
```

## Troubleshooting

### Common Issues

1. **Module not found**: Edit the module loading section in `submit_plinder_jobs.sh`
2. **Virtual environment not found**: Edit the virtual environment path in the sbatch script
3. **Permission denied**: Run `chmod +x scripts/*.sh`
4. **CUDA out of memory**: Reduce system size or increase memory request
5. **Job time limit**: Increase `DEFAULT_TIME` or set `SLURM_TIME`

### Log Files

- **Simulation logs**: `$OUTPUT_DIR/<simulation_id>/logs/`
- **SLURM logs**: `$OUTPUT_DIR/slurm_logs/`
- **Bash script logs**: `$OUTPUT_DIR/logs/`

### Resume Failed Jobs

The simulation pipeline supports checkpoint resuming. If a job fails partway through, simply resubmit it - it will resume from the last checkpoint.

## Performance Tips

1. **GPU utilization**: Monitor with `nvidia-smi` in your logs
2. **Memory usage**: Check logs for memory consumption patterns
3. **I/O performance**: Use fast storage for trajectory files
4. **Batch size**: Submit jobs in batches to avoid overwhelming the scheduler 