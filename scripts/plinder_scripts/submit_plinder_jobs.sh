#!/bin/bash

# Script to submit multiple PLINDER simulations to SLURM queue
# Usage: ./submit_plinder_jobs.sh <plinder_id_list.txt> [CONFIG_TEMPLATE] [OUTPUT_DIR]

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default SLURM parameters - EDIT THESE FOR YOUR CLUSTER
DEFAULT_PARTITION="short"         # Partition name
DEFAULT_NODES="1"                 # Number of nodes
DEFAULT_ACCOUNT="sfglab"          # Account name
DEFAULT_GPUS="1"                  # GPU resource request
DEFAULT_CPUS_PER_TASK="10"        # Number of CPU cores
DEFAULT_MEM="100G"                # Memory request
DEFAULT_TIME="24:00:00"           # Wall time limit (24 hours)

# Default simulation parameters
DEFAULT_CONFIG_TEMPLATE="config/plinder_parameters_bound.yaml"
DEFAULT_OUTPUT_DIR="${OUTPUT_DIR}"

# Parse command line arguments
PLINDER_ID_LIST="$1"
CONFIG_TEMPLATE="${2:-$DEFAULT_CONFIG_TEMPLATE}"
OUTPUT_DIR="${3:-$DEFAULT_OUTPUT_DIR}"

# =============================================================================
# VALIDATION
# =============================================================================

if [ -z "$PLINDER_ID_LIST" ]; then
    echo "Error: PLINDER_ID_LIST file is required"
    echo "Usage: $0 <plinder_id_list.txt> [CONFIG_TEMPLATE] [OUTPUT_DIR]"
    echo "Example: $0 plinder_ids.txt config/plinder_parameters_bound.yaml /path/to/output"
    echo ""
    echo "The plinder_id_list.txt should contain one PLINDER ID per line:"
    echo "1a0o__A_1_A.bfactor-90_01"
    echo "1a1e__A_1_A.bfactor-90_01"
    echo "..."
    exit 1
fi

if [ ! -f "$PLINDER_ID_LIST" ]; then
    echo "Error: PLINDER ID list file not found: $PLINDER_ID_LIST"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: OUTPUT_DIR is not set. Please set the OUTPUT_DIR environment variable or provide it as the third argument."
    exit 1
fi

if [ ! -f "$CONFIG_TEMPLATE" ]; then
    echo "Error: Config template file not found: $CONFIG_TEMPLATE"
    exit 1
fi

# =============================================================================
# SETUP
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment if .env file exists
if [ -f "$PROJECT_ROOT/plinder.env" ]; then
    echo "Loading environment from $PROJECT_ROOT/plinder.env"
    set -a  # automatically export all variables
    source "$PROJECT_ROOT/plinder.env"
    set +a
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
SBATCH_DIR="$OUTPUT_DIR/sbatch_scripts"
SLURM_LOGS_DIR="$OUTPUT_DIR/slurm_logs"
mkdir -p "$SBATCH_DIR" "$SLURM_LOGS_DIR"

# =============================================================================
# SLURM CONFIGURATION
# =============================================================================

# Allow override of SLURM parameters via environment variables
PARTITION="${SLURM_PARTITION:-$DEFAULT_PARTITION}"
NODES="${SLURM_NODES:-$DEFAULT_NODES}"
ACCOUNT="${SLURM_ACCOUNT:-$DEFAULT_ACCOUNT}"
GPUS="${SLURM_GPUS:-$DEFAULT_GPUS}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-$DEFAULT_CPUS_PER_TASK}"
MEM="${SLURM_MEM:-$DEFAULT_MEM}"
TIME="${SLURM_TIME:-$DEFAULT_TIME}"

# Optional additional SLURM parameters
RESERVATION="${SLURM_RESERVATION:-}"
EXCLUDE="${SLURM_EXCLUDE:-}"

echo "==============================================================================" 
echo "PLINDER SLURM Job Submission"
echo "==============================================================================" 
echo "PLINDER ID list: $PLINDER_ID_LIST"
echo "Config template: $CONFIG_TEMPLATE"
echo "Output directory: $OUTPUT_DIR"
echo "SLURM partition: $PARTITION"
echo "Nodes: $NODES"
echo "Account: $ACCOUNT"
echo "GPUs: $GPUS"
echo "CPUs per task: $CPUS_PER_TASK"
echo "Memory: $MEM"
echo "Wall time: $TIME"
if [ -n "$RESERVATION" ]; then echo "Reservation: $RESERVATION"; fi
if [ -n "$EXCLUDE" ]; then echo "Exclude nodes: $EXCLUDE"; fi
echo "==============================================================================" 

# =============================================================================
# READ PLINDER IDs
# =============================================================================

# Read PLINDER IDs from file, removing empty lines and comments
mapfile -t PLINDER_IDS < <(grep -v '^#' "$PLINDER_ID_LIST" | grep -v '^$' | tr -d '\r')

echo "Found ${#PLINDER_IDS[@]} PLINDER IDs to process"

if [ ${#PLINDER_IDS[@]} -eq 0 ]; then
    echo "Error: No valid PLINDER IDs found in $PLINDER_ID_LIST"
    exit 1
fi

# =============================================================================
# JOB SUBMISSION
# =============================================================================

SUBMITTED_JOBS=0
SKIPPED_JOBS=0
FAILED_SUBMISSIONS=0

for PLINDER_ID in "${PLINDER_IDS[@]}"; do
    # Clean the PLINDER ID (remove any extra whitespace)
    PLINDER_ID=$(echo "$PLINDER_ID" | xargs)
    
    if [ -z "$PLINDER_ID" ]; then
        continue
    fi
    
    echo "Processing PLINDER ID: $PLINDER_ID"
    
    # Create job-specific sbatch script
    SBATCH_SCRIPT="$SBATCH_DIR/plinder_${PLINDER_ID}.sbatch"
    SLURM_OUT="$SLURM_LOGS_DIR/plinder_${PLINDER_ID}_%j.out"
    SLURM_ERR="$SLURM_LOGS_DIR/plinder_${PLINDER_ID}_%j.err"
    
    # Generate sbatch script
    cat > "$SBATCH_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=plinder_${PLINDER_ID}
#SBATCH --nodes=${NODES}
#SBATCH --account=${ACCOUNT}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus=${GPUS}
#SBATCH --mem=${MEM}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --output=${SLURM_OUT}
#SBATCH --error=${SLURM_ERR}
EOF

    # Add optional SLURM parameters if set
    if [ -n "$RESERVATION" ]; then
        echo "#SBATCH --reservation=${RESERVATION}" >> "$SBATCH_SCRIPT"
    fi
    if [ -n "$EXCLUDE" ]; then
        echo "#SBATCH --exclude=${EXCLUDE}" >> "$SBATCH_SCRIPT"
    fi

    # Add job content
    cat >> "$SBATCH_SCRIPT" << EOF

# Job information
echo "==============================================================================" 
echo "SLURM Job Information"
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "Node: \$SLURM_JOB_NODELIST"
echo "Started at: \$(date)"
echo "PLINDER ID: ${PLINDER_ID}"
echo "==============================================================================" 

# Load modules and activate virtual environment (EDIT FOR YOUR SETUP)
# module load cuda/11.8
# module load python/3.11

# Activate virtual environment (EDIT FOR YOUR SETUP)
# Replace with your actual virtual environment path
# source /path/to/your/venv/bin/activate

# Change to project directory
cd ${PROJECT_ROOT}

# Set CUDA device (in case multiple GPUs are allocated)
export CUDA_VISIBLE_DEVICES=0

# Run the simulation
bash scripts/plinder_scripts/run_single_plinder.sh "${PLINDER_ID}" "${CONFIG_TEMPLATE}" "${OUTPUT_DIR}"

# Job completion
echo "==============================================================================" 
echo "Job completed at: \$(date)"
echo "Exit code: \$?"
echo "==============================================================================" 
EOF

    # Submit the job
    if sbatch "$SBATCH_SCRIPT"; then
        echo "✓ Submitted job for PLINDER ID: $PLINDER_ID"
        SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
    else
        echo "✗ Failed to submit job for PLINDER ID: $PLINDER_ID"
        FAILED_SUBMISSIONS=$((FAILED_SUBMISSIONS + 1))
    fi
    
    # Small delay to avoid overwhelming the scheduler
    sleep 0.1
done

# =============================================================================
# SUMMARY
# =============================================================================

echo "==============================================================================" 
echo "Job Submission Summary"
echo "==============================================================================" 
echo "Total PLINDER IDs processed: ${#PLINDER_IDS[@]}"
echo "Jobs submitted successfully: $SUBMITTED_JOBS"
echo "Failed submissions: $FAILED_SUBMISSIONS"
echo "Skipped jobs: $SKIPPED_JOBS"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Cancel all jobs: scancel -u \$USER"
echo "SLURM logs directory: $SLURM_LOGS_DIR"
echo "Sbatch scripts directory: $SBATCH_DIR"
echo "==============================================================================" 

# Exit with error if any submissions failed
if [ $FAILED_SUBMISSIONS -gt 0 ]; then
    exit 1
fi 