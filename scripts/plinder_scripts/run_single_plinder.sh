#!/bin/bash

# Script to run a single PLINDER simulation
# Usage: ./run_single_plinder.sh <PLINDER_ID> [CONFIG_TEMPLATE] [OUTPUT_DIR]

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default values
DEFAULT_CONFIG_TEMPLATE="config/plinder_parameters_bound.yaml"
DEFAULT_OUTPUT_DIR="${OUTPUT_DIR}"

# Parse command line arguments
PLINDER_ID="$1"
CONFIG_TEMPLATE="${2:-$DEFAULT_CONFIG_TEMPLATE}"
OUTPUT_DIR="${3:-$DEFAULT_OUTPUT_DIR}"

# =============================================================================
# VALIDATION
# =============================================================================

if [ -z "$PLINDER_ID" ]; then
    echo "Error: PLINDER_ID is required"
    echo "Usage: $0 <PLINDER_ID> [CONFIG_TEMPLATE] [OUTPUT_DIR]"
    echo "Example: $0 1a0o__A_1_A.bfactor-90_01 config/plinder_parameters_bound.yaml /path/to/output"
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

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# LOGGING
# =============================================================================

# Create logs directory
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Set up logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/plinder_${PLINDER_ID}_${TIMESTAMP}.log"

echo "==============================================================================" | tee -a "$LOG_FILE"
echo "PLINDER Simulation: $PLINDER_ID" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Config template: $CONFIG_TEMPLATE" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==============================================================================" | tee -a "$LOG_FILE"

# =============================================================================
# RESOURCE MONITORING
# =============================================================================

# Function to log resource usage
log_resources() {
    echo "--- Resource Usage at $(date) ---" >> "$LOG_FILE"
    echo "GPU Status:" >> "$LOG_FILE"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits >> "$LOG_FILE" 2>/dev/null || echo "No GPU found" >> "$LOG_FILE"
    echo "Memory Usage:" >> "$LOG_FILE"
    free -h >> "$LOG_FILE"
    echo "CPU Usage:" >> "$LOG_FILE"
    top -bn1 | grep "Cpu(s)" >> "$LOG_FILE"
    echo "---" >> "$LOG_FILE"
}

# Log initial resources
log_resources

# =============================================================================
# RUN SIMULATION
# =============================================================================

echo "Starting PLINDER simulation for ID: $PLINDER_ID" | tee -a "$LOG_FILE"

# Change to project root directory
cd "$PROJECT_ROOT"

# Run the simulation
source "/mnt/evafs/groups/sfglab/mwisniewski/Projects/MolecularDynamicsPipeline/plinder.env"
python plinder_scripts/run_single_plinder_simulation.py \
    --plinder_id "$PLINDER_ID" \
    --config_template "$CONFIG_TEMPLATE" \
    --output_dir "$OUTPUT_DIR" \
    --parallel 1 2>&1 | tee -a "$LOG_FILE"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

# =============================================================================
# FINALIZATION
# =============================================================================

# Log final resources
log_resources

# Log completion
echo "==============================================================================" | tee -a "$LOG_FILE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "PLINDER simulation completed successfully for ID: $PLINDER_ID" | tee -a "$LOG_FILE"
    echo "Finished at: $(date)" | tee -a "$LOG_FILE"
else
    echo "PLINDER simulation failed for ID: $PLINDER_ID with exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
    echo "Failed at: $(date)" | tee -a "$LOG_FILE"
fi
echo "==============================================================================" | tee -a "$LOG_FILE"

# Exit with the same code as the Python script
exit $EXIT_CODE 
