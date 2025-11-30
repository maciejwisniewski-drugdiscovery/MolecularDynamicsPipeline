#!/bin/bash

# Usage: ./run_single_tomek.sh <PROTEIN_FILE> <LIGAND_FILE> [CONFIG_TEMPLATE] [OUTPUT_DIR]

set -e

DEFAULT_CONFIG_TEMPLATE="config/tomek_parameters_bound.yaml"

PROTEIN_FILE="$1"
LIGAND_FILE="$2"
CONFIG_TEMPLATE="${3:-$DEFAULT_CONFIG_TEMPLATE}"
OUTPUT_DIR="${4:-$OUTPUT_DIR}"

if [ -z "$PROTEIN_FILE" ] || [ -z "$LIGAND_FILE" ]; then
    echo "Usage: $0 <PROTEIN_FILE> <LIGAND_FILE> [CONFIG_TEMPLATE] [OUTPUT_DIR]"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: OUTPUT_DIR is not set. Provide as 4th arg or set env OUTPUT_DIR"
    exit 1
fi

if [ ! -f "$PROTEIN_FILE" ]; then
    echo "Protein file not found: $PROTEIN_FILE"; exit 1; fi
if [ ! -f "$LIGAND_FILE" ]; then
    echo "Ligand file not found: $LIGAND_FILE"; exit 1; fi
if [ ! -f "$CONFIG_TEMPLATE" ]; then
    echo "Config template not found: $CONFIG_TEMPLATE"; exit 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$OUTPUT_DIR"

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PAIR_ID="$(basename "$PROTEIN_FILE")__$(basename "$LIGAND_FILE")"
LOG_FILE="$LOG_DIR/tomek_${PAIR_ID}_${TIMESTAMP}.log"

echo "==============================================================================" | tee -a "$LOG_FILE"
echo "TOMEK Simulation: $PROTEIN_FILE | $LIGAND_FILE" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Config template: $CONFIG_TEMPLATE" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "==============================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/.."

python scripts/tomek_scripts/run_single_tomek_simulation.py \
  --protein "$PROTEIN_FILE" \
  --ligand "$LIGAND_FILE" \
  --config_template "$CONFIG_TEMPLATE" \
  --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "==============================================================================" | tee -a "$LOG_FILE"
if [ $EXIT_CODE -eq 0 ]; then
  echo "TOMEK simulation completed successfully" | tee -a "$LOG_FILE"
else
  echo "TOMEK simulation failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
fi
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "==============================================================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE


