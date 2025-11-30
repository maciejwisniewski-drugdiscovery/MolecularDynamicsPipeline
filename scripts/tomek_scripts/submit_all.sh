#!/bin/bash
#
# Usage:
#   ./submit_md_array.sh --csv /path/to/dynamics_manifest.csv \
#                        --config config/tomek_parameters_bound.yaml \
#                        --output /path/to/output
#
set -euo pipefail

DEFAULT_CONFIG="config/tomek_parameters_bound.yaml"
DEFAULT_PARTITION="short"
DEFAULT_NODES="1"
DEFAULT_ACCOUNT="sfglab"
DEFAULT_GPUS="1"
DEFAULT_CPUS_PER_TASK="10"
DEFAULT_MEM="64G"
DEFAULT_TIME="12:00:00"

CSV_FILE=""
CONFIG="$DEFAULT_CONFIG"
OUTPUT_DIR=""

# --------------------- PARSE ARGS ----------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv) CSV_FILE="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --output) OUTPUT_DIR="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$CSV_FILE" || ! -f "$CSV_FILE" ]]; then
  echo "Error: CSV file not found"; exit 1; fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Error: OUTPUT_DIR not provided"; exit 1; fi
mkdir -p "$OUTPUT_DIR"

# ------------------ COUNT LINES -------------------------
CSV_LINES=$(($(wc -l < "$CSV_FILE") - 1))
if (( CSV_LINES < 1 )); then
    echo "CSV appears empty"; exit 1
fi

echo "Submitting job array of size: $CSV_LINES"

SBATCH_SCRIPT="$OUTPUT_DIR/submit_array.sbatch"

# ------------------ BUILD ARRAY SCRIPT -------------------
cat > "$SBATCH_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=md_array
#SBATCH --account=${DEFAULT_ACCOUNT}
#SBATCH --partition=${DEFAULT_PARTITION}
#SBATCH --nodes=${DEFAULT_NODES}
#SBATCH --gpus=${DEFAULT_GPUS}
#SBATCH --cpus-per-task=${DEFAULT_CPUS_PER_TASK}
#SBATCH --mem=${DEFAULT_MEM}
#SBATCH --time=${DEFAULT_TIME}
#SBATCH --array=1-${CSV_LINES}
#SBATCH --output=${OUTPUT_DIR}/slurm_%A_%a.out
#SBATCH --error=${OUTPUT_DIR}/slurm_%A_%a.err

module load anaconda
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate molecular_dynamics_pipeline

CSV_FILE="${CSV_FILE}"
CONFIG="${CONFIG}"
OUTPUT_DIR="${OUTPUT_DIR}"

# Fetch nth line (skip header)
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "\$CSV_FILE")

IFS=',' read -r PROTEIN_REL LIGAND_ID LIGAND_REL SMILES <<< "\$LINE"

# Resolve absolute paths
if [[ "\$PROTEIN_REL" = /* ]]; then
    PROTEIN_FILE="\$PROTEIN_REL"
else
    PROTEIN_FILE="\$(dirname "\$CSV_FILE")/\$PROTEIN_REL"
fi

if [[ "\$LIGAND_REL" = /* ]]; then
    LIGAND_FILE="\$LIGAND_REL"
else
    LIGAND_FILE="\$(dirname "\$CSV_FILE")/\$LIGAND_REL"
fi

echo "Running MD for:"
echo "  Protein = \$PROTEIN_FILE"
echo "  Ligand  = \$LIGAND_FILE"
echo "  Config  = \$CONFIG"

bash scripts/tomek_scripts/run_single_tomek.sh "\$PROTEIN_FILE" "\$LIGAND_FILE" "\$CONFIG" "\$OUTPUT_DIR"
EOF

chmod +x "$SBATCH_SCRIPT"

sbatch "$SBATCH_SCRIPT"
