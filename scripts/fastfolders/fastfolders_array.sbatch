#!/usr/bin/env bash
#SBATCH --mail-type=FAIL,TIME_LIMIT
# (Resource allocation is provided via 'sbatch' command line in the launcher)

set -euo pipefail

echo "=============================================================================="
echo "SLURM Job Info"
echo "Array JobID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Task ID:     ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Hostname:    $(hostname)"
echo "Started at:  $(date)"
echo "=============================================================================="

# ====== Validate env from launcher ======
: "${PROTEIN_CLEAN_LIST:?PROTEIN_CLEAN_LIST not exported}"
: "${INPUT_DIR:?INPUT_DIR not exported}"
: "${CONFIG_TEMPLATE:?CONFIG_TEMPLATE not exported}"
: "${OUTPUT_DIR:?OUTPUT_DIR not exported}"
: "${PROJECT_ROOT:?PROJECT_ROOT not exported}"

# ====== Get PROTEIN_NUMBER for this array index ======
INDEX=$((SLURM_ARRAY_TASK_ID + 1))  # sed is 1-indexed
PROTEIN_NUMBER="$(sed -n "${INDEX}p" "${PROTEIN_CLEAN_LIST}")"

if [[ -z "${PROTEIN_NUMBER}" ]]; then
  echo "Error: empty PROTEIN_NUMBER for index ${SLURM_ARRAY_TASK_ID}"
  exit 2
fi

echo "Processing PROTEIN_NUMBER='${PROTEIN_NUMBER}' (index ${SLURM_ARRAY_TASK_ID})"

# ====== Modules / conda (adjust to your cluster) ======
# module load cuda/11.8
module load anaconda
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate molecular_dynamics_pipeline
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || (echo "NO GPU AVAILABLE — FAIL" && exit 99)

# ====== Work in repo ======
cd "${PROJECT_ROOT}"

# Note: DO NOT override CUDA_VISIBLE_DEVICES — SLURM sets it correctly for --gpus

# ====== Idempotency (optional) – skip if result already exists ======
# Done flag based on simulation output structure
DONE_FLAG="${OUTPUT_DIR}/representative_frame_${PROTEIN_NUMBER}_simulation_production/_JOB_DONE"
if [[ -f "${DONE_FLAG}" ]]; then
  echo "Skip: found ${DONE_FLAG}"
  exit 0
fi

# ====== Run single task ======
echo ">>> Running run_production_simulation.py for protein_number=${PROTEIN_NUMBER}"
python scripts/fastfolders/run_production_simulation.py \
  --config_template "${CONFIG_TEMPLATE}" \
  --input_dir "${INPUT_DIR}" \
  --protein_number "${PROTEIN_NUMBER}" \
  --output_dir "${OUTPUT_DIR}"
EC=$?

# ====== Mark as done if successful ======
if [[ "${EC}" -eq 0 ]]; then
  DONE_DIR="${OUTPUT_DIR}/representative_frame_${PROTEIN_NUMBER}_simulation_production"
  mkdir -p "${DONE_DIR}"
  touch "${DONE_DIR}/_JOB_DONE"
  echo "Created done flag: ${DONE_DIR}/_JOB_DONE"
fi

echo "=============================================================================="
echo "Finished at: $(date)"
echo "=============================================================================="
exit "${EC}"
