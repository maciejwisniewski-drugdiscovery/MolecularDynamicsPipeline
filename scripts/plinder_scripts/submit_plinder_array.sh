#!/usr/bin/env bash
# Usage: ./submit_plinder_array.sh <plinder_id_list.txt> [CONFIG_TEMPLATE] [OUTPUT_DIR]
# Env overrides (opcjonalne):
#   SLURM_PARTITION, SLURM_NODES, SLURM_ACCOUNT, SLURM_GPUS, SLURM_CPUS_PER_TASK, SLURM_MEM, SLURM_TIME
#   SLURM_RESERVATION, SLURM_EXCLUDE, SLURM_ARRAY_THROTTLE (np. 8)

set -euo pipefail

# ============================ DEFAULTS (dopasowane do Twoich) ============================
DEFAULT_PARTITION="short"
DEFAULT_NODES="1"
DEFAULT_ACCOUNT="sfglab"
DEFAULT_GPUS="1"
DEFAULT_CPUS_PER_TASK="10"
DEFAULT_MEM="100G"
DEFAULT_TIME="24:00:00"
DEFAULT_CONFIG_TEMPLATE="/mnt/evafs/groups/sfglab/mwisniewski/Projects/MolecularDynamicsPipeline/config/plinder_parameters_bound.yaml"
DEFAULT_OUTPUT_DIR="/mnt/evafs/groups/sfglab/mwisniewski/data/plinder_md"
DEFAULT_ARRAY_THROTTLE="8"  # maks. ile zadań równolegle

# ============================ ARGS ============================
PLINDER_ID_LIST="${1:-}"
CONFIG_TEMPLATE="${2:-$DEFAULT_CONFIG_TEMPLATE}"
OUTPUT_DIR="${3:-$DEFAULT_OUTPUT_DIR}"

if [[ -z "${PLINDER_ID_LIST}" ]]; then
  echo "Error: PLINDER_ID_LIST file is required"
  echo "Usage: $0 <plinder_id_list.txt> [CONFIG_TEMPLATE] [OUTPUT_DIR]"
  exit 1
fi
if [[ ! -f "${PLINDER_ID_LIST}" ]]; then
  echo "Error: file not found: ${PLINDER_ID_LIST}"
  exit 1
fi
if [[ ! -f "${CONFIG_TEMPLATE}" ]]; then
  echo "Error: config template not found: ${CONFIG_TEMPLATE}"
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

# ============================ PROJECT ROOT & ENV ============================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

if [[ -f "${PROJECT_ROOT}/plinder.env" ]]; then
  echo "Loading env from ${PROJECT_ROOT}/plinder.env"
  set -a; source "${PROJECT_ROOT}/plinder.env"; set +a
fi

# ============================ CLEAN LIST & DIRS ============================
SBATCH_DIR="${OUTPUT_DIR}/sbatch_scripts"
SLURM_LOGS_DIR="${OUTPUT_DIR}/slurm_logs"
mkdir -p "${SBATCH_DIR}" "${SLURM_LOGS_DIR}"

CLEAN_LIST="${OUTPUT_DIR}/plinder_ids.clean.txt"
grep -v '^\s*#' "${PLINDER_ID_LIST}" | grep -v '^\s*$' | tr -d '\r' | xargs -I{} echo {} > "${CLEAN_LIST}"

mapfile -t PLINDER_IDS < "${CLEAN_LIST}"
NUM_IDS="${#PLINDER_IDS[@]}"
if [[ "${NUM_IDS}" -eq 0 ]]; then
  echo "Error: no valid PLINDER IDs after cleaning ${PLINDER_ID_LIST}"
  exit 1
fi

echo "Found ${NUM_IDS} PLINDER IDs"

# ============================ SLURM PARAMS (z ENV override) ============================
PARTITION="${SLURM_PARTITION:-$DEFAULT_PARTITION}"
NODES="${SLURM_NODES:-$DEFAULT_NODES}"
ACCOUNT="${SLURM_ACCOUNT:-$DEFAULT_ACCOUNT}"
GPUS="${SLURM_GPUS:-$DEFAULT_GPUS}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-$DEFAULT_CPUS_PER_TASK}"
MEM="${SLURM_MEM:-$DEFAULT_MEM}"
TIME="${SLURM_TIME:-$DEFAULT_TIME}"
RESERVATION="${SLURM_RESERVATION:-}"
EXCLUDE="${SLURM_EXCLUDE:-}"
ARRAY_THROTTLE="${SLURM_ARRAY_THROTTLE:-$DEFAULT_ARRAY_THROTTLE}"

# ============================ SBATCH FLAGS ============================
ARRAY_RANGE="0-$((NUM_IDS-1))%${ARRAY_THROTTLE}"
JOB_NAME="plinder"
OUT_PATTERN="${SLURM_LOGS_DIR}/plinder_%A_%a.out"
ERR_PATTERN="${SLURM_LOGS_DIR}/plinder_%A_%a.err"

echo "Submitting array: ${ARRAY_RANGE} (THROTTLE=${ARRAY_THROTTLE})"
echo "Partition=${PARTITION} Nodes=${NODES} Account=${ACCOUNT} GPUs=${GPUS} CPUs=${CPUS_PER_TASK} Mem=${MEM} Time=${TIME}"
[[ -n "${RESERVATION}" ]] && echo "Reservation=${RESERVATION}"
[[ -n "${EXCLUDE}" ]] && echo "Exclude=${EXCLUDE}"

# ============================ SUBMIT ============================
SBATCH_FLAGS=(
  --job-name="${JOB_NAME}"
  --partition="${PARTITION}"
  --nodes="${NODES}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --gpus="${GPUS}"
  --gres="gpu:${GPUS}"
  --mem="${MEM}"
  --time="${TIME}"
  --array="${ARRAY_RANGE}"
  --output="${OUT_PATTERN}"
  --error="${ERR_PATTERN}"
)

[[ -n "${ACCOUNT}" ]] && SBATCH_FLAGS+=( --account="${ACCOUNT}" )
[[ -n "${RESERVATION}" ]] && SBATCH_FLAGS+=( --reservation="${RESERVATION}" )
[[ -n "${EXCLUDE}" ]] && SBATCH_FLAGS+=( --exclude="${EXCLUDE}" )

# Exportujemy ścieżki/parametry do joba
EXPORT_VARS="ALL,PLINDER_CLEAN_LIST=${CLEAN_LIST},CONFIG_TEMPLATE=${CONFIG_TEMPLATE},OUTPUT_DIR=${OUTPUT_DIR},PROJECT_ROOT=${PROJECT_ROOT}"

JOBID=$(sbatch "${SBATCH_FLAGS[@]}" --export="${EXPORT_VARS}" "${SCRIPT_DIR}/plinder_array.sbatch" | awk '{print $NF}')
echo "Submitted array job: ${JOBID}"
echo "Monitor: squeue -j ${JOBID} -o \"%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %a\""
echo "Throttle change (on the fly): scontrol update JobId=${JOBID} ArrayTaskThrottle=<new_n>"
