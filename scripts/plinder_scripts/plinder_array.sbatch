#!/usr/bin/env bash
#SBATCH --mail-type=FAIL,TIME_LIMIT
# (Reszta zasobów dostarczamy z linii poleceń 'sbatch' w launcherze)

set -euo pipefail

echo "=============================================================================="
echo "SLURM Job Info"
echo "Array JobID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Task ID:     ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Hostname:    $(hostname)"
echo "Started at:  $(date)"
echo "=============================================================================="

# ====== Walidacja env od launchera ======
: "${PLINDER_CLEAN_LIST:?PLINDER_CLEAN_LIST not exported}"
: "${CONFIG_TEMPLATE:?CONFIG_TEMPLATE not exported}"
: "${OUTPUT_DIR:?OUTPUT_DIR not exported}"
: "${PROJECT_ROOT:?PROJECT_ROOT not exported}"

# ====== Pobierz konkretny PLINDER_ID dla tego indeksu tablicy ======
INDEX=$((SLURM_ARRAY_TASK_ID + 1))  # sed jest 1-indeksowane
PLINDER_ID="$(sed -n "${INDEX}p" "${PLINDER_CLEAN_LIST}")"

if [[ -z "${PLINDER_ID}" ]]; then
  echo "Error: empty PLINDER_ID for index ${SLURM_ARRAY_TASK_ID}"
  exit 2
fi

echo "Processing PLINDER_ID='${PLINDER_ID}' (index ${SLURM_ARRAY_TASK_ID})"

# ====== Moduły / conda (dopasowane do Twojego klastra) ======
# module load cuda/11.8
module load anaconda
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate molecular_dynamics_pipeline
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || (echo "NO GPU AVAILABLE — FAIL" && exit 99)


# ====== Praca w repo ======
cd "${PROJECT_ROOT}"

# Uwaga: NIE nadpisujemy CUDA_VISIBLE_DEVICES — SLURM ustawia to poprawnie dla --gpus

# ====== Idempotencja (opcjonalnie) – pominij jeśli wynik już istnieje ======
# Przyklad pliku "done" — zmodyfikuj do swojego pipeline'u:
DONE_FLAG="${OUTPUT_DIR}/${PLINDER_ID}/_JOB_DONE"
if [[ -f "${DONE_FLAG}" ]]; then
  echo "Skip: found ${DONE_FLAG}"
  exit 0
fi

# ====== Uruchomienie pojedynczego taska ======
echo ">>> Running run_single_plinder.sh"
bash plinder_scripts/run_single_plinder.sh "${PLINDER_ID}" "${CONFIG_TEMPLATE}" "${OUTPUT_DIR}"
EC=$?


echo "=============================================================================="
echo "Finished at: $(date)"
echo "=============================================================================="
exit "${EC}"
