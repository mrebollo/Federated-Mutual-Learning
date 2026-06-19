#!/bin/bash
#SBATCH --job-name=fml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=slurm-%A_%a.out
#SBATCH --mail-user=mrebollo@upv.es
#SBATCH --profile=task

# genera el fichero de configuracion
cd ~/fml

config=${1:-config.txt}
base_dir=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}

if [[ "$config" != /* ]]; then
  config="$base_dir/$config"
fi

if [ ! -f "$config" ]; then
  echo "Error: config file '$config' not found"
  exit 1
fi

ROUNDS=200
EPOCHS=5
BATCHSIZE=128
REPS=1
NODE_NUM=5

ArrayId=$SLURM_ARRAY_TASK_ID

dataset=$(awk -v ArrayTaskID=$ArrayId '$1==ArrayTaskID {print $2}' "$config")
partition=$(awk -v ArrayTaskID=$ArrayId '$1==ArrayTaskID {print $3}' "$config")
model=$(awk -v ArrayTaskID=$ArrayId '$1==ArrayTaskID {print $4}' "$config")
algorithm=$(awk -v ArrayTaskID=$ArrayId '$1==ArrayTaskID {print $5}' "$config")

if [ -z "$dataset" ] || [ -z "$model" ] || [ -z "$algorithm" ]; then
  echo "Error: experiment parameters not found for ArrayTaskID=$ArrayId"
  exit 1
fi

# Activa el entorno virtual en el que se ha instalado TensorFlow
source ~/miniconda3/bin/activate mut 


printf -v NUM "%03d" $ArrayId
NAME="exp_${NUM}"
NOW=$(date +"%Y%m%d")
OUTDIR="$base_dir/experiments/$NOW/$NAME"
mkdir -p "$OUTDIR"
cd "$OUTDIR"
mkdir -p ./saves/record

# Diagnostics in Slurm output to verify GPU visibility inside the job.
echo "[fml-runner-v2] Using script: $0"
echo "[$(date)] Host: $(hostname)"
echo "[$(date)] SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
nvidia-smi || true

if command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n mut python)
else
  echo "Error: conda not found in PATH"
  exit 1
fi

"${PYTHON_CMD[@]}" - <<'PY'
import sys
import torch
print('python', sys.executable)
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_version', torch.version.cuda)
print('device_count', torch.cuda.device_count())
PY

# "${PYTHON_CMD[@]}" 
python3 $base_dir/main.py \
  --dataset "$dataset" \
  --node_num "$NODE_NUM" \
  --iid "$partition" \
  --local_model "$model" \
  --global_model "$model" \
  --algorithm "$algorithm" \
  --R "$ROUNDS" \
  --E "$EPOCHS" \
  --batchsize "$BATCHSIZE" \
  --notes "$NAME" \
  --download True
