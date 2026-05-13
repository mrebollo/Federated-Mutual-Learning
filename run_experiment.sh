#!/bin/bash
#SBATCH --job-name=fml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com

config=${1:-config.txt}
base_dir=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}

if [[ "$config" != /* ]]; then
  config="$base_dir/$config"
fi

if [ ! -f "$config" ]; then
  echo "Error: config file '$config' not found"
  exit 1
fi

ROUNDS=50
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

printf -v NUM "%03d" $ArrayId
NAME="exp_${NUM}"
NOW=$(date +"%Y%m%d")
OUTDIR="$base_dir/experiments/$NOW/$NAME"
mkdir -p "$OUTDIR"
cd "$OUTDIR"

source "$base_dir/.venv/bin/activate"

python "$base_dir/main.py" \
  --dataset "$dataset" \
  --node_num "$NODE_NUM" \
  --iid "$partition" \
  --local_model "$model" \
  --global_model "$model" \
  --algorithm "$algorithm" \
  --R "$ROUNDS" \
  --E "$EPOCHS" \
  --batchsize "$BATCHSIZE" \
  --notes "$NAME"
