#!/bin/bash
# Lanzador principal de experimentos. Uso: ./launch.sh [config.txt]
config=${1:-config.txt}
base_dir=$(cd "$(dirname "$0")" && pwd)

if [ ! -f "$base_dir/$config" ]; then
  echo "Error: fichero '$config' no encontrado en $base_dir"
  exit 1
fi

N=$(tail -n +2 "$base_dir/$config" | wc -l)
echo "Launching $N experiments from $config"
sbatch --array=1-$N "$base_dir/run_experiment.sh" "$base_dir/$config"
