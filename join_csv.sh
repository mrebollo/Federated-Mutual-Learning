#!/bin/bash
# Une todos los CSV de resultados de experimentos expXXX en un solo archivo.
# Uso: ./join_csv.sh /ruta/a/experiments
base_dir=${1:-$(pwd)}
output="$base_dir/all_joined.csv"

header='job_id,dataset,model,net,size,iid,alg,rep,round,agent,acc,loss,err,nmsg,msiz'

echo "$header" > "$output"

find "$base_dir" -type f -name 'results.csv' | sort | while read -r file; do
  dir=$(dirname "$file")
  base=$(basename "$dir")
  if [[ "$base" =~ ^exp_0*([0-9]+)$ ]]; then
    job_id=${BASH_REMATCH[1]}
  else
    job_id="$base"
  fi
  tail -n +2 "$file" | awk -v jid="$job_id" '{print jid","$0}' >> "$output"
done

echo "Joined CSV saved to $output"
