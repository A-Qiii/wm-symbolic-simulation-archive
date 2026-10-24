#!/usr/bin/env bash
set -euo pipefail
T=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
P=$(cd -- "$T/../.." && pwd)
if [[ -e "$T/data/observed_latents_seed0.npz" || -e "$T/results/coordinates_seed0.csv" ]]; then
  echo 'Refusing to overwrite existing run; copy source/configs into a NEW sibling experiment directory.' >&2
  exit 1
fi
mkdir -p "$T"/{data,results,figures,logs,cache,cache/matplotlib}
source "$P/world_model_env.sh"
export MPLCONFIGDIR="$T/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
"$P/.envs/world_model/bin/python" "$T/source/extract.py" > "$T/logs/extraction.log" 2>&1
for script in project scene render; do
  "$P/.envs/executor/bin/python" "$T/source/$script.py" > "$T/logs/$script.log" 2>&1
done
