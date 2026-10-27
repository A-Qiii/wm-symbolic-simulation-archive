# Source from Bash. Local Python 3.11 learning/feature environment.
_wm_blocks_root="${WM_BLOCKS_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export WM_BLOCKS_ROOT="$_wm_blocks_root"
export PYTHONPATH="$_wm_blocks_root/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR="$_wm_blocks_root/cache/matplotlib"
. "$_wm_blocks_root/.envs/world_model/bin/activate"
unset _wm_blocks_root
