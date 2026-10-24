# Independent learned-model environment. Does not load HULC or old project code.
export WM_SYMBOLIC_ROOT=/home/robot/research/projects/wm_symbolic
export PYTHONPATH="$WM_SYMBOLIC_ROOT/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$WM_SYMBOLIC_ROOT/cache/matplotlib"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
. "$WM_SYMBOLIC_ROOT/.envs/world_model/bin/activate"
