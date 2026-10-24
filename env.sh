# Source to use only the new project's executor. Existing apps/envs are not touched.
export WM_SYMBOLIC_ROOT=/home/robot/research/projects/wm_symbolic
export PYTHONPATH="$WM_SYMBOLIC_ROOT/src:$WM_SYMBOLIC_ROOT/vendor/calvin/calvin_env:$WM_SYMBOLIC_ROOT/vendor/calvin/calvin_models:$WM_SYMBOLIC_ROOT/vendor/hulc"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYOPENGL_PLATFORM=egl
export EGL_VISIBLE_DEVICES=0
export MPLCONFIGDIR="$WM_SYMBOLIC_ROOT/cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
. "$WM_SYMBOLIC_ROOT/.envs/executor/bin/activate"
