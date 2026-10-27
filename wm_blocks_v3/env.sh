# Source from Bash. Local Python 3.10 simulation environment, no HULC dependency.
_wm_blocks_root="${WM_BLOCKS_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export WM_BLOCKS_ROOT="$_wm_blocks_root"
export PYTHONPATH="$_wm_blocks_root/src:${WM_BLOCKS_CALVIN_ENV:-$_wm_blocks_root/vendor/calvin/calvin_env}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYOPENGL_PLATFORM=egl EGL_VISIBLE_DEVICES=0
export MPLCONFIGDIR="$_wm_blocks_root/cache/matplotlib"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
. "$_wm_blocks_root/.envs/executor/bin/activate"
unset _wm_blocks_root
