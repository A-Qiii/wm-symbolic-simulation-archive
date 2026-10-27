"""Portable resource locations; no imports from the original workstation project."""
import os
from pathlib import Path
ROOT=Path(os.environ.get('WM_BLOCKS_ROOT',Path(__file__).resolve().parents[1])).expanduser().resolve()
RUNTIME=ROOT
CALVIN_ENV=Path(os.environ.get('WM_BLOCKS_CALVIN_ENV',ROOT/'vendor/calvin/calvin_env')).expanduser().resolve()
DINO_WEIGHTS=Path(os.environ.get('WM_BLOCKS_DINO_WEIGHTS',ROOT/'external_models/dinov2-small-ed25f3a31f01632728cabb09d1542f84ab7b0056')).expanduser().resolve()
GPU_LOCK=Path(os.environ.get('WM_BLOCKS_GPU_LOCK',ROOT/'runs/gpu.lock')).expanduser().resolve()
_font=os.environ.get('WM_BLOCKS_CJK_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
CJK_FONT=Path(_font)if Path(_font).is_file()else None
