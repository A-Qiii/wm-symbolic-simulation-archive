"""Only future predictor entry point; no private annotations are imported."""
from common import E,validate_public
from pathlib import Path
import numpy as np
def load_input(path):
 path=Path(path).resolve()
 assert path.parent==(E/'data/model_inputs').resolve(),'private/target path denied'
 with np.load(path,allow_pickle=False)as f:a={k:f[k].copy()for k in f.files}
 validate_public(a);return a
