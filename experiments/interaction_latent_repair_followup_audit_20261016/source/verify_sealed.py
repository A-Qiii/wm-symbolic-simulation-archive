"""Wrap original post-hoc verifier's missing sys import without editing its seal."""
import sys,importlib.util
from pathlib import Path
E=Path('/home/robot/research/projects/wm_symbolic/experiments/interaction_latent_repair_20261016')
sys.path.insert(0,str(E/'source'))
spec=importlib.util.spec_from_file_location('sealed_verifier',E/'source/verify.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.sys=sys
module.main()
