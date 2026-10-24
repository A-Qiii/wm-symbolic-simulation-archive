from common import *
import torch,fcntl,shutil
import data as d
from evaluate import assess
F=S/'fresh_test'
guard();verify();budget();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
assert read(F/'results/collection_complete.json')['passed']
shutil.copyfile(F/'cache/confirm_metadata.json',F/'cache/fresh_independent_metadata.json')
old_o=d.O;d.O=F;d.make_cache('fresh_independent');d.O=old_o
original_source=d.source_cache
def fresh_source(split,seed):
 if split=='fresh_independent':return dict(np.load(F/f'cache/confirm_seed{seed}.npz'))
 return original_source(split,seed)
d.source_cache=fresh_source
for seed in [0,1]:
 for arm in ARMS:assess(arm,seed,'fresh_independent')
save(S/'results/fresh_evaluation_complete.json',{'utc':now(),'fixed_weights_unchanged':True,'no_new_fit_or_normalization':True,'models':4,'new_test_complete':True})
