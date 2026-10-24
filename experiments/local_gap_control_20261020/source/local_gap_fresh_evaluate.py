from common import *
import torch,fcntl,shutil,ast
import data as d
from evaluate import assess
from wm_symbolic.public_geometry import visible_world_points,colour_masks
F=S/'fresh_test'
guard();verify();budget();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
assert read(F/'results/collection_complete.json')['passed']
node=next(n for n in ast.parse((OLD/'source/data.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='feature');exec(compile(ast.Module(body=[node],type_ignores=[]),'unchanged original public geometry','exec'))
shutil.copyfile(F/'cache/confirm_metadata.json',F/'cache/fresh_independent_metadata.json')
for seed in [0,1]:shutil.copyfile(F/f'cache/confirm_seed{seed}.npz',F/f'cache/fresh_independent_seed{seed}.npz')
eps=read(F/'cache/confirm_metadata.json');objects=[];contexts=[]
for e in eps:
 oo=[];cc=[]
 for item in e['features']:
  with np.load(item['public_inputs'][-1])as z:frame={k:z[k]for k in z.files}
  ob,ct,_=feature(frame);oo.append(ob);cc.append(ct)
 for dest,arr in [(objects,np.stack(oo)),(contexts,np.stack(cc))]:dest.append(np.concatenate([arr,np.repeat(arr[-1:],7-len(arr),axis=0)]))
np.savez_compressed(F/'cache/fresh_independent_geometry_raw.npz',objects=np.stack(objects),context=np.stack(contexts))
d.O=F;d.OLD=F;d.make('fresh_independent')
for seed in [0,1]:
 for arm in ARMS:assess(arm,seed,'fresh_independent')
save(S/'results/fresh_evaluation_complete.json',{'utc':now(),'fixed_weights_unchanged':True,'no_new_fit_or_normalization':True,'models':4,'new_test_complete':True})
