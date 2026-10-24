from config import *
from model import make,state_hash,loss
import torch,shutil,ast
def main():
 assert E.exists()and not(E/'configs/frozen.json').exists();torch.set_num_threads(4);refs={}
 def reg(p):p=Path(p);refs[str(p)]=sha(p);return p
 prior=[WHOLE/'configs/delivery_manifest.json',PROBE/'configs/delivery_manifest.json',COMBO/'configs/delivery_manifest.json'];prior_checks={str(p):verify_seal(p)for p in prior}
 for p in prior:reg(p)
 qn=dict(np.load(reg(WHOLE/'configs/query_normalization.npz')));cfgold=read(reg(WHOLE/'configs/protocol.json'));old=dict(np.load(reg(WHOLE/'data/train/control.npz')));raw=old['z']*qn['std']+qn['mean'];mean=raw[:,:3].reshape(-1,64).mean(0);std=np.maximum(raw[:,:3].reshape(-1,64).std(0),.01);np.savez_compressed(E/'configs/object_normalization.npz',mean=mean,std=std);np.savez_compressed(E/'configs/query_normalization.npz',**qn)
 for arm in ['control','coverage']:
  d=dict(np.load(reg(WHOLE/'data/train'/f'{arm}.npz')));assert all(str(r).startswith(('A_','B_'))for r in d['root_ids']);zraw=d['z']*qn['std']+qn['mean'];data={k:d[k]for k in ['y','v','root_ids','state_ids']};data['x']=((zraw[:,:3]-mean)/std).astype(np.float32);np.savez_compressed(E/'data/train'/f'{arm}.npz',**data)
  assert set(d['root_ids']).isdisjoint(cfgold['root_split']['validation']+cfgold['root_split']['diagnostic_D_seen'])
 for seed in [0,1]:
  p=reg(WHOLE/'configs'/f'schedule_seed{seed}.npz');shutil.copy2(p,E/'configs'/p.name);s=np.load(p);assert np.array_equal(s['control'][:,:24],s['coverage'][:,:24]);assert np.bincount(s['coverage'][:,24:].ravel()-156).tolist()==[800]*6
  torch.manual_seed(17075000+seed);q=make();path=E/'models'/f'initial_seed{seed}.pt';torch.save({'state':q.state_dict(),'initialization_seed':17075000+seed,'state_hash':state_hash(q)},path)
  test=np.load(E/'data/train/control.npz');l=loss(q(torch.tensor(test['x'][:32])).squeeze(-1),torch.tensor(test['y'][:32,3:6]),torch.tensor(test['v'][:32,3:6]),torch.tensor(cfgold['class_weights'][3:6]));assert torch.isfinite(l)
 for f in(WHOLE/'data/evaluation').glob('*'):reg(f)
 for name in ['actual_predictions.json','trajectory_predictions.json','sampling_budget.json']:reg(WHOLE/'results'/name)
 for name in ['mixed_seed0','latent_seed0','mixed_seed1','latent_seed1']:reg(COMBO/'results'/f'latents_{name}.npz')
 for name in ['data/combo/public/pairs.json','data/combo/_private/labels.json']:reg(COMBO/name)
 cfg={'utc':now(),'purpose':'Assess separate object In readout while preserving original formal Query Holding/drawer exactly. This changes architecture and objective, not a pure data-only comparison.','architecture':'shared64->32SiLU->1 MLP,2113parameters, corresponding context-enriched object token only','arms':['control','coverage'],'seeds':[0,1],'initialization_seeds':[17075000,17075001],'paired_initial_weights_identical':True,'batch_states':32,'objects_per_state':3,'updates':600,'schedule':'Exact previous24commonold+8replacement schedule, seed-specific. Control156oldstates;coverage156old+6dualstates. No additional states.','root_split':cfgold['root_split'],'normalization':'One common mean/std from156historicalABtrainingstates x3objecttokens;std floor0.01;no validation/C/D or added6states used in normalization.','optimizer':'freshAdamWlr0.001wd0.001,globalclip1','loss':'Only original In-group masked weightedBCE,mean overvalidobjects withinstate thenmean overstates;no Holding/drawer loss','positive_weights':cfgold['class_weights'][3:6],'fixed_outputs':'Original formal Query probability entries0:3 and6:9 copied unchanged for each seed/input; onlyentries3:6 replaced. No trueHolding/action mask.','selection':'fixedfinal600,no earlystop/hypersearch/thresholdfit;allfourrunsexposed','threshold':.5,'evaluation':'Actual all9,In3,all9facts confusion,dual/nondual,selfheldobjectIn andnotselfheld;rootpaired metrics,unknownmasks;then same savedmixed/latent one-step/recursivez;oldquery+previouswholequeryarms parallel.','D':'previouslyobserved diagnostic only,no train/norm/selection','saved_probe_C_review':'performed before thistraining;MLParchitecture already specified byauthorization,not selected by C','stop':'no automatic adoption,new collection or further tuning;stop at analysis if retention fails','world_model_updates':0,'new_physical_calls':0}
 save(E/'configs/protocol.json',cfg);save(E/'configs/source_references.json',{'sha256':refs,'prior_manifests':{str(p):sha(p)for p in prior},'prior_file_checks':prior_checks});shutil.copy2(WHOLE/'results/sampling_budget.json',E/'results/sampling_budget.json')
 for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text());shutil.copy2(f,E/'source'/f.name)
 for f in E.rglob('*'):
  if f.is_file()and '__pycache__'not in f.parts:refs[str(f)]=sha(f)
 save(E/'configs/frozen.json',{'utc':now(),'before_optimizer_creation':True,'sha256':refs,'allowed_fits':4,'updates_each':600});print('OBJECT_IN_FROZEN',len(refs),'refs;4fits;600updates;sharedhistoricalABnormalization',flush=True)
if __name__=='__main__':main()
