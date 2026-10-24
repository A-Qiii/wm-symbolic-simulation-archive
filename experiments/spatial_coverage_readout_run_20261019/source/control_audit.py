from common import *
import torch,ast
sys.addaudithook(guard)
# Load only the immutable Head definition, with no original writer or pipeline invocation.
ns={'nn':torch.nn,'torch':torch};tree=ast.parse((O/'source/common.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Head');exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen Head>','exec'),ns);Head=ns['Head']
def th(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
rows=[];members=read(O/'configs/protocol.json')['members']['train'];mapping=[]
for seed in [0,1]:
 raw=dict(np.load(O/f'data/features_seed{seed}_train.npz'));roots=sorted(set(raw['root_id']));assert roots==members and len(raw['cat'])==600
 for i,r in enumerate(roots):
  indices=np.where(raw['root_id']==r)[0];assert len(indices)==15
  expected=[(f'{r}_b{b}',h)for b in range(3)for h in range(5)]
  assert [(str(raw['episode_id'][j]),int(raw['step'][j]))for j in indices]==expected
  if seed==0:
   new=f'{r[0]}_coverage_train_{i%20:02d}'
   for j,(eid,h)in zip(indices,expected):mapping.append({'index':int(j),'old_root':r,'new_root':new,'branch':int(eid[-1]),'boundary':h,'old_episode':eid,'new_episode':f'{new}_b{eid[-1]}'})
 rng=np.random.default_rng(261019800+seed);groups=[np.where(raw['root_id']==r)[0]for r in roots]
 schedule=np.array([[groups[j][rng.integers(15)]for j in rng.integers(40,size=64)]for _ in range(1500)])
 assert np.array_equal(schedule,np.load(O/f'configs/schedule_seed{seed}.npy'))
 torch.manual_seed(261019800+seed);head=Head();init=th(head.state_dict());tr=read(O/f'results/train_object_seed{seed}.json');assert init==tr['initial_hash']
 cp=O/f'models/head_object_seed{seed}.pt';ck=torch.load(cp,map_location='cpu',weights_only=True);assert th(ck['state'])==tr['final_hash'] and sha(cp)==tr['weights_sha256']
 assert tr['updates']==1500 and tr['parameters']==309165 and ck['initial_hash']==init
 assert sha(O/f'configs/schedule_seed{seed}.npy')==tr['schedule_sha256']
 rows.append({'seed':seed,'head_sha256':sha(cp),'initial_hash':init,'final_hash':tr['final_hash'],'feature_sha256':sha(O/f'data/features_seed{seed}_train.npz'),'schedule_sha256':tr['schedule_sha256'],'schedule_exact_reconstructed':True,'index_mapping_exact':True})
source=(O/'source/run.py').read_text()
checks={'same_determinism_and_precision':"torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high')"in source,'no_class_weights':'cross_entropy(ob.reshape(-1,8),y[:,:3].reshape(-1))'in source,'equal4field_loss':'loss=.75*lo+.25*ld'in source,'adamw_lr_wd':'AdamW(model.parameters(),lr=.0003,weight_decay=.0001)'in source,'separate_clips':'clip_grad_norm_(params_o,1.)'in source and 'clip_grad_norm_(params_d,1.)'in source,'strict_majority_route':"pixel_colors[name][mask].mean())>.5"in source,'frozen_common_norm':"B/'cells/roots40/configs/normalization.json'"in source}
assert all(checks.values())
save(S/'configs/paired_boundary_indices.json',mapping)
save(S/'results/control_reuse_audit.json',{'utc':now(),'passed':True,'refits_required':False,'extra_control_updates':0,'seeds':rows,'source_checks':checks,'normalization_sha256':sha(B/'cells/roots40/configs/normalization.json'),'head_definition_ast_sha256':hashlib.sha256(ast.dump(node).encode()).hexdigest(),'source_sha256':sha(O/'source/run.py'),'interpretation':'Exact schedule, index, init, sealed weights and fixed implementation verified; does not assert novel outcomes.'})
print('CONTROL_REUSE_AUDIT_PASS',flush=True)
