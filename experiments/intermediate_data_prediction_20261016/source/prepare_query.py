from config import *
import torch,collections
def main():
 old=dict(np.load(PREV/'data/head_train/original_coverage.npz'));oldstates=read(REP/'data/public/states.json');roots=sorted(set(old['root_ids']),key=lambda x:hashlib.sha256(('old96'+x).encode()).hexdigest());common=[]
 for rank,root in enumerate(roots):
  candidates=[i for i,r in enumerate(old['root_ids'])if r==root];candidates.sort(key=lambda i:(-int(np.any(old['y'][i,:6])or old['y'][i,8]),hashlib.sha256(str(old['state_ids'][i]).encode()).hexdigest()));common+=candidates[:6 if rank<6 else 5]
 assert len(common)==96;newstates=read(E/'data/ABC/public/states.json');labels=read(E/'data/ABC/_private/labels.json');raw=np.load(E/'data/ABC/public/inputs.npz')['z'];qc=torch.load(REP/'models/query_seed0.pt',map_location='cpu',weights_only=True);stats=qc['normalization'];newz=((torch.tensor(raw)-stats['mean'])/stats['std']).numpy();target={}
 for stratum in ['held_open','hover_above_drawer','post_place','held_closed']:
  rr={}
  for i,s in enumerate(newstates):
   if s['split']=='train'and stratum in s['strata']:rr.setdefault(s['root_id'],i)
  target[stratum]=sorted(rr.values());assert len(rr)>0,'Empty coverage stratum: '+stratum
 added=sorted(set(sum(target.values(),[])));assert len(added)<=48
 def pack_new(ix):return {'z':newz[ix],'y':np.array([labels[str(i)]['y']for i in ix],np.float32),'v':np.array([labels[str(i)]['v']for i in ix],np.float32),'state_ids':np.array([f'new:{i}'for i in ix]),'root_ids':np.array([newstates[i]['root_id']for i in ix]),'dataset':np.array(['new']*len(ix))}
 original={k:old[k]for k in ['z','y','v','root_ids']};original['state_ids']=np.array([f'old:{i}'for i in old['state_ids']]);original['dataset']=np.array(['old']*len(old['z']));new=pack_new(added);coverage={k:np.concatenate([original[k][common],new[k]])for k in original}
 (E/'data/query_train').mkdir();(E/'data/query_evaluator').mkdir()
 pools={'old_coverage':original,'new_coverage':coverage}
 for name,pool in pools.items():np.savez_compressed(E/'data/query_train'/f'{name}.npz',**pool)
 for split,name in [('dev_select','C_selection'),('dev_validate','C_validation')]:
  ix=[i for i,s in enumerate(newstates)if s['split']==split];np.savez_compressed(E/'data'/('query_train'if split=='dev_select'else'query_evaluator')/f'{name}.npz',**pack_new(ix))
 for name,source in [('old_C_selection',PREV/'data/head_train/C_selection.npz'),('old_C_validation',PREV/'data/evaluator/C_validation.npz')]:
  pp=dict(np.load(source));pp['state_ids']=np.array([f'old:{i}'for i in pp['state_ids']]);pp['dataset']=np.array(['old']*len(pp['z']));np.savez_compressed(E/'data'/('query_train'if'selection'in name else'query_evaluator')/f'{name}.npz',**pp)
 def sample(rng,items,rootlookup,n):
  rs=sorted({rootlookup[i]for i in items});groups=[[i for i in items if rootlookup[i]==r]for r in rs];return np.array([groups[j][rng.integers(len(groups[j]))]for j in rng.integers(len(groups),size=n)])
 oldids=original['state_ids'];newids=coverage['state_ids'];shared=[oldids[i]for i in common];lookup={str(i):str(r)for pool in pools.values()for i,r in zip(pool['state_ids'],pool['root_ids'])}
 for seed in [0,1]:
  rng=np.random.default_rng(6100+seed);base=np.array([sample(rng,shared,lookup,20)for _ in range(1200)]);oldaux=np.array([sample(rng,list(oldids),lookup,12)for _ in range(1200)]);newaux=np.array([np.concatenate([sample(rng,[f'new:{i}'for i in target[k]],lookup,3)for k in target])for _ in range(1200)])
  for arm,aux in [('old_coverage',oldaux),('new_coverage',newaux)]:
   ids=np.concatenate([base,aux],1);mapping={str(i):j for j,i in enumerate(pools[arm]['state_ids'])};ix=np.array([[mapping[str(i)]for i in row]for row in ids]);np.savez_compressed(E/'configs'/f'query_schedule_{arm}_seed{seed}.npz',indices=ix,state_ids=ids)
 qcfg={'created_utc':now(),'class_weights':read(OLD/'results/query_interaction_seed0.json')['class_weights'],'new_strata_indices':target,'shared_old_ids':[int(old['state_ids'][i])for i in common],'new_indices':added,'training_pool_sizes':{k:len(v['z'])for k,v in pools.items()},'new_samples_chosen_by_phase_and_root_not_success':True,'original_input_sha256':sha(PREV/'data/head_train/original_coverage.npz')};save(E/'configs/query_inputs.json',qcfg)
 trainroots=set(original['root_ids'])|set(coverage['root_ids']);cs=set(np.load(E/'data/query_train/C_selection.npz')['root_ids']);cv=set(np.load(E/'data/query_evaluator/C_validation.npz')['root_ids']);assert not(trainroots&cs or trainroots&cv or cs&cv)
 save(E/'results/query_input_audit.json',{'pool_sizes':qcfg['training_pool_sizes'],'new_strata_counts':{k:len(v)for k,v in target.items()},'AB_roots':len({newstates[i]['root_id']for i in added}),'C_select_roots':len(cs),'C_validation_roots':len(cv),'split_overlap':0,'D_collected_before_training':False});print('QUERY_PACKS_READY',qcfg['training_pool_sizes'],{k:len(v)for k,v in target.items()},flush=True)
if __name__=='__main__':main()
