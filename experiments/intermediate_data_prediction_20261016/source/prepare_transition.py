from config import *
import sys,torch,fcntl
sys.path.insert(0,str(REP/'source'))
from runtime import load
def main():
 assert not(E/'results/collection_D3.json').exists();selected={};queryfiles=[]
 for seed in [0,1]:
  choices=[]
  for arm in ['old_coverage','new_coverage']:
   s=read(E/'models'/f'query_{arm}_seed{seed}/selected.json');t=read(E/'results'/f'train_query_{arm}_seed{seed}.json');assert t['updates']==1200;assert sha(s['path'])==s['sha256'];choices.append((s['C_score'],arm,s));queryfiles.append(Path(s['path']))
  best=min(choices,key=lambda x:x[0]);selected[str(seed)]={'arm':best[1],**best[2]}
  a=read(E/'results'/f'train_query_old_coverage_seed{seed}.json');b=read(E/'results'/f'train_query_new_coverage_seed{seed}.json');assert a['initial_hash']==b['initial_hash']
 save(E/'configs/query_choices_frozen.json',{'frozen_utc':now(),'criterion':'perseed lower old/new C selection score; old first on exact tie','selected':selected,'sha256':{str(p):sha(p)for p in queryfiles},'C_validation_and_D_used':False})
 oldrows=read(REP/'data/public/states.json');oldpairs=read(REP/'data/public/pairs.json');oldchains=read(REP/'data/public/chains.json');oi=dict(np.load(REP/'data/public/inputs.npz'));newrows=read(E/'data/ABC/public/states.json');newpairs=read(E/'data/ABC/public/pairs.json');ni=dict(np.load(E/'data/ABC/public/inputs.npz'));keep=[i for i,r in enumerate(newrows)if r['split']in ['train','dev_select']];lookup={i:len(oldrows)+j for j,i in enumerate(keep)};states=oldrows+[r|{'dataset_group':'new'}for i,r in enumerate(newrows)if i in lookup];rows=[r|{'dataset_group':'old','index':i}for i,r in enumerate(oldpairs)]
 for r in newpairs:
  if r['split']not in ['train','dev_select']:continue
  rows.append(r|{'before':lookup[r['before']],'after':lookup[r['after']],'dataset_group':'new','index':len(rows)})
 o=np.concatenate([oi['objects'],ni['objects'][keep]]);s=np.concatenate([oi['scene'],ni['scene'][keep]]);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);m,q,stats=load(0,'canonical_matched')
 with torch.no_grad():z=m.encode(torch.tensor(o,device='cuda'),torch.tensor(s,device='cuda')).cpu().numpy()
 d=E/'data/transition_train';d.mkdir();np.savez_compressed(d/'inputs.npz',objects=o,scene=s,z=z);save(d/'pairs.json',rows);save(d/'states.json',states)
 oldab=[i for i,r in enumerate(rows)if r['split']=='train'and r['dataset_group']=='old'];newab=[i for i,r in enumerate(rows)if r['split']=='train'and r['dataset_group']=='new'];newnormal=[i for i in newab if rows[i]['branch']=='normal'];newrefusal=[i for i in newab if rows[i]['branch']=='held_closed'];assert newnormal and newrefusal
 def sample(rng,ix,n):
  roots=sorted({rows[i]['root_id']for i in ix});groups=[[i for i in ix if rows[i]['root_id']==r]for r in roots];return np.array([groups[j][rng.integers(len(groups[j]))]for j in rng.integers(len(groups),size=n)])
 for seed in[0,1]:
  rng=np.random.default_rng(16965000+seed);base=np.stack([sample(rng,oldab,48)for _ in range(1000)]);newaux=np.stack([np.r_[sample(rng,newnormal,12),sample(rng,newrefusal,4)]for _ in range(1000)]);oldaux=np.array([[sample(rng,[i for i in oldab if rows[i]['skill']==rows[j]['skill']],1)[0]for j in row]for row in newaux]);seq=np.full((1000,48,3),-1,np.int64)
  for u in range(1000):
   for j,i in enumerate(base[u]):
    for t,pid in enumerate(oldchains[i][:3]):seq[u,j,t]=int(pid[1:])
  assert np.all(seq[:,:,0]==base)
  for u in range(1000):
   for j in range(48):
    for t in [1,2]:
     if seq[u,j,t]>=0:assert rows[seq[u,j,t-1]]['after']==rows[seq[u,j,t]]['before']
  np.savez_compressed(E/'configs'/f'transition_schedule_seed{seed}.npz',base=base,old_aux=oldaux,new_aux=newaux,observed_chains=seq)
 save(E/'results/transition_data_audit.json',{'old_AB_transitions':len(oldab),'new_AB_transitions':len(newab),'new_AB_normal_branch':len(newnormal),'new_AB_held_closed_branch':len(newrefusal),'old_C_roots':len({r['root_id']for r in rows if r['split']=='dev'}),'new_C_selection_roots':len({r['root_id']for r in rows if r['split']=='dev_select'}),'C_validation_states_in_pack':0,'D_states_in_pack':0,'same_base_and_skill_matched_auxiliary':True,'selected_query':selected});print('TRANSITION_DATA_READY',len(oldab),len(newab),flush=True)
if __name__=='__main__':main()
