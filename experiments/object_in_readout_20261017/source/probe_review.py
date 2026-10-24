"""Read-only evaluation of two saved object-token probes on historical native-label C."""
from config import *
import torch,csv,shutil
from torch import nn
def conf(y,p,v):
 y,p,v=map(lambda x:np.asarray(x,bool),(y,p,v));tp=int((v&y&p).sum());fn=int((v&y&~p).sum());fp=int((v&~y&p).sum());tn=int((v&~y&~p).sum());return dict(TP=tp,FN=fn,FP=fp,TN=tn)
def main():
 assert not E.exists();verify_seal(WHOLE/'configs/delivery_manifest.json');verify_seal(PROBE/'configs/delivery_manifest.json')
 for sub in ['source','configs','data/train','models','results','audit','logs']:(E/sub).mkdir(parents=True)
 R.mkdir();torch.set_num_threads(4);norm=np.load(WHOLE/'configs/query_normalization.npz');pn=np.load(PROBE/'models/normalization_object.npz');models={};rows=[];preds=[];old=read(WHOLE/'results/actual_predictions.json');refpaths=[WHOLE/'configs/delivery_manifest.json',PROBE/'configs/delivery_manifest.json',WHOLE/'configs/query_normalization.npz',PROBE/'models/normalization_object.npz']
 for arch in ['linear','mlp32']:
  path=PROBE/'models'/f'object_{arch}.pt';refpaths.append(path);q=nn.Linear(64,1)if arch=='linear'else nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1));q.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state']);q.eval();models[arch]=q
 for name in ['old_C_selection','old_C_validation','C_selection','C_validation']:
  file=WHOLE/'data/evaluation'/f'{name}.npz';refpaths.append(file);d=dict(np.load(file));raw=d['z']*norm['std']+norm['mean'];x=(raw[:,:3]-pn['mean'])/pn['std'];y=d['y'][:,3:6].astype(bool);v=d['v'][:,3:6].astype(bool);held=d['y'][:,:3].astype(bool)&d['v'][:,:3].astype(bool)
  candidates={}
  for arch,q in models.items():
   with torch.inference_mode():candidates['saved_object_'+arch]=q(torch.tensor(x)).squeeze(-1).sigmoid().numpy()
  for seed in [0,1]:
   rr=sorted([r for r in old if r['cohort']==name and r['query_arm']=='original'and r['seed']==seed],key=lambda r:r['state_index']);candidates[f'original_query_seed{seed}']=np.array([r['probability'][3:6]for r in rr])
  for model,p in candidates.items():
   pp=p>=.5;full=v.all(1);ok=full&(pp==y).all(1);rows.append({'cohort':name,'model':model,'states':len(y),'roots':len(set(d['root_ids'])),'In3_correct':int(ok.sum()),'In3_N':int(full.sum()),'unknown_In_fact_cells':int((~v).sum()),**conf(y,pp,v),**{'self_held_'+k:val for k,val in conf(y,pp,v&held).items()},'self_held_unknown_In_fact_cells':int((held&~v).sum())});preds.append({'cohort':name,'model':model,'In_probability':p})
 save(E/'results/saved_probe_C_review.json',rows);save(E/'results/saved_probe_C_predictions.json',preds);save(E/'configs/probe_review_references.json',{'utc':now(),'sha256':{str(p):sha(p)for p in refpaths},'fits':0,'C_never_used_to_fit_saved_probes':True,'normalization':'Undo original Query normalization; apply saved probe AB-only normalization; input corresponding context-enriched object token64.','self_held':'Ground-truth Holding used only to define evaluation subgroup, never to mask predictions.'})
 with(R/'saved_probe_C_review.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 shutil.copy2(__file__,E/'source/probe_review.py');shutil.copy2(Path(__file__).with_name('config.py'),E/'source/config.py');print('SAVED_PROBE_C_REVIEW '+json.dumps(rows),flush=True)
if __name__=='__main__':main()
