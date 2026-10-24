"""CPU-only frozen old-reader comparator and scoring; never trains."""
from common import *
import torch,importlib.util,csv
def main():
 assert not(E/'results/perception_development.json').exists();torch.set_num_threads(4);xx=dict(np.load(BASE/'data/public/reader_inputs.npz'));tt=dict(np.load(BASE/'data/_private/reader_targets.npz'));mm=read(BASE/'data/public/windows.json');new=np.load(E/'results/reader_probabilities.npy');spec=importlib.util.spec_from_file_location('baseline_reader',CAND/'model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);net=mod.PredicateReader().eval().requires_grad_(False);net.load_state_dict(torch.load(CAND/'reader_seed1.pt',map_location='cpu',weights_only=False)['model'])
 with torch.inference_mode():old=net.probabilities(**{k:torch.tensor(v)for k,v in xx.items()}).numpy()
 np.save(E/'results/old_reader_development.npy',old);results={};errors=[]
 for name,p in [('new_query_reader',new),('old_frozen_reader',old)]:
  results[name]={};co=known(p)&((p>=.5)==tt['y'])&tt['v']
  for split in ['train','dev']:
   for source in ['all','success_chain','state_conditions','contact_hard_examples']:
    ids=np.array([i for i,r in enumerate(mm)if r['split']==split and(source=='all'or r['source']==source)]);result=metrics(p,tt['y'],tt['v'],ids);result['independent_roots']=len({mm[i]['root_id']for i in ids});result['roots_all_correct']=sum(bool(co[[i for i in ids if mm[i]['root_id']==root]].all())for root in {mm[i]['root_id']for i in ids});results[name][split+':'+source]=result
  for i,r in enumerate(mm):
   if r['split']=='dev'and not co[i].all():errors.append({'model':name,'window_id':r['window_id'],'root_id':r['root_id'],'source':r['source'],'condition':r['condition'],'error_columns':np.flatnonzero(~co[i]).tolist(),'probabilities':p[i].tolist(),'truth':tt['y'][i].astype(int).tolist(),'public_frames':r['public_frames']})
 save(E/'results/perception_development.json',{'metrics':results,'errors':errors,'comparison_limit':'Old reader trained previously on a larger different corpus, so this is a functional baseline, not a capacity/data-controlled improvement claim.','test_used':False,'new_updates':0});print(json.dumps({n:{s:r['all7']for s,r in q.items()}for n,q in results.items()}),flush=True)
if __name__=='__main__':main()
