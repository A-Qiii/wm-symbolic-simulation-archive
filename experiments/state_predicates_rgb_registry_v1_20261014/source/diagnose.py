from pathlib import Path
import json,sys,hashlib,fcntl
import numpy as np
import torch
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_rgb_registry_v1_20261014';OLD=P/'experiments/state_predicates_long_pairs_20261014';CAND=P/'experiments/state_predicates_held_down_20261014/candidate';sys.path.insert(0,str(CAND))
from model import PredicateReader
from metrics import summarize,known
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not(E/'results/reader_diagnostic.json').exists();assert read(E/'results/rebuild.json')['passed'];scope=read(E/'configs/scope.json')
 for p,h in scope['frozen_sha256'].items():assert sha(p)==h,p
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');model=PredicateReader().cuda();ck=torch.load(CAND/'reader_seed1.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model']);model.eval().requires_grad_(False);meta=read(E/'data/public/windows_v1.json');pred={}
 for version,path in [('original',OLD/'data/public/paired_inputs_full.npz'),('rgb_v1',E/'data/public/paired_inputs_v1.npz')]:
  with np.load(path)as z,torch.inference_mode():pred[version]=model.probabilities(**{k:torch.as_tensor(z[k],device='cuda')for k in ['objects','context','robot']}).cpu().numpy()
 np.savez_compressed(E/'results/reader_probabilities.npz',**pred)
 with np.load(E/'data/_private/supervision.npz')as z:y=z['window_y'];v=z['window_valid']
 result={};details=[]
 for version,pr in pred.items():
  kk=known(pr,[.5,.5,.5]);hat=pr>=.5;result[version]={}
  for split in ['train','dev']:
   ii=np.array([r['split']==split for r in meta]);rootids=sorted({r['root_id']for r in meta if r['split']==split});allcorrect=(v&kk&(hat==y)).all(1);rs=[{'root_id':rid,'all5_windows_known_correct':bool(allcorrect[[i for i,r in enumerate(meta)if r['root_id']==rid]].all())}for rid in rootids];metrics=summarize(pr[ii],y[ii],v[ii],[.5,.5,.5]);gate=all((q['positive_recall']is None or q['positive_recall']>=.95)and(q['FPR']is None or q['FPR']<=.01)for q in metrics.values());result[version][split]={'windows':int(ii.sum()),'all7_known_correct_windows':int(allcorrect[ii].sum()),'raw_error_facts':int(((hat!=y)&v)[ii].sum()),'unknown_facts':int((~kk&v)[ii].sum()),'all5_windows_correct_roots':sum(r['all5_windows_known_correct']for r in rs),'roots':rs,'metrics':metrics,'predeclared_current_reader_gate_passed':gate}
 for i,r in enumerate(meta):
  detail={'window_id':r['window_id'],'root_id':r['root_id'],'split':r['split'],'truth':y[i].tolist(),'valid':v[i].tolist(),'new_missing_entire_window':r['missing_entire_window']}
  for ver,pr in pred.items():
   kk=known(pr[i:i+1],[.5,.5,.5])[0];detail[ver]={'probabilities':pr[i].tolist(),'raw_error_columns':np.flatnonzero(((pr[i]>=.5)!=y[i])&v[i]).tolist(),'unknown_columns':np.flatnonzero(~kk&v[i]).tolist(),'all7_known_correct':bool((kk&v[i]&((pr[i]>=.5)==y[i])).all())}
  details.append(detail)
 output={'status':'development_diagnostic_only_input_interface_changed','checkpoint_sha256':sha(CAND/'reader_seed1.pt'),'thresholds':[.5,.5,.5],'frozen_reader_forward_batches':2,'windows_each_batch':150,'training_updates':0,'new_D':False,'original_acceptance_does_not_transfer_to_new_interface':True,'metrics':result,'details':details,'no_rule_or_threshold_changes_after_C':True};save(E/'results/reader_diagnostic.json',output);print(json.dumps({k:v for k,v in output.items()if k!='details'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
