"""One frozen final inference pass. Saved predictions support later analysis."""
from paths import GPU_LOCK
from paths import ROOT as B, RUNTIME as P
import os,json,hashlib,fcntl,csv
from pathlib import Path
import numpy as np,torch
from models import RGBState,Dynamics
def read(p):return json.load(open(p))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor(v):return torch.tensor(np.asarray(v),device='cuda',dtype=torch.float32)
def action(a):return tuple(torch.tensor([x],device='cuda',dtype=torch.long)for x in [0 if a[0]=='Pick'else 1,0 if a[1]=='large'else 1,{None:0,'A':1,'B':2,'large_top':3}[a[2]]])
def decode(state,est):
 r=est.read_regions(state[:,:2,128:131]).argmax(-1)[0].cpu().tolist();z=(state[:,:2,134].sigmoid()>=.5)[0].cpu().tolist();return {'region':r,'held':z,'xyz':state[0,:2,128:131].cpu().tolist()}
def gt(s):return {k:[s['objects'][n][k]for n in ['large','small']]for k in ['region','held']}
def same(p,t,which='joint'):
 if which=='region':return p['region']==t['region']
 return p['region']==t['region']and p['held']==t['held']
def main():
 version=os.environ.get('COLLECTION_VERSION','v1');frozen=read(B/'configs/all_models_frozen.json');assert frozen['development_gate_passed']and frozen['complete'];dest=B/'reports'/version/'confirmation';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'complete.json').exists():print('CONFIRMATION_ALREADY_COMPLETED');return
 plan=read(B/'configs/experiment_plan.json');meta=read(B/f'cache/{version}/test/state_labels.json');features=np.load(B/f'cache/{version}/test/all_features.npz')['features'];index={(r['episode_id'],r['boundary']):i for i,r in enumerate(meta)};allpred=[];summaries=[];bad_initial={x['id']for x in read(B/f'data/{version}/test_quality.json')['bad_initializations']}
 for rec in frozen['models']:
  assert sha(rec['checkpoint'])==rec['sha256']and sha(rec['state_checkpoint'])==rec['state_sha256'];base=Path(rec['checkpoint']).parent;normal=np.load(base/'feature_normalization.npz');est=RGBState(features.shape[-1]).cuda();est.load_state_dict(torch.load(rec['state_checkpoint'],map_location='cuda',weights_only=False)['model']);est.eval().requires_grad_(False);net=Dynamics(rec['condition']).cuda();net.load_state_dict(torch.load(rec['checkpoint'],map_location='cuda',weights_only=False)['model']);net.eval().requires_grad_(False);preds=[];counts={k:[0,0]for k in ['current_all','current_after_pick','one','root_terminal','root_interval','feedback_end','region_one','region_root_terminal','non_target_one','non_target_feedback_end']};corrections=[0,0];invalid=0
  for eid in plan['experiments'][rec['experiment']]['arms'][rec['arm']]['test']:
   e=read(B/f'data/{version}'/rec['experiment']/eid/'episode.json');case={'episode':eid,'group':e['group'],'recipe':e['recipe'],'physical_valid':e['physical_valid'],'model':{k:rec[k]for k in ['experiment','arm','condition','seed']}}
   if not e['physical_valid']or eid in bad_initial:
    invalid+=1;case['failure']='initialization_stratum'if eid in bad_initial else'physical_or_initialization';case['actions']=e['actions'];preds.append(case);continue
   assert len(e['actions'])==2
   rows=[index[(eid,i)]for i in range(3)];x=tensor((features[rows]-normal['mean'])/normal['std'])
   with torch.no_grad():
    h,p,shape,z=est(x);posts=Dynamics.pack(h,p,shape,z);p1=net(posts[:1],*action(e['actions'][0]['action']),est.sd);p2_free=net(p1,*action(e['actions'][1]['action']),est.sd);p2_feedback=net(posts[1:2],*action(e['actions'][1]['action']),est.sd);cur=[decode(posts[i:i+1],est)for i in range(3)];pr1=decode(p1,est);pr2=decode(p2_free,est);pf2=decode(p2_feedback,est)
   truth=[gt(s)for s in e['states']];case.update({'truth':truth,'current':cur,'prior_after_pick':pr1,'free_endpoint':pr2,'feedback_endpoint':pf2,'actions':[a['action']for a in e['actions']]});actor=0 if e['actions'][1]['action'][1]=='large'else 1;other=1-actor
   for i in range(3):counts['current_all'][0]+=same(cur[i],truth[i]);counts['current_all'][1]+=1
   for name,pred,target in [('current_after_pick',cur[1],truth[1]),('one',pr1,truth[1]),('one',pf2,truth[2]),('root_terminal',pr2,truth[2]),('feedback_end',pf2,truth[2]),('region_one',pr1,truth[1]),('region_one',pf2,truth[2]),('region_root_terminal',pr2,truth[2])]:counts[name][0]+=same(pred,target,'region'if name.startswith('region_')else'joint');counts[name][1]+=1
   counts['root_interval'][0]+=bool(same(pr1,truth[1])and same(pr2,truth[2]));counts['root_interval'][1]+=1
   for pred,target in [(pr1,truth[1]),(pf2,truth[2])]:counts['non_target_one'][0]+=bool(pred['region'][other]==target['region'][other]and pred['held'][other]==target['held'][other]);counts['non_target_one'][1]+=1
   counts['non_target_feedback_end'][0]+=bool(pf2['region'][other]==truth[2]['region'][other]and pf2['held'][other]==truth[2]['held'][other]);counts['non_target_feedback_end'][1]+=1
   corrections[0]+=bool(not same(pr2,truth[2])and same(pf2,truth[2]));corrections[1]+=bool(same(pr2,truth[2])and not same(pf2,truth[2]));case['correct']={'free':same(pr2,truth[2]),'feedback':same(pf2,truth[2]),'interval':bool(same(pr1,truth[1])and same(pr2,truth[2]))};preds.append(case)
  pairs={}
  for x in preds:pairs.setdefault(x['group'],[]).append(x)
  paired={name:[sum(len(xs)==2 and all(x.get('correct',{}).get(name,False)for x in xs)for xs in pairs.values()),20]for name in ['free','feedback','interval']};summary={**{k:rec[k]for k in ['experiment','arm','condition','seed']},'counts':counts,'paired':paired,'feedback_corrected':corrections[0],'feedback_new_errors':corrections[1],'planned_episodes':40,'valid_episodes':40-invalid,'planned_groups':20,'invalid_physical_episodes':invalid};summaries.append(summary);allpred.extend(preds);save(dest/f'{rec["experiment"]}_{rec["arm"]}_{rec["condition"]}_seed{rec["seed"]}.json',{'summary':summary,'predictions':preds});print('CONFIRM',summary,flush=True);del net,est;torch.cuda.empty_cache()
 save(dest/'all_predictions.json',allpred);save(dest/'summary.json',summaries);save(dest/'complete.json',{'models':len(summaries),'seeds_reported_separately':True,'inference_frozen_before_test':True,'no_test_tuning':True,'plan_sha256':sha(B/'configs/experiment_plan.json'),'models_manifest_sha256':sha(B/'configs/all_models_frozen.json')});save(B/'STATUS.json',{'stage':'offline_confirmation_complete','models':len(summaries),'next':'actual model-supported task execution and videos; no model tuning'})

if __name__=='__main__':
 lock=GPU_LOCK.open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 try:main()
 finally:fcntl.flock(lock,fcntl.LOCK_UN)
