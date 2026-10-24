from config import *
from train import features,make
import torch,csv,collections,itertools

def confusion(rr):
 tp=sum(r['truth']and r['prediction']for r in rr);fn=sum(r['truth']and not r['prediction']for r in rr);fp=sum(not r['truth']and r['prediction']for r in rr);tn=sum(not r['truth']and not r['prediction']for r in rr);p=tp+fn;n=fp+tn
 return {'N':len(rr),'TP':tp,'FN':fn,'FP':fp,'TN':tn,'positive_N':p,'negative_N':n,'recall':tp/p if p else None,'FPR':fp/n if n else None,'accuracy':(tp+tn)/len(rr)if rr else None,'balanced_accuracy':(tp/p+tn/n)/2 if p and n else None,'roots':len({r['root_id']for r in rr})}
def writecsv(path,rows):
 if rows:
  with path.open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 verify();assert not(E/'results/predictions.json').exists();frozen=read(E/'configs/probe_models_frozen.json')
 for p,h in frozen['sha256'].items():assert sha(p)==h,p
 torch.set_num_threads(4);trainrows=read(E/'data/train/objects.json');trainy=[r['In']for r in trainrows];majority=sum(trainy)>len(trainy)/2;colorprior={c:sum(r['In']for r in trainrows if r['object_color']==c)>sum(r['object_color']==c for r in trainrows)/2 for c in COLORS};out=[];known=set()
 for split,folder in [('train','train'),('validation','validation'),('diagnostic_D_seen','evaluator_D')]:
  d=dict(np.load(E/f'data/{folder}/states.npz'));rows=read(E/f'data/{folder}/objects.json');states=read(E/f'data/{folder}/states.json');assert not(known&{s['root_id']for s in states});known|={s['root_id']for s in states};truth=d['y'][:,3:6].reshape(-1).tolist();assert truth==[r['In']for r in rows];outputs={'majority_train':np.full(len(rows),float(majority)),'colour_prior_train':np.array([float(colorprior[r['object_color']])for r in rows]),'query_seed0':d['query_seed0'][:,3:6].reshape(-1),'query_seed1':d['query_seed1'][:,3:6].reshape(-1)}
  for feature,arch in itertools.product(['object','context'],['linear','mlp32']):
   name=f'{feature}_{arch}';cp=torch.load(E/'models'/f'{name}.pt',weights_only=True,map_location='cpu');model=make(cp['dim'],arch);model.load_state_dict(cp['state']);model.eval();norm=np.load(E/'models'/f'normalization_{feature}.npz');x=features(d['z'],feature)
   with torch.inference_mode():pp=torch.sigmoid(model(torch.tensor((x-norm['mean'])/norm['std']))).numpy().reshape(-1)
   assert np.isfinite(pp).all();outputs[name]=pp
  for name,pp in outputs.items():
   for row,y,p in zip(rows,truth,pp):out.append(row|{'model':name,'truth':bool(y),'probability':float(p),'prediction':bool(p>=.5)})
 metrics=[];joint=[]
 for split,name in itertools.product(['train','validation','diagnostic_D_seen'],sorted({r['model']for r in out})):
  rr=[r for r in out if r['split']==split and r['model']==name]
  groups=[('all','all',rr),('condition','held',[r for r in rr if r['holding_count']>0]),('condition','empty',[r for r in rr if r['holding_count']==0]),('condition','dual',[r for r in rr if r['dual_relation']]),('condition','held_without_inside',[r for r in rr if r['holding_count']>0 and not r['dual_relation']]),('object_status','held_object',[r for r in rr if r['object_is_held']]),('object_status','not_held',[r for r in rr if not r['object_is_held']])]
  groups += [('object_color',c,[r for r in rr if r['object_color']==c])for c in COLORS]+[('held_color',c,[r for r in rr if r['held_color']==c])for c in ['none',*COLORS]]+[('dual_object_color',c,[r for r in rr if r['dual_relation']and r['object_color']==c])for c in COLORS]+[('dual_held_color',c,[r for r in rr if r['dual_relation']and r['held_color']==c])for c in COLORS]+[('root',rid,[r for r in rr if r['root_id']==rid])for rid in sorted({r['root_id']for r in rr})]
  for group,value,ss in groups:metrics.append({'split':split,'model':name,'group':group,'value':value,**confusion(ss)})
  bystate=collections.defaultdict(list)
  for r in rr:bystate[r['state_index']].append(r)
  for cond in ['all','dual','held','empty']:
   values=[vv for vv in bystate.values()if cond=='all'or(cond=='dual'and vv[0]['dual_relation'])or(cond=='held'and vv[0]['holding_count']>0)or(cond=='empty'and vv[0]['holding_count']==0)];assert all(len(v)==3 for v in values);rootmeans=[np.mean([all(r['truth']==r['prediction']for r in v)for v in values if v[0]['root_id']==rid])for rid in sorted({v[0]['root_id']for v in values})];joint.append({'split':split,'model':name,'condition':cond,'state_N':len(values),'three_In_correct':sum(all(r['truth']==r['prediction']for r in v)for v in values),'root_N':len(rootmeans),'root_macro_three_In_accuracy':float(np.mean(rootmeans))if rootmeans else None})
 save(E/'results/predictions.json',out);save(E/'results/metrics.json',metrics);save(E/'results/joint_metrics.json',joint);writecsv(R/'per_object_predictions.csv',out);writecsv(R/'metrics.csv',metrics);writecsv(R/'joint_metrics.csv',joint)
 save(E/'results/evaluation_audit.json',{'utc':now(),'threshold':.5,'models_frozen_before_validation_and_D':True,'D_only_seen_diagnostic':True,'roots_disjoint':len(known)==18,'root_ids_not_model_inputs':True,'colour_and_held_identity_metadata_only':True,'new_experiments_or_model_selection':False,'majority_train_value':majority,'colour_prior_train':colorprior,'constant_positive_cheat_blocked_by_three_objects_and_negative_controls':True,'source_model_or_normalization_changes':0});print('PROBE_EVALUATED',len(out),'rows',flush=True)
if __name__=='__main__':main()
