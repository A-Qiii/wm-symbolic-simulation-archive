from common import *
import csv,shutil

def main():
 checked=0
 for name in ['source_references','pre_encoding_frozen','training_frozen','final_heads_frozen']:
  for p,h in read(E/f'configs/{name}.json')['sha256'].items():assert sha(p)==h,(name,p);checked+=1
 for p,h in read(E/'configs/encoding_frozen.json')['code_sha256'].items():assert sha(p)==h,p
 rows=read(E/'data/public/states.json');cohorts=read(E/'configs/cohorts.json');labels=read(E/'data/private/labels.json');cfg=read(E/'configs/protocol.json');vects=np.load(E/'data/public/vectors.npz')
 trainroots={rows[i]['root_id']for h in HEADS for i in cohorts['train_'+h]}
 for r in rows:
  assert sha(r['path'])==r['sha256'];z=np.load(r['path']);assert set(z.files)=={'rgb_static','rgb_gripper'}
 for n,ix in cohorts.items():
  if not n.startswith('train'):assert trainroots.isdisjoint(rows[i]['root_id']for i in ix)
 for head in HEADS:
  train=np.load(E/f'data/train/{head}.npz');assert np.array_equal(train['node'],train['edge'][...,:74]);assert np.array_equal(vects[head][...,:74],vects['node'])
  for seed in (0,1):
   old=np.load(OLD/f'configs/schedule_{head}_seed{seed}.npz');new=np.load(E/f'configs/schedule_{head}_seed{seed}.npz')
   for k in old.files:assert np.array_equal(old[k],new[k])
 attempts=read(E/'results/training_attempts.json');runs=read(E/'results/training.json')['runs'];assert len(attempts)==len(runs)==8
 assert all(r['status']=='completed'for r in attempts);assert all(r['updates']==600 and r['parameters']==6689 for r in runs)
 out=read(E/'results/actual_metrics.json');conf=[];changes=[];cells=0;all3checks=0;flips=[]
 for dpath in sorted((E/'data/evaluation').glob('*.npz')):
  d=np.load(dpath);name=dpath.stem;y=d['y'].astype(bool);v=d['v'].astype(bool)
  for seed in(0,1):
   saved={arm:np.load(E/f'results/scores/{name}_{arm}_seed{seed}.npz')['scores']for arm in ARMS}
   for head in HEADS:
    offset=0 if head=='Holding' else 3;g=slice(offset,offset+3);yy=y[:,g];vv=v[:,g]
    for arm in ARMS:
     r=next(r for r in out if (r['cohort'],r['head'],r['arm'],r['seed'])==(name,head,arm,seed));p=saved[arm][:,g]>=.5
     assert r['all3_valid']==int(vv.all(1).sum());assert r['all3_correct']==int(((p==yy).all(1)&vv.all(1)).sum());all3checks+=1
     for j,c in enumerate(COLORS):
      tp=int((vv[:,j]&yy[:,j]&p[:,j]).sum());fn=int((vv[:,j]&yy[:,j]&~p[:,j]).sum());fp=int((vv[:,j]&~yy[:,j]&p[:,j]).sum());tn=int((vv[:,j]&~yy[:,j]&~p[:,j]).sum())
      cc=r['colors'][j];assert [tp,fn,fp,tn]==[cc[k]for k in['TP','FN','FP','TN']];cells+=4;conf.append({'cohort':name,'head':head,'seed':seed,'arm':arm,**cc})
    a=next(r for r in out if(r['cohort'],r['head'],r['arm'],r['seed'])==(name,head,'node',seed));b=next(r for r in out if(r['cohort'],r['head'],r['arm'],r['seed'])==(name,head,'edge',seed))
    changes.append({'cohort':name,'head':head,'seed':seed,'node_correct':a['all3_correct'],'edge_correct':b['all3_correct'],'valid':a['all3_valid'],'change':b['all3_correct']-a['all3_correct']})
    for j,c in enumerate(COLORS):
     aa=saved['node'][:,offset+j]>=.5;bb=saved['edge'][:,offset+j]>=.5
     for i in np.where(vv[:,j]&(aa!=bb))[0]:flips.append({'cohort':name,'head':head,'color':c,'seed':seed,'state_id':str(d['state_ids'][i]),'root_id':str(d['root_ids'][i]),'index':int(d['indices'][i]),'truth':bool(yy[i,j]),'node_score':float(saved['node'][i,offset+j]),'edge_score':float(saved['edge'][i,offset+j]),'effect':'regression'if aa[i]==yy[i,j]else'improvement','edge_error':'FN'if yy[i,j]and not bb[i]else'FP'if not yy[i,j]and bb[i]else None})
 save(E/'results/changes.json',changes);save(E/'results/decision_flips.json',flips)
 for name,rr in [('actual_confusion',conf),('actual_changes',changes),('decision_flips',flips)]:
  with(R/f'{name}.csv').open('w')as f:wr=csv.DictWriter(f,fieldnames=list(rr[0]));wr.writeheader();wr.writerows(rr)
 assert read(E/'results/unit_tests.json')['passed'];assert not read(E/'results/training.json')['validation_access'];assert read(E/'results/encoding.json')['robot_channels']==0
 save(E/'results/verification.json',{'utc':now(),'passed':True,'frozen_hash_checks':checked,'public_RGB_only_states':len(rows),'root_isolation':True,'same_schedules':True,'same_object_features_in_both_arms':True,'fits':8,'updates_each':600,'parameters_each':6689,'confusion_cells_recounted':cells,'all3_metrics_recounted':all3checks,'untrained_backbone_and_T':True,'no_SupCon_training':not(EX/'readout_supcon_ablation_20261018/results/training_attempts.json').exists()})
 shutil.copy2(__file__,E/'source/verify.py');print('VISUAL_NODE_EDGE_VERIFIED',cells,'confusion cells',all3checks,'all3 metrics',flush=True)
if __name__=='__main__':main()
