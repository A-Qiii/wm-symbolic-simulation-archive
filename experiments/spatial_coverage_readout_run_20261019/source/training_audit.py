from common import *
import csv,collections
sys.addaudithook(guard)
newroots={r['root_id']:r for r in read(S/'data/roots_manifest.json')if r['split']=='train'};oldroots={r['root_id']:r for r in read(P/'data/core1200_v1/roots_manifest.json')};oldchecks={r['root_id']:r for r in read(S/'results/old40_geometry.json')};mapping={r['root_id']:r['paired_old_root']for r in newroots.values()};oldm=read(P/'data/core1200_v1/data_manifest.json')['records'];newm=read(S/'data/data_manifest.json')['records'];assert sum(r['split']=='train'for r in newm)==480

def load(rows,ids):
 data={}
 for r in rows:
  if r['root_id']not in ids:continue
  d=data.setdefault(r['root_id'],{'boundaries':{},'labels':[],'results':[],'setup':[]});labels=read(r['labels_path']);res=read(Path(r['labels_path']).parent/'result_private.json');d['results'].append(res);b=int(r['episode_id'][-1]);d['boundaries'][b,r['step']+1]=labels['after']
  if r['step']==0:
   d['boundaries'][b,0]=labels['before'];d['setup']+=read(Path(r['labels_path']).parent.parent/'setup_private.json')['results']
 return data
old=load(oldm,set(mapping.values()));new=load(newm,set(mapping));rows=[];coverage=[];counts=[];groups={}
for group in['all40','old_pass29','old_fail11']:
 ids=[r for r,o in mapping.items()if group=='all40'or(not oldchecks[o]['reasons'])==(group=='old_pass29')];ca=np.zeros((4,8),int);cb=ca.copy();matched=0;h0match=0
 for rid in ids:
  oid=mapping[rid];a=old[oid];b=new[rid];assert set(a['boundaries'])==set(b['boundaries'])
  for key in a['boundaries']:
   ya=a['boundaries'][key]['categorical'][:4];yb=b['boundaries'][key]['categorical'][:4];matched+=ya==yb
   for j in range(4):ca[j,ya[j]]+=1;cb[j,yb[j]]+=1
  h0match+=a['boundaries'][0,0]['categorical'][:4]==b['boundaries'][0,0]['categorical'][:4]
  for arm,data,root in[('control',a,oldroots[oid]),('coverage',b,newroots[rid])]:
   res=data['results'];rows.append({'stratum':group,'paired_new_root':rid,'old_root':oid,'arm':arm,'transitions':12,'goal_success':sum(x['goal_success']for x in res),'status_success':sum(x['status']=='success'for x in res),'status_failed':sum(x['status']=='failed'for x in res),'status_refused':sum(x['status']=='refused'for x in res),'setup_calls':len(data['setup']),'setup_success':sum(x['status']=='success'for x in data['setup'])})
   label=data['boundaries'][0,0]
   for j,c in enumerate(COLORS):
    coverage.append({'stratum':group,'arm':arm,'root_id':rid,'old_root':oid,'scene':root['environment'],'initial_state':root['initial_state'],'color':c,'class':label['categorical'][j],'valid':label['categorical_valid'][j],'x_m':label['continuous']['objects'][c]['position_m'][0]})
 groups[group]={'roots':len(ids),'h0_exact_pairs':h0match,'all_boundary_exact_pairs':matched,'boundaries':len(ids)*15,'fields':[{'field':field,'old_counts':ca[j],'new_counts':cb[j],'total_variation':float(np.abs(ca[j]/ca[j].sum()-cb[j]/cb[j].sum()).sum()/2)}for j,field in enumerate(COLORS+['drawer'])]}
for name,data in [('paired_training_execution.csv',rows),('paired_h0_coverage.csv',coverage)]:
 with(S/'results'/name).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
save(S/'results/training_pair_audit.json',{'utc':now(),'strata':groups,'note':'29-pass and11-fail are pre-existing geometry strata of the paired training roots, not separately trained comparisons. Both frozen E and readout controls still use40 roots.','no_reweighting_or_deletion':True,'input_model_boundary':'DINO accepts only declared public NPZ; E.encode receives tokens, geometry, visible, robot; public_routes receives frames, anonymous cache IDs, valid; GT used only targets/audit','new_encoder_fits':0})
print('TRAINING_PAIR_AUDIT_COMPLETE',flush=True)
