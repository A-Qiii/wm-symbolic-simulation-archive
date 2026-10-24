from common import *
from loss import pair_masks
import shutil,collections,csv

def main():
 assert not E.exists()
 for d in ['source','configs','data/train','data/evaluation','data/metadata','models','results','logs']:(E/d).mkdir(parents=True)
 R.mkdir(parents=True);(R/'figures').mkdir();refs={}
 def reg(p):refs[str(p)]=sha(p);return p
 seal=read(reg(OLD/'configs/delivery_manifest.json'))
 for p,h in seal['sha256'].items():assert sha(p)==h,p
 cfg=read(reg(OLD/'configs/protocol.json'))
 protocol={'utc':now(),'scope':'Single fixed-representation readout training-method exploration; no formal adoption',
 'heads':HEADS,'arms':ARMS,'seeds':[0,1],'max_runs':8,'updates_each':600,'batch_states':32,'architecture':'same shared64->32->SiLU->1,2113parameters',
 'input':'same frozen own-object64; no extra actions,colors,coordinates or IDs fed to network',
 'data':'exact previous augmented arm, including unchanged24old+8official schedule; no new sampling',
 'positive_weights':cfg['positive_weights'],'optimizer':'unchanged AdamW lr.001 wd.001 clip1, fresh optimizer',
 'BCE':'identical previous masked weighted BCE, mean valid objects/state then batchmean',
 'contrastive':{'temperature':.1,'coefficient':.1,'hidden':'L2-normalized post-SiLU32, no extra parameters','form':'Eq2 positive log terms averaged outside log; denominator all allowed known positive and negative facts within current predicate',
 'identity':'SHA256 canonical rgb_static,rgb_gripper,robot_obs arrays + object color; same identity excluded from both numerator and denominator, including every duplicate draw',
 'anchor':'known fact with >=1 distinct-identity same-label positive and >=1 opposite-label negative',
 'reduction':'0.5 mean eligible label0 anchors +0.5 mean eligible label1 anchors; if either class has no eligible anchor, whole regularizer=0 and BCE proceeds unchanged',
 'duplicates':'repeated draws cannot form self-positive pairs; their other comparisons keep fixed schedule multiplicity, reported explicitly',
 'unknown':'excluded as anchors and candidates; no relabeling','no_positive':'skip anchor, never resample or change batch'},
 'applicability_before_outcomes':{'both_classes_active_batches_fraction_min':.9,'eligible_known_anchors_each_class_fraction_min':.75,'positive_pairs_each_class_cross_episode_fraction_min':.5,'eligible_anchors_each_class_with_same_color_negative_fraction_min':.5,'meaning':'exploratory feasibility limits, not paper acceptance criteria; fail stops fitting, no batch repair'},
 'baseline_reproduction':'Run four BCE baselines first; require exact previous augmented parameter tensor hashes before any SupCon fitting. If mismatch, stop and explain, no extra baseline retries.',
 'selection':'fixed final600 and threshold.5; no validation/D tuning, no predicted-z secondary this round',
 'evaluation_cohorts':cfg['evaluation_cohorts'],'frozen_query_heads':cfg['frozen_query_heads'],
 'no_updates':['DINO','representation','all T','drawer Query'],'no_new_collection_or_download':True,
 'literature':{'paper':'https://arxiv.org/pdf/2004.11362','implementation':'https://github.com/HobbitLong/SupContrast','loss_source_read':'https://raw.githubusercontent.com/HobbitLong/SupContrast/master/losses.py','access_date':'2026-10-18','basis':'paper section3.2.2 Eq2; author code equality mask, self exclusion, dot-product/temperature and mean log positives inspected only','differences':'single-view frozen tokens, joint BCE regularizer on existing hidden32, class-balanced valid anchors and duplicate/unknown masks; tau=.1 and lambda=.1 are fixed exploratory choices, not paper reproduction; no temperature/base-temperature extra factor'}}
 save(E/'configs/protocol.json',protocol)
 for path in (OLD/'data/train').glob('*.npz'):shutil.copy2(reg(path),E/'data/train'/path.name)
 for path in (OLD/'data/evaluation').glob('*.npz'):shutil.copy2(reg(path),E/'data/evaluation'/path.name)
 for head in HEADS:
  shutil.copy2(reg(OLD/f'configs/normalization_{head}.npz'),E/f'configs/normalization_{head}.npz')
  for seed in(0,1):
   for src,dest in[(OLD/f'configs/schedule_{head}_seed{seed}.npz',E/f'configs/schedule_{head}_seed{seed}.npz'),(OLD/f'models/initial_{head}_seed{seed}.pt',E/f'models/initial_{head}_seed{seed}.pt'),(OLD/f'models/{head}_augmented_seed{seed}_update0600.pt',E/f'models/reference_{head}_seed{seed}.pt')]:shutil.copy2(reg(src),dest)
 shutil.copy2(reg(OLD/'configs/query_normalization.npz'),E/'configs/query_normalization.npz')
 shutil.copy2(reg(OLD/'data/private/labels.json'),E/'data/metadata/official_labels.json')
 shutil.copy2(reg(OLD/'source/train.py'),E/'source/previous_train_reference.py.txt')
 om=read(reg(REP/'data/public/states.json'));nm=read(reg(PRIOR/'data/ABC/public/states.json'));cm={str(r['state_index']):r for r in read(reg(EX/'simultaneous_relation_probe_20261017/data/train/states.json'))};pm={r['state_id']:r for r in read(reg(OLD/'data/public/states.json'))}
 def public(sid):
  if sid.startswith('official:'):return pm[sid]['path']
  if sid.startswith('oldpool:'):sid=sid[8:]
  prefix,index=sid.split(':');return cm[index]['public_path']if prefix=='combo'else(om if prefix=='old'else nm)[int(index)]['path']
 obsids={};identities={};audit=[];details=[]
 official=np.load(E/'data/train/official_train.npz')
 for head in HEADS:
  old=np.load(E/f'data/train/old_{head}.npz');sids=np.concatenate([old['state_ids'],official['state_ids']]);roots=np.concatenate([old['root_ids'],official['root_ids']]);g=slice(0,3)if head=='Holding'else slice(3,6);yy=np.concatenate([old['y'][:,g],official['y'][:,g]]);vv=np.concatenate([old['v'][:,g],official['v'][:,g]])
  hashes=[]
  for sid in sids:
   sid=str(sid)
   if sid not in obsids:
    path=reg(public(sid));d=np.load(path);hasher=hashlib.sha256()
    for k in ['rgb_static','rgb_gripper','robot_obs']:
     a=d[k];hasher.update(k.encode()+str(a.shape).encode()+str(a.dtype).encode()+a.tobytes())
    obsids[sid]=hasher.hexdigest()
   hashes.append(obsids[sid])
  unique=sorted(set(hashes));idmap={x:i for i,x in enumerate(unique)};ids=np.array([[idmap[x]*3+j for j in range(3)]for x in hashes]);rootmap={x:i for i,x in enumerate(sorted(set(roots)))};rootids=np.array([rootmap[x]for x in roots]);colors=np.tile(np.arange(3),32)
  identitylabels={}
  for ii,y,v in zip(ids.reshape(-1),yy.reshape(-1),vv.reshape(-1)):
   if v:
    if int(ii)in identitylabels:assert identitylabels[int(ii)]==bool(y),'inconsistent label for exact repeated observation/object'
    identitylabels[int(ii)]=bool(y)
  np.savez_compressed(E/f'data/train/identities_{head}.npz',ids=ids,root_ids=rootids,state_ids=sids,observation_hashes=np.array(hashes))
  for seed in(0,1):
   sched=np.load(E/f'configs/schedule_{head}_seed{seed}.npz');sc=np.concatenate([sched['common_old'],sched['official']+len(old['z'])],1);assert sc.shape==(600,32)
   totals=[collections.Counter(),collections.Counter()];active=0;mins=[999,999];maxs=[0,0];unique_anchor_total=0
   for u,ix in enumerate(sc):
    y=yy[ix].reshape(-1).astype(bool);v=vv[ix].reshape(-1).astype(bool);identifier=ids[ix].reshape(-1);root=np.repeat(rootids[ix],3)
    a,p,w,eligible=pair_masks(y,v,identifier);n=a&~p;active+=bool(w.sum()>0);unique_anchor_total+=len(set(identifier[v].tolist()));samecolor=colors[:,None]==colors[None,:];crossroot=root[:,None]!=root[None,:];crosscolor=~samecolor
    row={'head':head,'seed':seed,'update':u+1,'known':int(v.sum()),'unique_known_identities':len(set(identifier[v].tolist())),'active':bool(w.sum()>0)}
    for c in(0,1):
     q=v&(y==c);anchor=eligible&q;effective=p&anchor[:,None];den=int(effective.sum());totals[c].update({'known_anchors':int(q.sum()),'eligible_anchors':int(anchor.sum()),'positive_pairs':den,'positive_cross_color':int((effective&crosscolor).sum()),'positive_cross_episode':int((effective&crossroot).sum()),'negative_pairs':int((n&anchor[:,None]).sum()),'same_color_negative_pairs':int((n&anchor[:,None]&samecolor).sum()),'anchors_with_same_color_negative':int((anchor&(n&samecolor).any(1)).sum()),'anchors_with_cross_episode_positive':int((anchor&(p&crossroot).any(1)).sum()),'excluded_same_identity_pairs':int((q[:,None]&v[None,:]&(identifier[:,None]==identifier[None,:])).sum())})
     ec=int(anchor.sum());mins[c]=min(mins[c],ec);maxs[c]=max(maxs[c],ec);row[f'class{c}_eligible']=ec
    details.append(row)
   classes=[]
   for c,t in enumerate(totals):
    r={'label':c,**dict(t),'eligible_fraction':t['eligible_anchors']/t['known_anchors'],'positive_cross_color_fraction':t['positive_cross_color']/t['positive_pairs'],'positive_cross_episode_fraction':t['positive_cross_episode']/t['positive_pairs'],'eligible_same_color_negative_fraction':t['anchors_with_same_color_negative']/t['eligible_anchors'],'eligible_cross_episode_positive_fraction':t['anchors_with_cross_episode_positive']/t['eligible_anchors'],'eligible_per_batch_min':mins[c],'eligible_per_batch_max':maxs[c]};classes.append(r)
   gate=protocol['applicability_before_outcomes'];ok=active/600>=gate['both_classes_active_batches_fraction_min']and all(r['eligible_fraction']>=gate['eligible_known_anchors_each_class_fraction_min']and r['positive_cross_episode_fraction']>=gate['positive_pairs_each_class_cross_episode_fraction_min']and r['eligible_same_color_negative_fraction']>=gate['eligible_anchors_each_class_with_same_color_negative_fraction_min']for r in classes)
   audit.append({'head':head,'seed':seed,'active_batches':active,'batches':600,'unique_known_identity_draws':unique_anchor_total,'classes':classes,'applicable':ok})
 save(E/'results/pair_audit.json',{'utc':now(),'before_any_fit':True,'applicable':all(r['applicable']for r in audit),'runs':audit,'pair_counts':'directed pairs across frozen scheduled batches, not independent observations','network_receives_identity_or_color':False})
 save(E/'results/pair_audit_batches.json',details)
 save(E/'configs/source_references.json',{'sha256':refs})
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,E/'source'/p.name)
 with (R/'pair_audit.csv').open('w')as f:
  rows=[{'head':r['head'],'seed':r['seed'],'active_batches':r['active_batches'],**c}for r in audit for c in r['classes']];wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 print('PAIR_AUDIT_COMPLETE',json.dumps({'applicable':all(r['applicable']for r in audit),'runs':[{'head':r['head'],'seed':r['seed'],'active_batches':r['active_batches'],'classes':[{k:v for k,v in c.items()if 'fraction'in k or k=='label'}for c in r['classes']]}for r in audit]}),flush=True)
if __name__=='__main__':main()
