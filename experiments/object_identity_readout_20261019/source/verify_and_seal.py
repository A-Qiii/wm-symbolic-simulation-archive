from common import *
import argparse,csv,collections,re
from PIL import Image

def csvread(name):return list(csv.DictReader((S/'results'/name).open()))
def inspect():
 frozen=read(S/'configs/all_heads_frozen.json');implementation=read(S/'configs/implementation_frozen.json')
 for p,h in frozen['sha256'].items():assert sha(p)==h,p
 for p,h in implementation['source_sha256'].items():assert sha(p)==h,p
 assert sha(S/'configs/protocol.json')==implementation['protocol_sha256']==read(S/'configs/preregistration_receipt.json')['protocol_sha256']
 assert sha(S/'configs/routing_amendment.json')==implementation['routing_amendment_sha256']
 protected=protected_seals();assert protected==read(S/'configs/protected_before.json')
 preflight=read(S/'results/preflight.json');assert preflight['routing_splits_before_fit']==['train','validation']
 assert preflight['test_C_samples_before_fits']==0 and preflight['utc']<frozen['utc']
 groups=csvread('counts.csv');rows=0;checks=0;rootsets={};decomp=[];heldcases=[]
 manifest_roots={r['root_id']:r for r in read(P/'data/core1200_v1/roots_manifest.json')}
 for seed in [0,1]:
  fits=[read(S/f'results/train_{arm}_seed{seed}.json')for arm in ARMS]
  for key in ['initial_hash','schedule_sha256','drawer_hash']:assert fits[0][key]==fits[1][key],key
  schedule=np.load(S/f'configs/schedule_seed{seed}.npy');assert schedule.shape==(1500,64)
  assert sha(S/f'configs/schedule_seed{seed}.npy')==fits[0]['schedule_sha256']
  for fit in fits:
   assert fit['updates']==1500 and fit['parameters']==309165 and fit['utc']<frozen['utc']
   path=S/f"models/head_{fit['arm']}_seed{seed}.pt";assert sha(path)==fit['weights_sha256']
   cp=torch.load(path,map_location='cpu',weights_only=True);assert th(cp['state'])==fit['final_hash']
  for split,n in [('train',600),('validation',150),('test',150),('externalC',300)]:
   ext=read(S/f'results/extraction_seed{seed}_{split}.json')
   assert ext['global_matches_original_exactly'] and ext['global_max_abs_difference']==0 and ext['encoder_weights_unchanged']
   route=read(S/f'results/routing_{split}.json');assert not route['wrong_assignments']
   if split in ['test','externalC']:assert route['utc']>frozen['utc'] and ext['utc']>frozen['utc']
   arrays={arm:dict(np.load(S/f'data/predictions_{arm}_seed{seed}_{split}.npz'))for arm in ARMS}
   for k in ['root_id','episode_id','step','cat','availability']:assert np.array_equal(arrays['global'][k],arrays['object'][k])
   assert np.array_equal(arrays['global']['prob'][:,24:],arrays['object']['prob'][:,24:])
   rootsets[split]=set(arrays['global']['root_id']);assert len(rootsets[split])==n//15
   for arm,a in arrays.items():
    p=a['prob'];y=a['cat'];assert p.shape==(n,27) and y.shape==(n,4) and np.isfinite(p).all()
    assert all(np.allclose(p[:,off:off+width].sum(1),1,atol=2e-6)for off,width in [(0,8),(8,8),(16,8),(24,3)])
    g=np.column_stack([p[:,j*8:j*8+8].argmax(1)for j in range(3)]+[p[:,24:].argmax(1)])
    H=p[:,[6,14,22]]>=.5;I=p[:,[3,11,19]]>=.5;hy=y[:,:3]==6;iy=y[:,:3]==3;rows+=n
    initial=np.array([manifest_roots[r]['initial_state']=='open_held'for r in a['root_id']])
    h0=(a['step']==0)&np.char.endswith(a['episode_id'],'_b0');assert h0.sum()==len(rootsets[split])
    for c in [r for r in groups if(r['seed'],r['split'],r['arm'])==(str(seed),split,arm)]:
     mask=h0.copy()if c['cohort']=='h0'else np.ones(n,bool)
     if c['stratum']=='initial_held':mask&=initial
     if c['stratum']=='initial_not_held':mask&=~initial
     values={'n':mask.sum(),'roots':len(set(a['root_id'][mask])),'exact4':((g==y).all(1)&mask).sum(),'nine':((H==hy).all(1)&(I==iy).all(1)&(g[:,3]==y[:,3])&mask).sum(),'holding_identity_correct_positive':((H==hy).all(1)&hy.any(1)&mask).sum()}
     for label,pred,truth in [('Holding',H,hy),('InDrawer',I,iy)]:
      for stat,b in [('TP',pred&truth),('FN',~pred&truth),('FP',pred&~truth),('TN',~pred&~truth)]:values[label+'_'+stat]=b[mask].sum()
     for k in range(8):values['loc_true'+str(k)]=(y[mask,:3]==k).sum();values['loc_correct'+str(k)]=((y[mask,:3]==k)&(g[mask,:3]==k)).sum()
     assert all(float(c[k])==float(v)for k,v in values.items()),(seed,split,arm,c['cohort'],c['stratum'],values)
     checks+=1
    if split=='externalC':
     err=g!=y;right=(y[:,:3]==2)&err[:,:3];table=(y[:,:3]<3)&err[:,:3]
     d={'seed':seed,'arm':arm,'records':n,'exact4':int((~err.any(1)).sum()),'failed_states':int(err.any(1).sum()),'states_with_table_error':int(table.any(1).sum()),'states_with_right_error':int(right.any(1).sum()),'object_field_errors':int(err[:,:3].sum()),'right_to_middle_errors':int(((y[:,:3]==2)&(g[:,:3]==1)).sum()),'only_drawer_error':int((~err[:,:3].any(1)&err[:,3]).sum())};decomp.append(d)
     for i in np.where(h0&initial)[0]:heldcases.append({'seed':seed,'arm':arm,'root_id':str(a['root_id'][i]),'truth':y[i].tolist(),'prediction_argmax':g[i].tolist(),'Holding_prob':p[i,[6,14,22]].tolist(),'Holding_threshold_prediction':H[i].tolist(),'held_identity_correct':bool((H[i]==hy[i]).all())})
 assert rows==4800 and checks==96
 assert all(not rootsets[a]&rootsets[b]for i,a in enumerate(SPLITS)for b in list(SPLITS)[i+1:])
 perm=read(S/'results/permutation.json')['tests'];assert len(perm)==16 and all(r['passed']and r['max_abs_logit_permutation_difference']==0 and r['scene_difference']==0 for r in perm)
 assert len(csvread('per_example.csv'))==4800
 text=(S/'REPORT.html').read_text();assert 'src="figures/'not in text and '<table>'in text and 'before_heldout_read'in text
 pictures=list((S/'figures').glob('*.png'));assert text.count('data:image/png;base64,')==len(pictures)
 for path in pictures:
  with Image.open(path)as im:im.verify()
 assert read(S/'results/failure.json')==read(S/'results/failure_attempt1.json')
 return {'utc':now(),'status':'VERIFIED','protected_original_files_unchanged':len(protected['sha256']),'frozen_head_files_unchanged':4,'training_updates_per_head':1500,'prediction_records':rows,'independently_recomputed_count_groups':checks,'disjoint_physical_roots':{k:len(v)for k,v in rootsets.items()},'global_features_exact_original':True,'paired_initialization_schedule_drawer_identical':True,'permutation_checks':16,'test_C_routing_extraction_prediction_after_all_heads_frozen':True,'validation_routing_preflight_before_training':True,'figures_checked_for_file_integrity':len(pictures),'html_figures_embedded':True,'failure_json_status':'Retained first preflight failure; superseded by preflight.json and compute_complete.json, not a current execution failure','C_error_decomposition':decomp,'C_h0_held_cases':heldcases,'scope':'Current observation readout only; does not validate RSSM free rollout or complete original guide'}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');args=parser.parse_args();sys.addaudithook(audit_writes)
 manifest=S/'manifest.json'
 if args.seal:assert not manifest.exists(),'Already sealed; use default read-only verification'
 elif manifest.exists():
  for name,h in read(manifest)['sha256'].items():assert sha(S/name)==h,name
 else:raise RuntimeError('No manifest; use --seal once after review')
 result=inspect()
 if args.seal:
  save(S/'results/final_verification.json',result)
  paths=sorted(p for p in S.rglob('*')if p.is_file()and 'cache'not in p.relative_to(S).parts and p!=manifest)
  save(manifest,{'utc':now(),'scope':'object_identity_readout_20261019 only','files':len(paths),'total_bytes':sum(p.stat().st_size for p in paths),'sha256':{str(p.relative_to(S)):sha(p)for p in paths}})
  print('SEALED',len(paths),'files; manifest SHA256',sha(manifest))
 print(json.dumps({k:v for k,v in result.items()if k not in ['C_error_decomposition','C_h0_held_cases']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
