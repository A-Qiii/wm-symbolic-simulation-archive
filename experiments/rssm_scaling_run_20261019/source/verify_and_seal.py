from common import *
import subprocess,sys,csv,itertools,torch

def verify_manifest():
 m=read(B/'configs/delivery_manifest.json')
 for p,h in m['sha256'].items():assert sha(p)==h,p
 print('SEALED_FILES_VERIFIED',len(m['sha256']),sha(B/'configs/delivery_manifest.json'),flush=True)

def main():
 guard('verify')
 if '--verify' in sys.argv:verify_manifest();return
 assert read(B/'results/compute_complete.json')['main_fits']==8
 design=read(B/'configs/fixed_design.json');mm={k:set(v)for k,v in design['members'].items()}
 assert sha(B/'configs/fixed_design.json')==sha(DESIGN/'configs/proposed_protocol.json')
 assert mm['train20']<mm['train40']
 for a,b in itertools.combinations(['train40','validation10','test10','C20'],2):assert not mm[a]&mm[b]
 for receipt in ['all_models_frozen.json','all_probes_frozen.json']:
  for p,h in read(B/'configs'/receipt)['sha256'].items():assert sha(p)==h,p
 fitrecords=[];pred_checks=[];init={};probeinit={};manual=[]
 for size in [20,40]:
  cell=B/f'cells/roots{size}'
  for p,h in read(cell/'configs/frozen.json')['sha256'].items():assert sha(p)==h,p
  assert read(cell/'configs/normalization.json')==read(B/'configs/common20_normalization.json')
  assert set(read(cell/'configs/normalization.json')['fit_root_ids'])==mm['train20']
  for seed in [0,1]:
   schedule=np.load(cell/f'configs/schedule_seed{seed}.npz');assert schedule['indices'].shape==(6000,2,8)and schedule['indices'].max()<size*3
   original=torch.load(cell/f'models/initial_seed{seed}.pt',map_location='cpu',weights_only=True);assert original['fresh_random_initialization'];init.setdefault(seed,set()).add(original['hash'])
   for arm in ARMS:
    q=read(cell/f'results/train_{arm}_seed{seed}.json');assert q['updates']==6000 and q['initial_hash']==original['hash'] and q['schedule_sha256']==sha(cell/f'configs/schedule_seed{seed}.npz');fitrecords.append(q)
    log=[json.loads(t)for t in(cell/f'logs/train_{arm}_seed{seed}.jsonl').read_text().splitlines()];assert log[-1]['update']==6000 and all(np.isfinite(r['total'])for r in log)
    assert all(r['k']==(0 if r['update']<=5000 else 3)for r in log)
    probe=read(cell/f'results/probe_train_{arm}_seed{seed}.json');assert probe['updates']==600 and probe['train_roots']==20 and probe['train_records']==300;probeinit.setdefault(seed,set()).add(probe['initial_hash'])
    for split,group in [('train','train'+str(size)),('common20','train20'),('validation','validation10'),('test','test10'),('externalC','C20')]:
     for prefix in ['predictions','probe_predictions']:
      path=cell/f'data/{prefix}_{arm}_seed{seed}_{split}.npz';a=dict(np.load(path));assert set(a['root_id'])==mm[group];assert a['cat_valid'].all()and a['binary_valid'].all();assert a['prob'].shape==(len(a['root_id']),32)and np.isfinite(a['prob']).all()
      modes=set(a['mode']);assert 'free2'not in modes
      for mode,perroot in [('probe',15),('current',15),('one_step',12),('free3',3),('free4',3)]:
       mask=a['mode']==mode
       if mask.any():assert int(mask.sum())==len(mm[group])*perroot
      pred_checks.append({'size':size,'seed':seed,'arm':arm,'split':split,'kind':prefix,'roots':len(mm[group]),'rows':len(a['root_id']),'sha256':sha(path)})
      for mode in ['probe','free3']:
       mask=a['mode']==mode
       if not mask.any():continue
       pp=a['prob'][mask];yy=a['cat'][mask];rr=a['root_id'][mask]
       losses=np.stack([-np.log(np.clip(pp[np.arange(len(pp)),off+yy[:,j]],1e-9,1))for j,off in enumerate([0,8,16,24])],1).mean(1)
       rv={r:float(losses[rr==r].mean())for r in sorted(set(rr))}
       manual.append({'size':size,'seed':seed,'arm':arm,'split':split,'mode':mode,'value':float(np.mean(list(rv.values())))})
 assert all(len(v)==1 for v in init.values())and all(len(v)==1 for v in probeinit.values())
 for seed in [0,1]:
  assert np.array_equal(np.load(B/f'cells/roots20/data/probe_schedule_seed{seed}.npy'),np.load(B/f'cells/roots40/data/probe_schedule_seed{seed}.npy'))
 scores=read(B/'results/scores.json')
 for r in manual:
  q=next(q for q in scores['primary']if all(q[k]==v for k,v in r.items()if k!='value')and q['fact']=='primary4'and q['metric']=='NLL');assert abs(q['value']-r['value'])<1e-12
 freeze_time=read(B/'configs/all_probes_frozen.json')['utc'];heldouts=[]
 for size in [20,40]:
  for seed in [0,1]:
   for arm in ARMS:
    for split in ['validation','test','externalC']:
     q=read(B/f'cells/roots{size}/results/collapse_{arm}_seed{seed}_{split}.json');assert q['utc']>freeze_time;heldouts.append(q['utc'])
 old=[]
 for path,args in [(P/'experiments/rssm_encoder_gradient_20261019/source/verify_sealed.py',[]),(DESIGN/'source/verify_delivery.py',['--verify-seal'])]:
  q=subprocess.run([sys.executable,str(path),*args],capture_output=True,text=True,check=True);old.append({'script':str(path),'stdout':q.stdout.strip(),'stderr':q.stderr.strip()})
 save(B/'results/prediction_inventory.json',{'utc':now(),'files':pred_checks})
 save(B/'results/final_verification.json',{'utc':now(),'passed':True,'main_fits':8,'main_updates':sum(r['updates']for r in fitrecords),'probe_fits':8,'probe_updates':4800,'protocol_and_frozen_training_hashes_unchanged':True,'memberships_unchanged':True,'common20_normalization_andprobe_pool_only':True,'paired_initialization_andprobe_schedule':True,'final_only':True,'prediction_files':len(pred_checks),'primary_NLL_recomputations':len(manual),'test_C_prediction_after_all_models_and_probes_frozen':True,'first_heldout_prediction_completion':min(heldouts),'all_probes_frozen':freeze_time,'old_seal_verifications':old,'new_collection':0,'D_samples':0,'full_path_correctness_claimed':False,'visual_review':'See results/visual_review.json; generated PDF is inspected separately before final seal.'})
 files=[]
 for root in [B,REPORT]:
  for p in root.rglob('*'):
   if p.is_file()and p!=B/'configs/delivery_manifest.json'and '__pycache__'not in p.parts and 'matplotlib'not in p.parts:files.append(p)
 save(B/'configs/delivery_manifest.json',{'utc':now(),'status':'COMPLETED_FIXED_PROTOCOL','files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':{str(p):sha(p)for p in sorted(files)},'main_fits':8,'probe_fits':8})
 verify_manifest()

if __name__=='__main__':main()
