from common import *
import torch,csv
from collections import Counter
def main():
 verify();guard('verify');checks={};count=0
 for f in ['evaluation_frozen.json','models_frozen.json','probes_frozen.json']:
  for p,h in read(E/'configs'/f)['sha256'].items():assert sha(p)==h,p;count+=1
 checks['frozen_hashes']=len(read(E/'configs/frozen.json')['sha256'])+count
 assert read(E/'results/preflight.json')['passed']and read(E/'results/preflight_bf16.json')['passed']
 assert read(E/'results/score_tests.json')['passed'];checks['preflights_and_metric_tests']=True
 fits=[];features={};probes=[]
 for seed in [0,1]:
  for arm in ARMS:
   fit=read(E/f'results/train_{arm}_seed{seed}.json');assert fit['updates']==6000 and not fit['Cselection'];assert fit['C_samples_read']==fit['D_samples_read']==0;fits.append(fit)
   log=[json.loads(l)for l in(E/f'logs/train_{arm}_seed{seed}.jsonl').read_text().splitlines()];assert log[-1]['update']==6000 and log[-1]['k']==3;assert all(r['k']==(0 if r['update']<=5000 else 3)for r in log)
   cp=torch.load(E/f'models/{arm}_seed{seed}/final6000.pt',map_location='cpu',weights_only=True);assert cp['update']==6000 and th(cp['model'])==fit['final_hash'];assert sum(t.numel()for t in cp['model'].values())==11513563;del cp
   probe=read(E/f'results/probe_train_{arm}_seed{seed}.json');assert probe['updates']==600 and probe['C_samples_read']==0;probes.append(probe)
   for split,n_ep,n_root in [('train',180,60),('dev',60,20)]:
    a=dict(np.load(E/f'data/predictions_{arm}_seed{seed}_{split}.npz'));f=dict(np.load(E/f'data/features_{arm}_seed{seed}_{split}.npz'));pr=dict(np.load(E/f'data/probe_predictions_{arm}_seed{seed}_{split}.npz'))
    assert len(a['prob'])==n_ep*23 and len(f['x'])==len(pr['prob'])==n_ep*5
    assert len(set(a['root_id']))==n_root and not any(r.startswith('D')for r in a['root_id'])
    assert all(r.startswith(('A','B')if split=='train'else'C')for r in a['root_id'])
    assert len(set(a['episode_id']))==n_ep and set(a['root_id'])==set(f['root_id'])==set(pr['root_id'])
    assert np.isfinite(f['x']).all()and np.isfinite(a['prob']).all()and np.isfinite(pr['prob']).all()
    counts=Counter(a['mode']);assert counts['one_step']==n_ep*4 and counts['current']==n_ep*5 and counts['free3']==n_ep
    one=a['mode']=='one_step';assert int((a['controls'][one]==0).sum())==(120 if split=='train'else 41)
    three=a['mode']=='free3';assert int(a['all_prefix_actuated'][three].sum())==(60 if split=='train'else 19)
    for eid in set(a['episode_id']):
     rootprob=a['prob'][(a['episode_id']==eid)&(a['mode']=='current')&(a['step']==0)][0]
     persist=a['prob'][(a['episode_id']==eid)&np.char.startswith(a['mode'],'root_persistence')];assert np.array_equal(persist,np.broadcast_to(rootprob,persist.shape))
    assert all(np.array_equal(f[k],pr[k])for k in ['root_id','episode_id','step','cat','cat_valid','binary','binary_valid'])
    features[arm,seed,split]=f
    c=read(E/f'results/collapse_{arm}_seed{seed}_{split}.json');assert 1<=c['E_effective_rank']<=512.0001;assert c['posterior_entropy_gap_mean_nats']>=-1e-5;assert c['support_hard_negative_FPR']is None
 for seed in [0,1]:
  a,b=[next(x for x in fits if x['arm']==arm and x['seed']==seed)for arm in ARMS];assert a['initial_hash']==b['initial_hash']and a['schedule_sha256']==b['schedule_sha256'];assert a['final_hash']!=b['final_hash']
  a,b=[next(x for x in probes if x['arm']==arm and x['seed']==seed)for arm in ARMS];assert a['initial_hash']==b['initial_hash']and a['schedule_sha256']==b['schedule_sha256']
  for split in ['train','dev']:
   a,b=[features[arm,seed,split]for arm in ARMS]
   assert all(np.array_equal(a[k],b[k])for k in a if k!='x')
 checks['paired_initialization_batches_probe_schedule']=True;checks['four_main_fits_updates']=[r['updates']for r in fits];checks['four_probe_fits_updates']=[r['updates']for r in probes];checks['prediction_denominators_and_root_persistence']=True
 # Independently recompute primary statistics directly from saved probabilities.
 stats=read(E/'results/scores.json');rv={};rngseed=261019900
 for seed in [0,1]:
  for arm in ARMS:
   for split in ['train','dev']:
    for mode in ['current','one_step','free3','probe']:
     pre='probe_predictions'if mode=='probe'else'predictions';a=dict(np.load(E/f'data/{pre}_{arm}_seed{seed}_{split}.npz'));mask=a['mode']==mode;p=a['prob'][mask];y=a['cat'][mask];roots=a['root_id'][mask]
     values=np.stack([-np.log(np.maximum(p[np.arange(len(p)),offset+y[:,j]],1e-9))for j,offset in enumerate([0,8,16,24])],1).mean(1)
     rootmeans={r:float(values[roots==r].mean())for r in sorted(set(roots))};rv[arm,seed,split,mode]=rootmeans
     r=next(x for x in stats['primary']if all(x[k]==v for k,v in {'arm':arm,'seed':seed,'split':split,'mode':mode}.items()))
     assert abs(np.mean(list(rootmeans.values()))-r['value'])<1e-7
 for seed in [0,1]:
  for split in ['train','dev']:
   for mode in ['current','one_step','free3','probe']:
    a,b=[rv[arm,seed,split,mode]for arm in ARMS];d=np.array([a[r]-b[r]for r in sorted(a)]);rng=np.random.default_rng(rngseed);boot=d[rng.integers(len(d),size=(2000,len(d)))].mean(1);lo,hi=np.quantile(boot,[.025,.975]);r=next(x for x in stats['paired_primary']if x['seed']==seed and x['split']==split and x['mode']==mode)
    assert max(abs(d.mean()-r['joint_minus_control']),abs(lo-r['lo95']),abs(hi-r['hi95']))<1e-7
 checks['independent_primary_and_paired_bootstrap_recompute']=True
 assert read(E/'results/evaluation.json')['model_weights_unchanged'];checks['no_model_update_during_evaluation']=True
 assert not(P/'experiments/uninterrupted_action_prediction162_20261019').exists();checks['unused162_not_executed']=True
 save(E/'results/verification.json',{'utc':now(),'passed':True,'checks':checks,'scope':'source/hash/model metadata and saved prediction checks; no new fitting or D sample access','D_samples_read':0});print('VERIFICATION_PASSED',json.dumps(checks),flush=True)
if __name__=='__main__':main()
