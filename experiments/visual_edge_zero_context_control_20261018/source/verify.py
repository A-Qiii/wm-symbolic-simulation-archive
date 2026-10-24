from common import *
import torch,collections,shutil

def main():
 for mf in[BASE/'configs/delivery_manifest.json',E/'configs/source_references.json',E/'configs/training_frozen.json',E/'configs/final_heads_frozen.json']:
  for p,h in read(mf)['sha256'].items():assert sha(p)==h,p
 attempts=read(E/'results/training_attempts.json');assert len(attempts)==4 and all(r['status']=='completed'for r in attempts)
 runs=read(E/'results/training.json')['runs'];assert len(runs)==4 and all(r['updates']==600 and r['parameters']==6689 for r in runs)
 for head in HEADS:
  tr=np.load(E/f'data/train/{head}.npz');base=np.load(BASE/f'data/train/{head}.npz');assert np.array_equal(tr['x'][:,:,:74],base['edge'][:,:,:74]);assert not np.count_nonzero(tr['x'][:,:,74:])
  for k in['y','v','indices','state_ids','root_ids']:assert np.array_equal(tr[k],base[k])
  assert sha(E/f'configs/norm_{head}.npz')==sha(BASE/f'configs/norm_{head}_edge.npz')
  for ep in(E/'data/evaluation').glob('*.npz'):assert not(set(tr['root_ids'])&set(np.load(ep)['root_ids']))
  for seed in(0,1):
   assert sha(E/f'configs/schedule_{head}_seed{seed}.npz')==sha(BASE/f'configs/schedule_{head}_seed{seed}.npz')
   a=torch.load(E/f'models/initial_{head}_seed{seed}.pt',weights_only=True,map_location='cpu');b=torch.load(E/f'models/reference_full_{head}_seed{seed}.pt',weights_only=True,map_location='cpu');c=torch.load(E/f'models/{head}_zero_context_seed{seed}_update0600.pt',weights_only=True,map_location='cpu')
   assert a['initial_hash']==b['initial_hash']==c['initial_hash']==tensor_hash(a['state']);assert c['final_hash']==tensor_hash(c['state']);q=make();q.load_state_dict(c['state']);assert sum(p.numel()for p in q.parameters())==6689
 cells=0;groups=0
 for r in read(E/'results/actual_metrics.json'):
  d=np.load(E/f"data/evaluation/{r['cohort']}.npz");s=np.load(E/f"results/scores/{r['cohort']}_{r['arm']}_seed{r['seed']}.npz");assert np.array_equal(s['state_ids'],d['state_ids'])
  off=0 if r['head']=='Holding'else 3;cnt=[collections.Counter({'TP':0,'FN':0,'FP':0,'TN':0})for _ in COLORS];valid=correct=0
  for i in range(len(d['y'])):
   match=[]
   for j in range(3):
    y=bool(d['y'][i,off+j]);p=bool(s['scores'][i,off+j]>=.5);match.append(p==y)
    if d['v'][i,off+j]:cnt[j]['TP'if y and p else'FN'if y else'FP'if p else'TN']+=1
   if all(d['v'][i,off:off+3]):valid+=1;correct+=int(all(match))
  assert(valid,correct)==(r['all3_valid'],r['all3_correct']);groups+=1
  for j in range(3):
   for k in ['TP','FN','FP','TN']:assert cnt[j][k]==r['colors'][j][k];cells+=1
 for r in read(E/'results/combined6_metrics.json'):
  d=np.load(E/f"data/evaluation/{r['cohort']}.npz");s=np.load(E/f"results/scores/{r['cohort']}_{r['arm']}_seed{r['seed']}.npz")['scores'];valid=correct=0
  for i in range(len(s)):
   if all(d['v'][i]):valid+=1;correct+=int(all(bool(a>=.5)==bool(b)for a,b in zip(s[i],d['y'][i])))
  assert(valid,correct)==(r['valid6'],r['correct6'])
 rep=read(E/'results/reference_reproduction.json');assert len(rep)==40 and all(r['bit_identical_saved_reference_scores']for r in rep)
 strata=read(E/'results/matched_error_decomposition.json');by=collections.defaultdict(list)
 for r in strata:by[r['cohort'],r['head'],r['seed'],r['arm'],r['dimension']].append(r)
 metrics={(r['cohort'],r['head'],r['seed'],r['arm']):r for r in read(E/'results/actual_metrics.json')}
 for key,rs in by.items():
  for k in['TP','FN','FP','TN']:assert sum(r[k]for r in rs)==sum(c[k]for c in metrics[key[:4]]['colors'])
 save(E/'results/verification.json',{'utc':now(),'passed':True,'independently_recounted_confusion_cells':cells,'all3_groups':groups,'all6_groups':40,'reference_reproduction_groups':len(rep),'stratification_partition_checks':len(by)*4,'new_fits':4,'updates_each':600,'parameters_each':6689,'same_initialization_norm_schedule_labels_masks':True,'object_channels_identical_extra_channels_exact_zero':True,'train_evaluation_roots_disjoint':True,'base_delivery_manifest_unchanged':True})
 shutil.copy2(__file__,E/'source/verify.py');print('MATCHED_CONTROL_VERIFIED',cells,groups,len(by)*4,flush=True)
if __name__=='__main__':main()
