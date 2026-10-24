from train import *
def concat(ds,preds,names):
 return np.concatenate([preds[n]for n in names]),{'y':np.concatenate([ds[n]['y']for n in names]),'v':np.concatenate([ds[n]['v']for n in names]),'meta':[r for n in names for r in ds[n]['meta']]}
def extra(p,d,tau):
 m=evaluate_predictions(p,d,tau)
 for name,cols in GROUPS.items():
  pp=p[:,cols][d['v'][:,cols]];yy=d['y'][:,cols][d['v'][:,cols]];m[name]['BCE']=float(-(yy*np.log(np.clip(pp,1e-7,1-1e-7))+(1-yy)*np.log(np.clip(1-pp,1e-7,1-1e-7))).mean())
 return m
def main():
 assert not(E/'results/evaluation.json').exists();T=read(E/'results/training.json');out={};cases={};freeze=[]
 for arm in ARMS:
  ds,_=build(arm);out[arm]=[]
  for run in T[arm]:
   seed=run['seed'];row={'seed':seed,'modes':{}}
   for mode,r in run['modes'].items():
    preds={n:np.load(E/'results'/f'{arm}_seed{seed}_{mode}_{n}.npy')for n in NAMES};tau=r['calibration']['thresholds'];fp,fresh=concat(ds,preds,['fresh_tune','fresh_cal']);hp,hist=concat(ds,preds,['old_tune','old_cal','new_dev']);conditions={}
    for cond in ['hard_negative','ordinary_negative','held_positive']:
     ix=np.array([q['condition']==cond for q in fresh['meta']]);conditions[cond]=extra(fp[ix],{'y':fresh['y'][ix],'v':fresh['v'][ix]},tau)
    row['modes'][mode]={'update':r['update'],'qualified':r['qualified'],'thresholds':tau,'fresh':extra(fp,fresh,tau),'historical':extra(hp,hist,tau),'conditions':conditions,'domains':{n:extra(p,ds[n],tau)for n,p in preds.items()}}
    freeze.append({'arm':arm,'seed':seed,'mode':mode,'update':r['update'],'checkpoint':r['checkpoint'],'checkpoint_sha256':r['checkpoint_sha256'],'thresholds':tau})
    for i,m in enumerate(fresh['meta']):
     q=cases.setdefault(m['sample_id'],{'metadata':m,'truth':fresh['y'][i].tolist(),'valid':fresh['v'][i].tolist(),'predictions':{},'raw_errors':[]});q['predictions'][f'{arm}_seed{seed}_{mode}']={'p':fp[i].tolist(),'known':known(fp[i:i+1],tau)[0].tolist()}
     for col in np.where(fresh['v'][i]&((fp[i]>=.5)!=fresh['y'][i]))[0]:q['raw_errors'].append({'arm':arm,'seed':seed,'mode':mode,'column':int(col),'p':float(fp[i,col]),'truth':bool(fresh['y'][i,col])})
   out[arm].append(row)
 paired=[]
 for seed in range(3):
  d={a:out[a][seed]['modes']['fixed600']for a in ARMS};paired.append({'seed':seed,'current':{g:d['current_only']['fresh']['held'][g]for g in ['FP','FN','BCE','Brier','known_true_positive','positive_UNKNOWN']},'short':{g:d['short_history']['fresh']['held'][g]for g in ['FP','FN','BCE','Brier','known_true_positive','positive_UNKNOWN']},'long':{g:d['long_history']['fresh']['held'][g]for g in ['FP','FN','BCE','Brier','known_true_positive','positive_UNKNOWN']},'long_minus_short_held_BCE':d['long_history']['fresh']['held']['BCE']-d['short_history']['fresh']['held']['BCE'],'long_minus_short_held_raw_errors':sum(d['long_history']['fresh']['held'][g]-d['short_history']['fresh']['held'][g]for g in ['FP','FN']),'historical_raw_error_counts':{a:{g:d[a]['historical'][g]['FP']+d[a]['historical'][g]['FN']for g in GROUPS}for a in ARMS}})
 out['paired_fixed600']=paired;out['scope']='Same sixCdevelopmentroots; not independent test, no equivalence claim.';save(E/'results/evaluation.json',out);save(E/'results/fresh_cases.json',list(cases.values()));save(E/'configs/candidates_frozen.json',{'created_utc':now(),'models':freeze,'no_test':True,'no_promotion':True});(E/'configs/candidates_frozen.json').chmod(0o444)
 print(json.dumps({'paired_fixed600':paired,'fresh_failure_window_count':sum(bool(q['raw_errors'])for q in cases.values()),'long_fixed600_fresh_in_open_errors':[[out['long_history'][s]['modes']['fixed600']['fresh'][g]['FP'],out['long_history'][s]['modes']['fixed600']['fresh'][g]['FN']]for s in range(3)for g in ['in_drawer','open_drawer']]},ensure_ascii=False))
if __name__=='__main__':main()
