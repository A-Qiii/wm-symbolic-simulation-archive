from coverage_common import *
import sys,torch,time,collections
sys.path.insert(0,str(REP/'source'))
from model import Query,probs
def metrics(p,y,v,roots):
 pred=p>=.5;yy=y.astype(bool);vv=v.astype(bool);allvalid=vv.all(1);empty=allvalid&~yy.any(1);out={'states':len(y),'roots':len(set(roots)),'all3_valid':int(allvalid.sum()),'all3_correct':int(((pred==yy).all(1)&allvalid).sum()),'all3_accuracy':float((pred[allvalid]==yy[allvalid]).all(1).mean())if allvalid.any()else None,'empty_states':int(empty.sum()),'empty_any_FP':int(pred[empty].any(1).sum()),'colors':[]}
 for j,c in enumerate(COLORS):
  pos=vv[:,j]&yy[:,j];neg=vv[:,j]&~yy[:,j];tp=int((pos&pred[:,j]).sum());fn=int((pos&~pred[:,j]).sum());fp=int((neg&pred[:,j]).sum());tn=int((neg&~pred[:,j]).sum());out['colors'].append({'color':c,'TP':tp,'FN':fn,'FP':fp,'TN':tn,'positive_episodes':len(set(roots[pos])),'negative_episodes':len(set(roots[neg])),'FNR':fn/(tp+fn)if tp+fn else None,'FPR':fp/(fp+tn)if fp+tn else None})
 rr=[(c['FPR']+c['FNR'])/2 for c in out['colors']if c['FPR']is not None and c['FNR']is not None];out['mean_color_balanced_error']=float(np.mean(rr))if rr else None
 out['root_exact']={r:{'correct':int(((pred==yy).all(1)&allvalid&(roots==r)).sum()),'valid':int((allvalid&(roots==r)).sum())}for r in sorted(set(roots))};return out
def main():
 assert not(E/'results/actual_metrics.json').exists();cfg=read(E/'configs/protocol.json');torch.set_num_threads(4);start=time.perf_counter()
 for p,h in read(E/'configs/final_heads_frozen.json')['sha256'].items():assert sha(p)==h,p
 normq=np.load(E/'configs/query_normalization.npz',allow_pickle=False);out=[];full=[];paired=[];draweraudit=[];(E/'results/actual_probabilities').mkdir()
 private=read(E/'data/private/labels.json');officialmeta={r['state_id']:r for r in private};frozenheads={}
 for seed in[0,1]:
  query=Query().eval().requires_grad_(False);query.load_state_dict(torch.load(cfg['frozen_query_heads'][str(seed)]['path'],weights_only=True,map_location='cpu')['state']);qh=tensor_hash(query.state_dict())
  heads={}
  for head in HEADS:
   for arm in ARMS:
    q=make().eval().requires_grad_(False);q.load_state_dict(torch.load(E/'models'/f'{head}_{arm}_seed{seed}_update0600.pt',weights_only=True,map_location='cpu')['state']);heads[head,arm]=q
  for path in sorted((E/'data/evaluation').glob('*.npz')):
   d=np.load(path,allow_pickle=False);cohort=path.stem;z=d['z'];yy=d['y'];vv=d['v'];roots=d['root_ids']
   with torch.inference_mode():original=probs(query(torch.tensor((z-normq['mean'])/normq['std']))).numpy()
   probabilities={}
   for head in HEADS:
    n=np.load(E/'configs'/f'normalization_{head}.npz',allow_pickle=False);x=torch.tensor(((z[:,:3]-n['mean'])/n['std']).astype(np.float32));g=slice(0,3)if head=='Holding'else slice(3,6)
    for arm in ARMS:
     with torch.inference_mode():p=heads[head,arm](x).squeeze(-1).sigmoid().numpy()
     probabilities[head,arm]=p;out.append({'head':head,'arm':arm,'seed':seed,'cohort':cohort,**metrics(p,yy[:,g],vv[:,g],roots)})
     if cohort=='official_validation':
      by=collections.defaultdict(list)
      for i,sid in enumerate(d['state_ids']):
       for hint in officialmeta[str(sid)]['candidate_hints']:by[hint['annotation_index']].append(i)
      for j,c in enumerate(COLORS):
       records=[]
       for ann,indices in sorted(by.items()):
        ix=np.array(sorted(set(indices)));known=vv[ix,g][:,j].astype(bool);pos=known&yy[ix,g][:,j].astype(bool);neg=known&~yy[ix,g][:,j].astype(bool)
        if pos.any()and neg.any():records.append({'annotation_index':ann,'root_id':str(roots[ix[0]]),'positive_states':int(pos.sum()),'negative_states':int(neg.sum()),'paired_correct':bool(((p[ix,j]>=.5)==yy[ix,g][:,j].astype(bool))[known].all())})
       paired.append({'head':head,'arm':arm,'seed':seed,'color':c,'eligible_segment_color_pairs':len(records),'paired_correct':sum(r['paired_correct']for r in records),'accuracy':sum(r['paired_correct']for r in records)/len(records)if records else None,'records':records,'not_executed_transition':True})
   for arm in ARMS:
    pp=np.concatenate([probabilities['Holding',arm],probabilities['In',arm],original[:,6:]],axis=1);assert np.array_equal(pp[:,6:],original[:,6:]);pred=pp>=.5;pred[:,6:]=False;pred[np.arange(len(pred)),6+pp[:,6:].argmax(1)]=True;valid=vv.astype(bool).all(1);correct=(pred==yy.astype(bool)).all(1)&valid
    full.append({'seed':seed,'arm':arm,'cohort':cohort,'states':len(z),'all9_valid':int(valid.sum()),'all9_correct':int(correct.sum()),'all9_accuracy':float(correct[valid].mean())if valid.any()else None,'drawer3_correct':int((((pred[:,6:]==yy[:,6:]).all(1))&vv[:,6:].all(1)).sum()),'drawer3_valid':int(vv[:,6:].all(1).sum())});np.savez_compressed(E/'results/actual_probabilities'/f'{cohort}_{arm}_seed{seed}.npz',probability=pp,state_ids=d['state_ids'],root_ids=roots)
   draweraudit.append({'seed':seed,'cohort':cohort,'both_arms_drawer_exactly_equal_original':True,'query_state_hash':qh})
  assert tensor_hash(query.state_dict())==qh
 save(E/'results/actual_metrics.json',out);save(E/'results/full9_actual_metrics.json',full);save(E/'results/official_matched_pairs.json',paired);save(E/'results/frozen_drawer_audit.json',draweraudit)
 def get(head,seed,arm,cohort):return next(r for r in out if r['head']==head and r['seed']==seed and r['arm']==arm and r['cohort']==cohort)
 gates=[]
 for head in HEADS:
  seeds=[]
  for seed in[0,1]:
   reasons=[];c=get(head,seed,'control','official_validation');a=get(head,seed,'augmented','official_validation');gain=c['mean_color_balanced_error']-a['mean_color_balanced_error']
   if gain<.01-1e-12:reasons.append('official balanced-error improvement below .01')
   if a['all3_accuracy']<c['all3_accuracy']-1e-12:reasons.append('official all3 exact accuracy decreased')
   for cohort in ['official_validation']+cfg['secondary_gate']['retention']:
    c=get(head,seed,'control',cohort);a=get(head,seed,'augmented',cohort)
    if cohort!='official_validation'and c['all3_accuracy']-a['all3_accuracy']>.02+1e-12:reasons.append(cohort+':all3 retention decline above .02')
    for cc,aa in zip(c['colors'],a['colors']):
     for rate in['FPR','FNR']:
      if cc[rate]is not None and aa[rate]-cc[rate]>.05+1e-12:reasons.append(cohort+':'+cc['color']+':'+rate+' increase above .05')
   seeds.append({'seed':seed,'official_balanced_error_gain':gain,'passes':not reasons,'reasons':reasons})
  gates.append({'head':head,'eligible_for_saved_prediction_diagnostic':all(s['passes']for s in seeds),'seeds':seeds})
 save(E/'results/secondary_gate.json',{'utc':now(),'head_decisions':gates,'D_used_for_gate':False,'no_model_or_checkpoint_selection':True,'no_extra_fit':True,'actual_evaluation_seconds':time.perf_counter()-start})
 primary=[{k:r[k]for k in['head','arm','seed','cohort','all3_correct','all3_valid','mean_color_balanced_error','colors']}for r in out if r['cohort']in['official_validation','C_validation','C_endpoints48','new_AB_validation','D12_seen_actual135']];save(E/'results/primary_numbers.json',primary)
 print('ACTUAL_READOUT_COMPLETE');print(json.dumps({'gates':gates,'official':[r for r in primary if r['cohort']=='official_validation']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
