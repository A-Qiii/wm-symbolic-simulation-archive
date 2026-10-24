from common import *
import torch,collections,csv,time

def metrics(p,y,v,roots):
 pred=p>=.5;yy=y.astype(bool);vv=v.astype(bool);known=vv.all(1);result={'states':len(y),'roots':len(set(roots)),'all3_valid':int(known.sum()),'all3_correct':int(((pred==yy).all(1)&known).sum()),'colors':[]}
 result['all3_accuracy']=result['all3_correct']/result['all3_valid']if result['all3_valid']else None
 for j,c in enumerate(COLORS):
  pos=vv[:,j]&yy[:,j];neg=vv[:,j]&~yy[:,j];tp=int((pos&pred[:,j]).sum());fn=int((pos&~pred[:,j]).sum());fp=int((neg&pred[:,j]).sum());tn=int((neg&~pred[:,j]).sum())
  result['colors'].append({'color':c,'TP':tp,'FN':fn,'FP':fp,'TN':tn,'FNR':fn/(tp+fn)if tp+fn else None,'FPR':fp/(tn+fp)if tn+fp else None,'positive_roots':len(set(roots[pos])),'negative_roots':len(set(roots[neg]))})
 be=[(c['FNR']+c['FPR'])/2 for c in result['colors']if c['FNR']is not None and c['FPR']is not None];result['mean_balanced_error']=float(np.mean(be))if be else None
 result['per_root']={str(r):{'valid':int((known&(roots==r)).sum()),'correct':int(((pred==yy).all(1)&known&(roots==r)).sum())}for r in sorted(set(roots))};return result

def main():
 assert not(E/'results/actual_metrics.json').exists()
 for p,h in read(E/'configs/final_heads_frozen.json')['sha256'].items():assert sha(p)==h,p
 torch.set_num_threads(4);start=time.perf_counter();segments={r['state_id']:r for r in read(E/'data/private/official_segments.json')};cohorts=read(E/'configs/cohorts.json');labels=read(E/'data/private/labels.json')
 trainunion=sorted(set(cohorts['train_Holding']+cohorts['train_In']));signatures={tuple(bool(x)for x in labels[i]['y'])for i in trainunion if all(labels[i]['v'])}
 out=[];combined=[];paired=[];swaps=[];visibility=[];support=[];(E/'results/scores').mkdir()
 for seed in (0,1):
  models={};norms={}
  for head in HEADS:
   for arm in ARMS:
    q=make(arm).eval().requires_grad_(False);q.load_state_dict(torch.load(E/f'models/{head}_{arm}_seed{seed}_update0600.pt',weights_only=True,map_location='cpu')['state']);models[head,arm]=q;norms[head,arm]=np.load(E/f'configs/norm_{head}_{arm}.npz')
  for path in sorted((E/'data/evaluation').glob('*.npz')):
   d=np.load(path);name=path.stem;y=d['y'];v=d['v'];roots=d['root_ids'];novel=v.all(1)&np.array([tuple(bool(a)for a in yy)not in signatures for yy in y]);scores={}
   if seed==0:support.append({'cohort':name,'valid6':int(v.all(1).sum()),'novel6':int(novel.sum()),'novel_roots':len(set(roots[novel])),'training_valid_signature_count':len(signatures)})
   for head in HEADS:
    g=slice(0,3)if head=='Holding'else slice(3,6)
    for arm in ARMS:
     n=norms[head,arm];raw=d['node'if arm=='node'else head];x=((raw-n['mean'])/n['std']).astype(np.float32)
     with torch.inference_mode():p=models[head,arm](torch.tensor(x)).squeeze(-1).sigmoid().numpy()
     scores[head,arm]=p;out.append({'seed':seed,'head':head,'arm':arm,'cohort':name,**metrics(p,y[:,g],v[:,g],roots)})
     if novel.any():out.append({'seed':seed,'head':head,'arm':arm,'cohort':name+'__unseen6',**metrics(p[novel],y[novel,g],v[novel,g],roots[novel])})
     visible=d['visibility'].any(1)
     for j,c in enumerate(COLORS):
      for state,mask in [('visible_either',visible[:,j]),('not_detected_both',~visible[:,j])]:
       ix=mask&v[:,g][:,j].astype(bool);pred=p[:,j]>=.5;truth=y[:,g][:,j].astype(bool)
       visibility.append({'seed':seed,'head':head,'arm':arm,'cohort':name,'color':c,'visibility':state,'valid':int(ix.sum()),'positive':int((ix&truth).sum()),'FP':int((ix&~truth&pred).sum()),'FN':int((ix&truth&~pred).sum())})
     if name=='official_validation':
      by=collections.defaultdict(list)
      for i,sid in enumerate(d['state_ids']):
       for hint in segments[str(sid)]['candidate_hints']:by[hint['annotation_index']].append(i)
      for j,c in enumerate(COLORS):
       records=[]
       for ann,ii in sorted(by.items()):
        ix=np.array(sorted(set(ii)));known=v[ix,g][:,j].astype(bool);truth=y[ix,g][:,j].astype(bool)
        if(known&truth).any()and(known&~truth).any():records.append({'annotation_index':ann,'root_id':str(roots[ix[0]]),'positive':int((known&truth).sum()),'negative':int((known&~truth).sum()),'correct':bool(((p[ix,j]>=.5)==truth)[known].all())})
       paired.append({'head':head,'arm':arm,'seed':seed,'color':c,'eligible':len(records),'correct':sum(r['correct']for r in records),'records':records})
     if arm=='edge':
      perm=np.arange(len(x))
      for scene in sorted(set(d['scene'])):
       ix=np.where(d['scene']==scene)[0];perm[ix]=np.roll(ix,1)
      for what in['endpoint','object_and_relative']:
       xx=x.copy()
       if what=='endpoint':xx[:,:,74:202]=x[perm,:,74:202]
       else:xx[:,:,:74]=x[perm,:,:74];xx[:,:,202:]=x[perm,:,202:]
       with torch.inference_mode():pp=models[head,arm](torch.tensor(xx)).squeeze(-1).sigmoid().numpy()
       swaps.append({'cohort':name,'head':head,'seed':seed,'swap':what,'mean_absolute_score_change':float(abs(pp-p).mean()),'max_absolute_score_change':float(abs(pp-p).max()),'threshold_flips':int(((pp>=.5)!=(p>=.5)).sum()),'facts':p.size,'permutation':'cyclic by same scene; counterpart feature swap only, no physical counterfactual ground truth'})
   for arm in ARMS:
    pp=np.concatenate([scores['Holding',arm],scores['In',arm]],1);valid=v.all(1);correct=(pp>=.5)==y.astype(bool)
    combined.append({'seed':seed,'arm':arm,'cohort':name,'valid6':int(valid.sum()),'correct6':int((correct.all(1)&valid).sum()),'more_than_one_Holding':int(((pp[:,:3]>=.5).sum(1)>1).sum()),'same_object_Holding_and_In':int(((pp[:,:3]>=.5)&(pp[:,3:]>=.5)).sum()),'novel6_valid':int(novel.sum()),'novel6_correct':int((correct.all(1)&novel).sum())})
    np.savez_compressed(E/f'results/scores/{name}_{arm}_seed{seed}.npz',scores=pp,state_ids=d['state_ids'],indices=d['indices'])
 save(E/'results/actual_metrics.json',out);save(E/'results/combined6_metrics.json',combined);save(E/'results/matched_pairs.json',paired);save(E/'results/endpoint_object_swaps.json',swaps);save(E/'results/visibility_metrics.json',visibility);save(E/'results/unseen_signature_support.json',support)
 for name,rr in [('combined6_metrics',combined),('endpoint_object_swaps',swaps),('visibility_metrics',visibility),('unseen_signature_support',support)]:
  with(R/f'{name}.csv').open('w')as f:
   wr=csv.DictWriter(f,fieldnames=list(rr[0]));wr.writeheader();wr.writerows(rr)
 save(E/'results/evaluation_audit.json',{'utc':now(),'seconds':time.perf_counter()-start,'model_selection':False,'T_calls':0,'posthoc_model_feature_swap_only':True,'all_outputs_uncalibrated_scores':True})
 print('VISUAL_NODE_EDGE_ACTUAL_COMPLETE',json.dumps([{k:r[k]for k in ['cohort','head','seed','arm','all3_correct','all3_valid','mean_balanced_error']}for r in out if r['cohort']in['official_validation','AB_heldout16','C_validation','new_AB_validation','D12_seen_actual135']]),flush=True)
if __name__=='__main__':main()
