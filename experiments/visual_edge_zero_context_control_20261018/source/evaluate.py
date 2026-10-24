from common import *
from metrics import metrics
import torch,collections,csv

def main():
 assert not(E/'results/actual_metrics.json').exists()
 for p,h in read(E/'configs/final_heads_frozen.json')['sha256'].items():assert sha(p)==h,p
 torch.set_num_threads(4);out=[];combined=[];paired=[];strata=[];flips=[];reproduction=[]
 segments={r['state_id']:r for r in read(BASE/'data/private/official_segments.json')}
 rows=read(BASE/'data/public/states.json');raw=np.load(BASE/'data/public/raw_features.npz');(E/'results/scores').mkdir()
 for seed in(0,1):
  models={};norms={}
  for head in HEADS:
   for arm in['zero_context','full_context']:
    q=make().eval().requires_grad_(False);file=E/f'models/{head}_zero_context_seed{seed}_update0600.pt'if arm=='zero_context'else E/f'models/reference_full_{head}_seed{seed}.pt'
    q.load_state_dict(torch.load(file,weights_only=True,map_location='cpu')['state']);models[head,arm]=q
   norms[head]=np.load(E/f'configs/norm_{head}.npz')
  for path in sorted((E/'data/evaluation').glob('*.npz')):
   d=np.load(path);name=path.stem;y=d['y'];v=d['v'];roots=d['root_ids'];scores={};ref=np.load(BASE/f'results/scores/{name}_edge_seed{seed}.npz')
   assert np.array_equal(ref['state_ids'],d['state_ids'])
   for head,off in [('Holding',0),('In',3)]:
    g=slice(off,off+3);n=norms[head];full=((d[head]-n['mean'])/n['std']).astype(np.float32)
    for arm in['zero_context','full_context']:
     x=full.copy()
     if arm=='zero_context':x[:,:,74:]=0;assert np.count_nonzero(x[:,:,74:])==0;assert np.array_equal(x[:,:,:74],full[:,:,:74])
     with torch.inference_mode():p=models[head,arm](torch.tensor(x)).squeeze(-1).sigmoid().numpy()
     if arm=='full_context':
      assert np.array_equal(p,ref['scores'][:,g]),(name,head,seed,float(abs(p-ref['scores'][:,g]).max()))
      reproduction.append({'cohort':name,'head':head,'seed':seed,'bit_identical_saved_reference_scores':True})
     scores[head,arm]=p;out.append({'seed':seed,'head':head,'arm':arm,'cohort':name,**metrics(p,y[:,g],v[:,g],roots)})
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
     if name not in ['C_all_available','C_validation','official_validation','AB_heldout16','D12_seen_actual135']:continue
     counters={}
     for i,index in enumerate(d['indices']):
      rr=rows[int(index)];family='intermediate_v3'if'intermediate_v3'in rr['root_id']else'legacy_semantics'if'semantics'in rr['root_id']else'official_state'if rr['root_id'].startswith('official')else'identity_D12'if'identity'in rr['root_id']else'other'
      for j,color in enumerate(COLORS):
       if not v[i,off+j]:continue
       vv=d['visibility'][i,:,j].astype(bool);visibility='both'if vv.all()else'static_only'if vv[0]else'wrist_only'if vv[1]else'neither'
       geom=raw['geometry'][int(index),0,j];lo=np.maximum(geom[:2],BOX[:2]);hi=np.minimum(geom[2:4],BOX[2:]);area=np.prod(np.maximum(geom[2:4]-geom[:2],0));frac=float(np.prod(np.maximum(hi-lo,0))/area)if area>0 else None
       overlap='not_detected_static'if frac is None else'outside_ROI'if frac<=1e-8 else'fully_in_ROI'if frac>=1-1e-6 else'partial_ROI'
       truth=bool(y[i,off+j]);pr=bool(p[i,j]>=.5);conf='TP'if truth and pr else'FN'if truth else'FP'if pr else'TN'
       for dimension,level in [('color',color),('root',str(roots[i])),('origin',family),('visibility',visibility),('target_bbox_drawer_ROI',overlap),('color_x_origin',color+'/'+family),('origin_x_visibility',family+'/'+visibility)]:
        key=(dimension,level)
        if key not in counters:counters[key]=collections.Counter({'TP':0,'FN':0,'FP':0,'TN':0})
        counters[key][conf]+=1
     for(dimension,level),c in counters.items():strata.append({'cohort':name,'seed':seed,'arm':arm,'head':head,'dimension':dimension,'level':level,**dict(c)})
    for i,j in np.argwhere((scores[head,'zero_context']>=.5)!=(scores[head,'full_context']>=.5)):
     if not v[i,off+j]:continue
     truth=bool(y[i,off+j]);flips.append({'cohort':name,'head':head,'seed':seed,'index':int(d['indices'][i]),'state_id':str(d['state_ids'][i]),'root_id':str(roots[i]),'color':COLORS[j],'truth':truth,'zero_score':float(scores[head,'zero_context'][i,j]),'full_score':float(scores[head,'full_context'][i,j]),'effect':'improvement'if bool(scores[head,'full_context'][i,j]>=.5)==truth else'regression'})
   for arm in['zero_context','full_context']:
    pp=np.concatenate([scores['Holding',arm],scores['In',arm]],1);valid=v.all(1)
    combined.append({'seed':seed,'arm':arm,'cohort':name,'valid6':int(valid.sum()),'correct6':int((((pp>=.5)==y.astype(bool)).all(1)&valid).sum())});np.savez_compressed(E/f'results/scores/{name}_{arm}_seed{seed}.npz',scores=pp,state_ids=d['state_ids'],indices=d['indices'])
 for name,rr in [('actual_metrics',out),('combined6_metrics',combined),('matched_pairs',paired),('matched_error_decomposition',strata),('decision_flips',flips),('reference_reproduction',reproduction)]:
  save(E/f'results/{name}.json',rr)
  if name in['combined6_metrics','matched_error_decomposition','decision_flips']:
   with(R/f'{name}.csv').open('w')as f:wr=csv.DictWriter(f,fieldnames=list(rr[0]));wr.writeheader();wr.writerows(rr)
 save(E/'results/evaluation_audit.json',{'utc':now(),'reference_scores_identical_cases':len(reproduction),'new_fits':4,'T_calls':0,'DINO_calls':0,'threshold':.5,'selection_or_tuning':False,'all_cohorts_previously_viewed':True})
 print('MATCHED_CONTROL_EVALUATED',len(out),'metric groups',flush=True)
if __name__=='__main__':main()
