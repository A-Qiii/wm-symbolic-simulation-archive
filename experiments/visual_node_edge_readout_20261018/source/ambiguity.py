from common import *
import collections,csv,shutil

def main():
 summary=[];details=[]
 for p in sorted((E/'data/evaluation').glob('*.npz')):
  d=np.load(p)
  for head in HEADS:
   off=0 if head=='Holding' else 3
   for arm in ARMS:
    x=d['node'if arm=='node'else head];groups=collections.defaultdict(list)
    for i in range(len(x)):
     for j,color in enumerate(COLORS):
      if d['v'][i,off+j]:
       vector=x[i,j].copy();assert np.isfinite(vector).all();vector[vector==0]=0.0
       groups[hashlib.sha256(vector.tobytes()).hexdigest()].append((i,j,bool(d['y'][i,off+j])))
    ambiguous=[];bound=0
    for hh,items in groups.items():
     pos=sum(a[2]for a in items);neg=len(items)-pos
     if pos and neg:
      bound+=min(pos,neg);ambiguous.append({'feature_hash':hh,'positive':pos,'negative':neg,'minimum_unweighted_errors':min(pos,neg),'facts':[{'state_id':str(d['state_ids'][i]),'root_id':str(d['root_ids'][i]),'color':COLORS[j],'truth':y,'visible_either':bool(d['visibility'][i,:,j].any())}for i,j,y in items]})
    row={'cohort':p.stem,'head':head,'arm':arm,'ambiguous_feature_groups':len(ambiguous),'facts_in_ambiguous_groups':sum(a['positive']+a['negative']for a in ambiguous),'minimum_unweighted_fact_errors':bound,'valid_facts':int(d['v'][:,off:off+3].sum())};summary.append(row);details.append(row|{'groups':ambiguous})
 save(E/'results/representation_ambiguity.json',{'utc':now(),'method':'Exact float32 readout-input vector collisions with signed zero canonicalized, across known facts of one predicate/cohort, before normalization (invertible affine per feature). Mixed-label group lower bound min(n_positive,n_negative) for any deterministic shared head. Not a claim that original RGB is inherently indistinguishable.','summary':summary,'details':details,'no_training_or_threshold_change':True})
 with(R/'representation_ambiguity.csv').open('w')as f:wr=csv.DictWriter(f,fieldnames=list(summary[0]));wr.writeheader();wr.writerows(summary)
 shutil.copy2(__file__,E/'source/ambiguity.py');print('REPRESENTATION_AMBIGUITY_COMPLETE',json.dumps([r for r in summary if r['cohort']in['AB_heldout16','official_validation','C_validation']]),flush=True)
if __name__=='__main__':main()
