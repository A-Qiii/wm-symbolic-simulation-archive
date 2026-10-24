from common import *
from symbolic import *
def main():
 assert not(E/'results/rule_evaluation.json').exists();m=read(TRAIN/'data/_private/evaluation_metadata.json');a=dict(np.load(TRAIN/'data/_private/targets.npz'));wi={r['window_id']:i for i,r in enumerate(read(E/'data/public/windows.json'))};p=np.load(E/'results/reader_probabilities.npy');tr=[i for i,r in enumerate(m)if r['split']=='train'];dv=[i for i,r in enumerate(m)if r['split']=='dev'];assert len(tr)==132 and len(dv)==42 and a['before_valid'].all()and a['after_valid'].all()
 if (E/'configs/rules_frozen.json').exists():operators=read(E/'configs/rules_frozen.json')['operators']
 else:
  operators,audit=learn([m[i]for i in tr],a['before_y'][tr],a['after_y'][tr]);save(E/'configs/rules_frozen.json',{'created_utc':now(),'operators':operators,'training_supervision':'AB physical sevenfacts + actual controller success/refusal only; no future GT at inference','scope':'fixed predicate library, learned precondition/effect rules; no predicate invention','audit':audit})
 beforep=p[[wi[r['before_window_id']]for r in m]];afterp=p[[wi[r['after_window_id']]for r in m]];out=[]
 for rulesname,rules in [('learned',operators),('fixed_baseline',fixed_baseline())]:
  for inputname,states in [('GT_symbols',a['before_y'].astype(int)),('new_visual_reader',decode(beforep))]:
   for split,ids in [('AB_train',tr),('C_dev',dv)]:
    details=[]
    for i in ids:
     r=m[i];s=states[i];x=binding(r,s);matches=[o for o in rules if o['family']==r['family']and applicable(o,s,x)];expected=r['primitive_status']=='success';predicted=bool(matches);nxt=apply(matches[0],s,x)if matches else None;correctnext=bool(nxt is not None and np.array_equal(nxt,a['after_y'][i]));details.append({'pair_id':r['pair_id'],'root_id':r['root_id'],'family':r['family'],'condition':r['condition'],'status':r['primitive_status'],'expected_applicable':expected,'predicted_applicable':predicted,'applicability_correct':predicted==expected,'next_all7_correct':correctnext if expected else None,'multiple_matching_operators':len(matches)>1,'before':s.tolist(),'truth_after':a['after_y'][i].astype(int).tolist(),'prediction_after':list(nxt)if nxt else None})
    out.append({'rules':rulesname,'input':inputname,'split':split,'calls':len(ids),'roots':len({m[i]['root_id']for i in ids}),'applicability_correct':sum(r['applicability_correct']for r in details),'real_successful_calls':sum(r['expected_applicable']for r in details),'success_effect_all7_correct':sum(r['next_all7_correct']is True for r in details),'details':details})
 # Dataset-level query independence: alias/permutation checked in training; logical round trip tests.
 for o in operators:assert not any(c in str(o)for c in COLORS)
 s=(0,0,0,0,0,0,0);g=[(3,1),(6,0)];plan=search(s,g,operators);save(E/'results/symbolic_smoke.json',{'initial':s,'goal':g,'result':plan,'learned_successor_only':True});assert plan['status']=='solved'
 save(E/'results/rule_evaluation.json',out);print(json.dumps({'operators':operators,'evaluation':[{k:v for k,v in r.items()if k!='details'}for r in out],'smoke_plan':plan},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
