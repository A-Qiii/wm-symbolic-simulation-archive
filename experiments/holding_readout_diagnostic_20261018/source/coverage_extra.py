from config import *
import collections
def main():
 ps=read(PRIOR/'data/transition_train/pairs.json');ss=read(PRIOR/'data/transition_train/states.json');old=read(REP/'data/_private/labels.json');new=read(PRIOR/'data/ABC/_private/labels.json');cache={}
 def lab(i):
  s=ss[i]
  if s.get('dataset_group')=='new':return new[str(int(s['id'][1:]))]
  if str(i)in old:return old[str(i)]
  src=Path(s.get('source',''));f=src.parent/'labels_ownership_v2_private.json'
  if f.exists()and src.name in ['initial_public.npz','final_public.npz']:
   if f not in cache:cache[f]=read(f)
   l=cache[f]['before'if src.name=='initial_public.npz'else'after']
  elif 'longpair'in s['root_id']:
   f=src.parent.parent/'_private'/f"{s['root_id']}__t{int(src.stem):05d}.json"
   if not f.exists():return None
   if f not in cache:cache[f]=read(f)
   frame=next(r for r in cache[f]['frames']if r['frame']==int(src.stem));l=frame['labels'];assert l['held_definition']=='bilateral force-bearing descendant finger contacts; no external support'
  else:return None
  q=l['continuous']['drawer_m'];return {'y':l['holding_derived']+[False]*3+[q<=.02,q>=.18,.02<q<.18],'v':l['categorical_valid'][:3]+[False]*3+[min(abs(q-.02),abs(q-.18))>.003]*3}
 labs=[lab(i)for i in range(len(ss))];index={}
 def seq(p):return p.get('episode_id')or(p['root_id']+'::'+p['branch']if 'branch'in p else None)
 for i,p in enumerate(ps):
  if seq(p)is not None:index[(seq(p),p.get('step',p.get('slot')))]=i
 cats=[];records=[]
 for i,p in enumerate(ps):
  b=labs[p['before']];a=labs[p['after']];step=p.get('step',p.get('slot'));prev=ps[index[(seq(p),step-1)]]if seq(p)is not None and(seq(p),step-1)in index else None;c=[]
  if b and all(b['v'][:3])and b['v'][7]and b['y'][7]and not any(b['y'][:3])and p['skill']=='close_drawer':
   c+=['empty_open_close_before_known']
   if prev and prev['after']==p['before']and prev['skill']=='open_drawer':c+=['executed_open_then_empty_close']
   if prev and prev['after']==p['before']and prev['skill']=='place_in_drawer':c+=['executed_place_then_empty_close']
   if prev is None:c+=['no_recorded_immediate_predecessor_empty_close']
  if prev and prev['after']==p['before']and prev['skill']=='open_drawer'and p['skill']=='close_drawer':
   c+=['executed_open_close_any_Holding'];
   if b and all(b['v'][:3])and any(b['y'][:3]):c+=['executed_open_close_held']
  cats.append(c);records.append({'pair_id':p['id'],'root_id':p['root_id'],'split':p['split'],'skill':p['skill'],'posthold':p['action'][7],'sequence':seq(p),'step':step,'before_Holding_valid':bool(b and all(b['v'][:3])),'after_Holding_valid':bool(a and all(a['v'][:3])),'categories':c})
 categories=['empty_open_close_before_known','executed_open_then_empty_close','executed_place_then_empty_close','no_recorded_immediate_predecessor_empty_close','executed_open_close_any_Holding','executed_open_close_held'];out=[]
 for seed in [0,1]:
  sc=np.load(ABL/'configs'/f'schedule_seed{seed}.npz')
  for u in [750,1000]:
   for stream,ix in [('single',sc['single'][:u].ravel()),('multi_forward',sc['multi'][3:u:4].ravel()),('multi_supervised',sc['multi'][3:u:4,:,1:].ravel())]:
    counts=np.bincount(ix,minlength=len(ps))
    for cat in categories:
     for hold in ['all',0,1]:
      ids=[i for i,p in enumerate(ps)if counts[i]and cat in cats[i]and(hold=='all'or p['action'][7]==hold)];out.append({'seed':seed,'updates_prefix':u,'stream':stream,'category':cat,'posthold':hold,'unique_transitions':len(ids),'roots':len({ps[i]['root_id']for i in ids}),'draws':int(counts[ids].sum())})
 save(E/'results/exact_sequence_coverage.json',out);save(E/'results/coverage_pair_provenance.json',records);save(E/'results/coverage_interpretation.json',{'earlier_table':'historical_transition_coverage requires BOTH before and after Holding validity, omits recoverable longpair snapshots; its state-linked predecessor flag is not evidence of actual immediate execution; use exact_sequence_coverage for final empty-close statistics','current_table':'exact_sequence_coverage includes recovered longpair physical labels, requires known precondition for empty close and matches recorded same episode/branch, consecutive slot, and exact shared boundary for sequential categories','missing_post_Holding_not_assumed_negative':True});refs=read(E/'configs/references.json');refs['sha256'].update({str(p):sha(p)for p in cache});save(E/'configs/references.json',refs)
 for r in out:
  if r['seed']==0 and r['updates_prefix']==1000 and r['stream']=='single'and r['posthold']==1:print('COVERAGE',r,flush=True)
if __name__=='__main__':main()
