from config import *
def main():
 data=read(HOR/'results/trajectories.json');states=[];seen={};public=[];truth=[];sources={str(HOR/'results/trajectories.json'):sha(HOR/'results/trajectories.json')}
 def state(path):
  assert '/public/'in path and '/_private/'not in path;h=content_hash(path)
  if h not in seen:seen[h]=len(states);states.append({'index':len(states),'path':path,'content_sha256':h,'sha256':sha(path)})
  return seen[h]
 for group,roots in data.items():
  for r in roots:
   ps=[]
   for row in r['steps']:
    if not row['observed']:continue
    rr={'step':row['horizon'],'skill':row['skill'],'action':row['action'],'before':state(row['before_path']),'after':state(row['after_path'])}
    if ps:assert ps[-1]['after']==rr['before']
    ps.append(rr)
   public.append({'trajectory_id':group+':'+r['root_id'],'source':group,'root_id':r['root_id'],'steps':ps})
   truth.append({'trajectory_id':group+':'+r['root_id'],'source':group,'root_id':r['root_id'],'four_step_eligible':r['four_step_eligible'],'contains_refusal':r.get('contains_refusal_in_first4',False),'planned_steps':r['planned_steps'],'observed_steps':r['observed_steps'],'exclusion_reason':r['exclusion_reason'],'steps':r['steps']})
 save(E/'data/public/part1_manifest.json',{'states':states,'trajectories':public});save(E/'data/evaluator/part1_truth.json',truth);save(E/'configs/part1_source_hashes.json',{'sha256':sources|{s['path']:s['sha256']for s in states}})
 print('PART1_PREPARED',len(states),'states',sum(len(t['steps'])for t in public),'transitions',flush=True)
if __name__=='__main__':main()
