from common import *
def main():
 collection=read(E/'results/D2_collection.json');states=[];pairs=[];labels={};outcomes={};slots=[];paths={};prep=0
 for episode in collection:
  prep+=len(episode['preparation'])
  for task in episode['tasks']:
   slots.append({'root_id':episode['root_id'],'condition':episode['condition'],'slot':task['slot'],'skill':task['skill'],'status':task['status']})
   if not task.get('native'):continue
   ids=[]
   for key in['before','after']:
    d=task[key];path=d['public_path']
    if path not in paths:
     i=len(states);paths[path]=i;states.append({'id':f's{i:05d}','root_id':episode['root_id'],'split':'test','path':path});labels[str(i)]=d['labels']
    ids.append(paths[path])
   pid=f'p{len(pairs):05d}';pairs.append({'id':pid,'root_id':episode['root_id'],'split':'test','before':ids[0],'after':ids[1],'skill':task['skill'],'action':action(task['skill'],None,96)});outcomes[pid]={'pair_id':pid,'status':task['status'],'condition':episode['condition'],'target':episode['target'],'reason':task['native'].get('reason')}
 assert len(slots)==90 and prep<=24;save(E/'data/D2/public/states.json',states);save(E/'data/D2/public/pairs.json',pairs);save(E/'data/D2/_private/labels.json',labels);save(E/'data/D2/_private/outcomes.json',outcomes);save(E/'results/D2_slots.json',slots);save(E/'results/D2_denominators.json',{'roots':12,'planned_task_slots':90,'executed_tasks':len(pairs),'states':len(states),'preparation_calls':prep,'status_counts':{status:sum(r['status']==status for r in slots)for status in sorted({r['status']for r in slots})},'all_roots_retained':True,'replacement_roots':0});print('D2_CONVERTED',read(E/'results/D2_denominators.json'),flush=True)
if __name__=='__main__':main()
