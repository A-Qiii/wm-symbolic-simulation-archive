from config import *
import argparse,collections
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['ABC','D4'],required=True);phase=ap.parse_args().phase;episodes=read(E/'results'/f'collection_{phase}.json');states=[];pairs=[];labels={};outcomes={};slots=[];paths={};rootcov={};rawhash={}
 for ep in episodes:
  assert ep['error']is None;folder=Path(ep['folder']);points=read(folder/'_private/states.json');frame={}
  for p in points:
   path=p['public_path'];assert path not in paths;i=len(states);paths[path]=i;frame[p['frame']]=i
   strata=[]
   if ep['branch']=='normal':
    for tag,name in [('before_call:1','pre_grasp'),('after_call:1','held_open'),('phase_end:2:above_drawer_interior','hover_above_drawer'),('before_release','before_release'),('after_call:2','post_place')]:
     if tag in p['tags']:strata.append(name)
   if ep['branch']=='held_closed'and'after_call:0'in p['tags']:strata.append('held_closed')
   boundary=any(t.startswith(('before_call:','after_call:'))for t in p['tags'])
   states.append({'id':f'n{i:05d}','root_id':ep['root_id'],'environment':ep['environment'],'split':ep['split'],'target':ep['target'],'branch':ep['branch'],'path':path,'frame':p['frame'],'strata':strata,'skill_boundary':boundary,'phase':p['phase'],'tags':p['tags']});labels[str(i)]=p['labels'];rawhash[path]=sha(path)
  for call in ep['slots']:
   slots.append({k:ep[k]for k in ['root_id','split','target','branch']}|{k:call[k]for k in ['slot','skill','status']})
   if'after_frame'not in call:continue
   b,a=frame[call['before_frame']],frame[call['after_frame']];pid=f't{len(pairs):05d}';pairs.append({'id':pid,'root_id':ep['root_id'],'split':ep['split'],'target':ep['target'],'branch':ep['branch'],'slot':call['slot'],'before':b,'after':a,'skill':call['skill'],'action':call['action']});outcomes[pid]={'status':call['status'],'native':call['native'],'start_frame':call['start_frame'],'end_frame':call['end_frame'],'primitive_end':call['primitive_end']}
  for f in ['states.json','calls.json','summary.json','provenance.json']:rawhash[str(folder/'_private'/f)]=sha(folder/'_private'/f)
 for split in sorted({s['split']for s in states}):
  rootcov[split]={}
  for colour in COLORS:
   rootcov[split][colour]={}
   for stratum in ['pre_grasp','held_open','hover_above_drawer','before_release','post_place','held_closed']:
    ix=[i for i,s in enumerate(states)if s['split']==split and s['target']==colour and stratum in s['strata']];j=COLORS.index(colour)
    def achieved(i):
     l=labels[str(i)];y=l['y'];v=l['v']
     if stratum in ['held_open','hover_above_drawer']:return v[j]and v[7]and y[j]and y[7]
     if stratum=='held_closed':return v[j]and v[6]and y[j]and y[6]
     if stratum=='post_place':return v[3+j]and all(v[:3])and y[3+j]and not any(y[:3])
     if stratum=='pre_grasp':return all(v[:3])and not any(y[:3])
     return None
    good=[i for i in ix if achieved(i)];rootcov[split][colour][stratum]={'states':len(ix),'roots':len({states[i]['root_id']for i in ix}),'full9_states':sum(all(labels[str(i)]['v'])for i in ix),'unknown_facts':sum(9-sum(labels[str(i)]['v'])for i in ix),'condition_confirmed_states':len(good)if stratum!='before_release'else None,'condition_confirmed_roots':len({states[i]['root_id']for i in good})if stratum!='before_release'else None}
 d=E/'data'/phase;save(d/'public/states.json',states);save(d/'public/pairs.json',pairs);save(d/'_private/labels.json',labels);save(d/'_private/outcomes.json',outcomes);save(d/'_private/slots.json',slots)
 summary={'phase':phase,'roots':len({x['root_id']for x in episodes}),'episodes':len(episodes),'actual_transitions':len(pairs),'planned_slots':len(slots),'status_counts':dict(collections.Counter(x['status']for x in slots)),'states':len(states),'full9_states':sum(all(x['v'])for x in labels.values()),'unknown_facts':sum(9-sum(x['v'])for x in labels.values()),'coverage':rootcov,'collection_seconds':sum(x['elapsed_seconds']for x in episodes),'raw_bytes':sum(p.stat().st_size for x in episodes for p in Path(x['folder']).rglob('*')if p.is_file()),'replacement_roots':0}
 save(E/'results'/f'coverage_{phase}.json',summary);save(E/'configs'/f'raw_{phase}_hashes.json',{'sha256':rawhash});print('DATA_PREPARED',phase,json.dumps({k:v for k,v in summary.items()if k!='coverage'}),flush=True)
 if phase=='ABC':
  ix={p[k]for p in pairs[:50]for k in ['before','after']};paths=[Path(states[i]['path'])for i in ix];save(E/'results/first50_storage.json',{'transitions':len(pairs[:50]),'unique_boundary_files':len(paths),'boundary_observation_bytes':sum(p.stat().st_size for p in paths),'average_boundary_bytes_per_transition_deduplicated':sum(p.stat().st_size for p in paths)/len(pairs[:50]),'whole_collection_bytes_including_continuous_videos_and_physical_trace':summary['raw_bytes'],'whole_collection_transitions':len(pairs),'note':'Boundary bytes for first50 actual transitions; continuous episode videos/trace measured separately over whole collection, not misattributed to exactly50.'})
if __name__=='__main__':main()
