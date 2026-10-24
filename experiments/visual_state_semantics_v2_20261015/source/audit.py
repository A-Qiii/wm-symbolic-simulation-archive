from common import *
from labels_v2_1 import evaluate,SPEC,classify_drawer
import gzip,argparse,collections,copy

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','full'],required=True);arg=ap.parse_args();m=read(B/'manifest.json');ww=[];calls=[];differences=[];cases=[]
 for ep in m['episodes']:
  assert not ep['error'];f=B/'episodes'/ep['episode_id'];wins=read(f/'_private/windows.json');aa=read(f/'_private/actions.json');pass
  with gzip.open(f/'_private/full_physics_trace.json.gz','rt')as h:trace=json.load(h)
  assert len(trace)==ep['frames']and all(r['physics_ticks']==r['frame']*8 for r in trace)
  for row in wins:
   t=trace[row['anchor']];new=evaluate(t['physical']);oldcurrent=row['new_labels'];assert row['indices']==[row['anchor']-k for k in[63,42,21,0]]
   for p,h in zip(row['public_frames'],row['public_sha256']):assert sha(p)==h
   for p in row['public_frames']:
    z=np.load(p);assert set(z.files)=={'rgb_static','rgb_gripper','robot_obs'}
   if new['y']!=oldcurrent['y']or new['v']!=oldcurrent['v']:differences.append({'window_id':row['window_id'],'stored_version':oldcurrent['version'],'stored_y':oldcurrent['y'],'stored_v':oldcurrent['v'],'recomputed_y':new['y'],'recomputed_v':new['v']})
   ww.append(row|{'stored_labels_version':oldcurrent['version'],'targets_v2_1':new,'full_physics_source':str(f/'_private/full_physics_trace.json.gz')});cases.append({'root_id':ep['root_id'],'condition':ep['condition'],'split':ep['split'],'anchor':row['anchor'],'valid9':all(new['v']),'in_positive':[i for i in range(3)if new['y'][3+i]and new['v'][3+i]],'old_In':[i for i in range(3)if row['old_labels']['categorical'][i]==3],'drawer_class':new['drawer_class'],'drawer_valid':all(new['v'][6:]),'q':new['drawer_q_m'],'outside_with_drawer_contact':any(d['outside']and bool(d['load_bearing_paths'])for d in new['objects'].values()),'indirect_protruding':any(d['outside']and any(len(p)>2 for p in d['load_bearing_paths'])for d in new['objects'].values())})
  for a in aa:
   a['before_v2_1']=evaluate(trace[a['start']]['physical']);a['after_v2_1']=evaluate(trace[a['end']]['physical']);calls.append(a|{'root_id':ep['root_id'],'condition':ep['condition'],'split':ep['split'],'episode_id':ep['episode_id']})
 # A real saved positive state provides a fixture; counterfactual modifications are unit tests only.
 sample=next(e for e in m['episodes']if e['condition']=='single_inside')
 with gzip.open(B/'episodes'/sample['episode_id']/'_private/full_physics_trace.json.gz','rt')as h:t=json.load(h)
 snap=t[127]['physical'];base=evaluate(snap);target=next(j for j in range(3)if base['y'][j+3]and base['v'][j+3]);name='block_'+COLORS[target];uid=snap['objects'][name]['uid'];tests=[]
 def check(desc,s,expected,valid=True):
  out=evaluate(s);assert out['y'][target+3]==expected and out['v'][target+3]==valid,(desc,out);tests.append({'case':desc,'expected':expected,'valid':valid,'passed':True})
 check('actual contained and supported',snap,True);s=copy.deepcopy(snap);s['contacts']=[c for c in s['contacts']if c[1]!=uid and c[2]!=uid];check('suspended inside no contact',s,False);s=copy.deepcopy(snap);s['held']=[name];check('held cannot be In',s,False);s=copy.deepcopy(snap);s['objects'][name]['position'][0]+=.8;check('outside with original contact cannot be In',s,False)
 for desc,nz in [('side brushing',0),('downward contact',-1)]:
  s=copy.deepcopy(snap)
  for c in s['contacts']:
   if c[1]==uid:c[7]=[1,0,nz]
   elif c[2]==uid:c[7]=[-1,0,-nz]
  check(desc,s,False)
 for q,expected,valid in [(.005,0,True),(.1,2,True),(.22,1,True),(.02,0,False),(.18,1,False)]:
  y,v=classify_drawer(q);assert y[expected]and all(v)==valid;tests.append({'case':'drawer '+str(q),'class':expected,'valid':valid,'passed':True})
 split={}
 for name in ['train','dev']:
  rows=[r for r in ww if r['split']==name]
  if not rows:continue
  y=np.array([r['targets_v2_1']['y']for r in rows]);v=np.array([r['targets_v2_1']['v']for r in rows]);split[name]={'windows':len(rows),'roots':len({r['root_id']for r in rows}),'valid9':int(v.all(1).sum()),'positive_valid':(y&v).sum(0).tolist(),'negative_valid':((~y)&v).sum(0).tolist(),'invalid_by_fact':(~v).sum(0).tolist(),'actual_outside_supported_windows':sum(r['outside_with_drawer_contact']for r in cases if r['split']==name),'actual_indirect_protruding_windows':sum(r['indirect_protruding']for r in cases if r['split']==name)}
 if arg.phase=='full':assert len(m['episodes'])==171 and len({e['root_id']for e in m['episodes']})==27
 out={'phase':arg.phase,'infrastructure_passed':True,'full_physics_saved':True,'physics240Hz_control30Hz':True,'no_D':True,'windows':len(ww),'calls':len(calls),'episodes':len(m['episodes']),'splits':split,'label_migration_count':len(differences),'unit_tests':tests,'all_fixture_attempts_retained':True};save(E/'results'/('pilot_audit.json'if arg.phase=='pilot'else'data_audit.json'),out);save(E/'data'/('pilot_windows.json'if arg.phase=='pilot'else'windows.json'),ww);save(E/'results'/('pilot_label_migration.json'if arg.phase=='pilot'else'label_migration.json'),differences);save(E/'data'/('pilot_calls.json'if arg.phase=='pilot'else'calls.json'),calls);save(E/'results'/('pilot_coverage.json'if arg.phase=='pilot'else'coverage.json'),cases);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
