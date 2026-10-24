from pathlib import Path
import json,hashlib,datetime,csv,importlib.util
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_implementation_diagnostic_20261015';D=P/'experiments/state_predicates_independent_D_20261015';DB=P/'data/predicate_independent_D_20261015';TRAIN=P/'experiments/state_predicates_condition_training_20261015';OLD=P/'experiments/state_predicates_one_step_20261014';ABC=P/'experiments/state_predicates_state_dependence_20261014';RGB=P/'experiments/state_predicates_rgb_registry_v1_20261014';CAND=P/'experiments/state_predicates_held_down_20261014/candidate';COLORS=['red','blue','pink'];NAMES=['held(red)','held(blue)','held(pink)','in(red)','in(blue)','in(pink)','open(drawer)']
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def module(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def csvsave(p,rows):
 if not rows:return
 with Path(p).open('w',newline='',encoding='utf-8-sig')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(dict,list))else v for k,v in r.items()}for r in rows)
def frozen():
 q=read(E/'configs/scope.json')
 for p,h in q['frozen_files'].items():assert sha(p)==h,p
 return q
def action(r):
 a=np.zeros(9,np.float32);skill=r['skill'];k=0 if skill=='open_drawer'else 1 if skill.startswith('lift_')else 2 if skill=='place_in_drawer'else 3;a[k]=1
 if k==1:
  color=r['declared_arguments']['color'];assert skill=='lift_'+color+'_block_table';a[4+COLORS.index(color)]=1
 else:assert r['declared_arguments']=={}
 a[7]=r['declared_primitive_budget']/900;a[8]=r['declared_posthold_controls']/96;return a
def verify_sealed():
 output=[]
 for root,expected in [(D,'d12a08430c118741527c54eb44110c3b3929c5a9003193546c9f32ce2c7912f4'),(DB,'573aadc3c640e15264d81defd2aaff4dde53603acdf585c73a1248330ff36e28'),(TRAIN,'daaaa75497eb09be78770ba7f08f5e11796a36cdfaa1f7948eabb04e13ba8018')]:
  assert sha(root/'delivery_manifest.json')==expected;man=read(root/'delivery_manifest.json');entries=man.get('entries',man.get('files'));total=0
  for entry in entries:
   p=root/entry['path'];assert p.stat().st_mode&0o222==0 and sha(p)==entry['sha256'],str(p);total+=p.stat().st_size
  assert len([p for p in root.rglob('*')if p.is_file()])==len(entries)+1
  output.append({'root':str(root),'manifest_sha256':expected,'entries_verified':len(entries),'bytes':total,'all_read_only':True})
 return output
