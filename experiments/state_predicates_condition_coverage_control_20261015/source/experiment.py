from pathlib import Path
import json,hashlib,datetime
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_condition_coverage_control_20261015';B=P/'data/predicate_condition_coverage_control_20261015';TRAIN=P/'experiments/state_predicates_condition_training_20261015';OLD=P/'experiments/state_predicates_one_step_20261014';ABC=P/'experiments/state_predicates_state_dependence_20261014';RGB=P/'experiments/state_predicates_rgb_registry_v1_20261014';COPY=P/'experiments/state_predicates_copy_effect_diagnostic_20261014';CAND=P/'experiments/state_predicates_held_down_20261014/candidate';COLORS=['red','blue','pink'];NAMES=['held(red)','held(blue)','held(pink)','in(red,drawer)','in(blue,drawer)','in(pink,drawer)','open(drawer)']
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def frozen():
 q=read(E/'configs/protocol.json')
 for p,h in q['frozen_sha256'].items():assert sha(p)==h,p
 assert sha(B/'roots_manifest.json')==q['roots_manifest_sha256'];assert sha(E/'configs/root_exclusion.json')==q['exclusion_sha256'];return q

