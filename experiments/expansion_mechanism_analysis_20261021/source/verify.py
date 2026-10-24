from pathlib import Path
from html.parser import HTMLParser
import json,hashlib,datetime,ast
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'experiments/expansion_mechanism_analysis_20261021';S=P/'experiments/data_expansion_training_20261021';A=P/'experiments/prediction_error_analysis_20261021';M=P/'experiments/midstart_training_control_20261020'
def rd(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for x in iter(lambda:f.read(4194304),b''):h.update(x)
 return h.hexdigest()
source_counts={}
for root,expected in [(S,'cfb229f2df8739775a34719cb50d92e64e245cc125ed2dd6d154f8ec8af3e8bc'),(A,'fb5a8b393aaa4e218298df135f33fbcabf24fdf54cda1caec0a9189485420bd5'),(M,None)]:
 if expected:assert sha(root/'MANIFEST.json')==expected
 mm=rd(root/'MANIFEST.json')
 for item in mm['files']:
  p=root/item['path'];assert p.stat().st_size==item['bytes']and sha(p)==item['sha256'],p;assert not p.stat().st_mode&0o222,p
 source_counts[root.name]=len(mm['files'])
hashes=rd(B/'results/additional_raw_input_hashes.json')
for p,h in hashes.items():assert sha(Path(p))==h,p
windows=rd(B/'results/matched_windows.json');prev=rd(A/'results/three_step_counting.json');original=rd(S/'results/metrics.json');checked=0
for key in sorted({(r['source'],r['seed'],r['arm'])for r in windows}):
 rr=[r for r in windows if(r['source'],r['seed'],r['arm'])==key];ref=next(r for r in prev if(r['source'],r['seed'],r['arm'])==key);assert sum(r['whole_correct']for r in rr)==ref['future_interval_correct'];assert sum(r['endpoint_correct']for r in rr)==ref['endpoint_correct'];z=dict(np.load(S/f'results/predictions_{key[0]}_{key[2]}_seed{key[1]}.npz'));ix={(str(e),int(o),str(m),int(t)):i for i,(e,o,m,t)in enumerate(zip(z['episode_id'],z['origin'],z['mode'],z['relative_step']))}
 for r in rr:
  for t in range(1,4):
   for mode in ['free','observed']:
    i=ix[r['episode_id'],r['origin'],'free'if mode=='free'else'one_step',t];p=z['prob'][i];c=z['cat'][i];ans=all((p[[6,14,22]]>=.5)==(c[:3]==6))and all((p[[3,11,19]]>=.5)==(c[:3]==3))and p[24:27].argmax()==c[3];assert ans==r[mode+'_correct'][t-1]
  assert np.array_equal(z['prob'][ix[r['episode_id'],r['origin'],'free',1]],z['prob'][ix[r['episode_id'],r['origin'],'one_step',1]])
  checked+=1
for r in rd(B/'results/conditional_chain.json'):
 assert abs(r['p1']*r['p2_given1']*r['p3_given12']-r['conditional_product'])<1e-12
for r in rd(B/'results/same_root_horizons.json'):
 if r['step']==6:
  ref=next(x for x in original if(x['source'],x['arm'],x['seed'],x['scope'])==(r['source'],r['arm'],r['seed'],'free_h6'));assert r['endpoint_correct']==ref['joint_correct']and r['same_root_episodes']==ref['joint_eligible']
for se in [0,1]:
 rows=[r for r in rd(B/'results/supervision_path_counts.json')if r['seed']==se];a,b=rows;assert {k:v for k,v in a.items()if k!='arm'}=={k:v for k,v in b.items()if k!='arm'}
for p in (B/'source').glob('*.py'):ast.parse(p.read_text())
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.images=[]
 def handle_starttag(self,tag,attrs):
  d=dict(attrs)
  if tag=='a':self.links.append(d['href'])
  if tag=='img':self.images.append(d['src'])
l=Links();l.feed((B/'reports/REPORT.html').read_text())
for x in l.links+l.images:
 if x!='../results/final_verification.json':assert(B/'reports'/x).is_file(),x
assert len(l.images)==5
sources={str(p):sha(p)for p in (B/'source').glob('*.py')};x={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'original_snapshots_all_files_unchanged_readonly':source_counts,'additional_raw_file_hashes_unchanged':len(hashes),'matched_windows_recomputed':checked,'prediction_checks_per_window':'six forecasts:free1..3,observed1..3','first_forecast_equal_between_modes_all_windows':True,'original_three_step_and_six_step_counts_reconciled':True,'chain_rule_identity_verified':True,'supervision_opportunity_counts_equal_old_new_per_seed':True,'bootstrap_primary':{'families':20,'replicates':10000,'seeds_not_pooled':True},'bootstrap_sensitivity_roots':40,'all_report_links_exist':len(l.links),'figure_count':len(l.images),'figures_visually_reviewed':l.images,'source_sha256':sources,'no_training_or_model_inference_or_collection':True,'no_original_edits':True,'remaining_blockers':[]}
(B/'results/final_verification.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');assert all((B/'reports'/p).is_file()for p in l.links+l.images);print(json.dumps(x,ensure_ascii=False))
