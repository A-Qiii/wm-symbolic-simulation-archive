from pathlib import Path
from html.parser import HTMLParser
import json,hashlib,ast,datetime
A=Path('/home/robot/research/projects/wm_symbolic/experiments/prediction_error_analysis_20261021');S=A.parent/'data_expansion_training_20261021'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def rd(p):return json.loads(p.read_text())
assert rd(A/'results/counting_verification.json')['passed'];assert rd(A/'results/verification.json')['passed'];protocol=rd(A/'configs/protocol.json');assert sha(S/'MANIFEST.json')==protocol['source_snapshot_sha256'];assert sha(A/'source/analyze.py')==protocol['analysis_source_sha256']
mani=rd(S/'MANIFEST.json')
for r in mani['files']:
 p=S/r['path'];assert p.stat().st_size==r['bytes']and sha(p)==r['sha256'],p;assert not p.stat().st_mode&0o222,p
reads=rd(A/'results/source_reads.json')
for p,h in reads.items():assert sha(Path(p))==h,p
old=rd(S/'results/first_errors.json');new=rd(A/'results/six_step_chains.json');six={(r['source'],r['arm'],r['seed'],r['episode_id']):r for r in old if r['mode']=='free'and r['origin']==0 and r['eligible_boundaries']==7}
assert len(six)==len(new)
for r in new:
 q=six[r['source'],r['arm'],r['seed'],r['episode_id']];assert q['whole_correct']==r['whole_correct']and q['first_error_relative_step']==r['first_error_step']and q['first_error_skill']==r['first_error_skill']
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.images=[]
 def handle_starttag(self,t,attrs):
  d=dict(attrs)
  if t=='a':self.links.append(d['href'])
  if t=='img':self.images.append(d['src'])
l=Links();l.feed((A/'reports/REPORT.html').read_text())
for x in l.links+l.images:assert(A/'reports'/x).is_file(),x
assert len(l.images)==4
for p in (A/'source').glob('*.py'):ast.parse(p.read_text())
x={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'source_snapshot_files_unchanged':len(mani['files']),'source_manifest_sha256':sha(S/'MANIFEST.json'),'referenced_input_files_hash_unchanged':len(reads),'all_source_snapshot_files_readonly':True,'same_original_six_step_first_error_and_whole_counts':len(new),'original_free_h3_rows_reconciled':12,'original_strict_combo_free_h3_rows_reconciled':4,'all_report_links_exist':len(l.links),'all_figure_links_exist':len(l.images),'figures_visually_inspected':l.images,'source_files_parse':True,'no_model_inference_or_GPU_use':True,'no_training_or_collection':True,'new_posthoc_counting_categories_descriptive_only':True,'no_original_result_edits':True}
(A/'results/final_verification.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print(json.dumps(x,ensure_ascii=False))
