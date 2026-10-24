from common import *
import gzip,shutil,collections,re
def main():
 assert len(read(B/'manifest.json')['episodes'])==9;assert read(E/'results/probes.json')['closed_form_fits']==5
 frozen=read(E/'configs/frozen.json');checks={p:sha(p)==h for p,h in frozen['sha256'].items()};assert all(checks.values()),checks
 original=P/'reports/state_semantics_v2_20261015';delivery=read(original/'delivery_manifest.json');assert all(sha(original/f['path'])==f['sha256']for f in delivery['files'])
 old=read(original/'tables/source_data_manifest.json');bad=[f['path']for f in old['files']if sha(P/f['path'])!=f['sha256']];assert not bad,bad
 fix=read(E/'configs/probe_runtime_fix.json');assert sha(E/'models/frozen_edge_latent384.npz')==fix['first_fit_preserved_sha256']
 physical=[]
 for r in read(B/'manifest.json')['episodes']:
  with gzip.open(B/'episodes'/r['episode_id']/'_private/full_physics_trace.json.gz','rt')as f:trace=json.load(f)
  assert len(trace)==r['frames']and all(t['frame']==i and t['physics_ticks']==i*8 for i,t in enumerate(trace));assert r['calls']==1 and r['error']is None
  public=sorted((B/'episodes'/r['episode_id']/'public').glob('*.npz'));assert len(public)==len(trace)
  for f in public:
   with np.load(f)as z:assert set(z.files)=={'rgb_static','rgb_gripper','robot_obs'}
  physical.append({'episode_id':r['episode_id'],'frames':len(trace),'calls':r['calls'],'all_public_keys_allowlisted':True,'ticks_exact_240Hz_30Hz':True})
 save(R/'tables/integrity.json',{'passed':True,'previous_delivery_files_unchanged':len(delivery['files']),'previous_experiment_data_files_unchanged':len(old['files']),'frozen_collection_protocol_unchanged':checks,'first_probe_model_reused_unchanged':True,'physical_evidence':physical,'new_D':0,'neural_updates':0,'unique_linear_fits':5})
 for d in ['source','configs']:
  dest=R/'source'/d;dest.mkdir(parents=True,exist_ok=True)
  for f in (E/d).iterdir():
   if f.is_file():shutil.copy2(f,dest/f.name)
 assets=[]
 for base in [E,B]:
  for f in sorted(base.rglob('*')):
   if f.is_file()and'__pycache__'not in f.parts:assets.append({'path':str(f.relative_to(P)),'bytes':f.stat().st_size,'sha256':sha(f)})
 save(R/'tables/source_data_manifest.json',{'project':str(P),'files':assets,'bytes':sum(f['bytes']for f in assets),'free_bytes':shutil.disk_usage(P).free})
 for fn in ['REPORT.md','README.md','gallery.html']:
  text=(R/fn).read_text();links=re.findall(r'\]\(([^)]+)\)',text)if fn.endswith('md')else re.findall(r'(?:href|src)="([^"]+)"',text)
  for target in links:
   if not target.startswith(('http','/')):assert(R/target).exists(),target
 files=[{'path':str(f.relative_to(R)),'bytes':f.stat().st_size,'sha256':sha(f)}for f in sorted(R.rglob('*'))if f.is_file()and f.name!='delivery_manifest.json'and'__pycache__'not in f.parts]
 save(R/'delivery_manifest.json',{'created_utc':now(),'files':files,'extra_real_Close_calls':9,'linear_probe_fits':5,'new_D':0,'neural_updates':0,'source_files':len(assets),'source_bytes':sum(f['bytes']for f in assets)});print(json.dumps({'report':str(R/'REPORT.md'),'delivery_files':len(files),'source_files':len(assets),'source_bytes':sum(f['bytes']for f in assets),'old_source_files_unchanged':len(old['files']),'old_report_files_unchanged':len(delivery['files'])}),flush=True)
if __name__=='__main__':main()
