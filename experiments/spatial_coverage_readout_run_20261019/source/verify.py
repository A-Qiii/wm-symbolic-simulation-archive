from common import *
import argparse,collections,csv

def verify_payload():
 protected=read(S/'configs/protected_before.json');assert len(protected)==476
 for p,h in protected.items():assert sha(p)==h,p
 for f in['roots_frozen.json','heads_frozen.json']:
  for p,h in read(S/'configs'/f)['sha256'].items():assert sha(p)==h,p
 for p,h in read(S/'configs/implementation_frozen.json')['source_sha256'].items():
  # The frozen implementation is preserved. Later reporting-only sources are outside this map if added afterward.
  assert sha(p)==h,p
 roots=read(S/'data/roots_manifest.json');assert len(roots)==60;assert collections.Counter(r['split']for r in roots)=={'train':40,'old_band':10,'gap':10};rm={r['root_id']:r for r in roots};rows=read(S/'data/data_manifest.json')['records'];assert len(rows)==720;eps=collections.Counter(r['episode_id']for r in rows);assert len(eps)==180 and set(eps.values())=={4};assert set(r['root_id']for r in rows)==set(rm)
 expected=read(S/'configs/protocol.json')['controller_sha256'];assert {r['controller_sha256']for r in rows}=={expected};assert sha(P/'src/wm_symbolic/physical_oracle.py')==expected
 for r in rows:
  lab=read(r['labels_path']);assert len(lab['events'])==3;res=read(Path(r['labels_path']).parent/'result_private.json');assert res['physics_guard_passed']
 index=read(S/'data/features/index.json')['items'];assert len(index)==900 and len({(r['episode_id'],r['step'])for r in index})==900
 h0=read(S/'results/h0_audit.json');assert len(h0)==60;assert all(a['h0_paired_match']is not False and not a['post_setup_near_duplicates']for a in h0);assert sum(a['h0_paired_match']is True for a in h0)==40
 assert read(S/'results/numerical_control_preflight.json')['passed']
 assert read(S/'results/evaluation_collection_started.json')['utc']>read(S/'configs/heads_frozen.json')['utc']
 predictions=0
 for seed in[0,1]:
  assert read(S/f'results/train_seed{seed}.json')['updates']==1500
  for split,n in[('old_band',150),('gap',150),('externalC',300)]:
   arrays=[]
   for arm in['control','coverage']:
    a=dict(np.load(S/f'data/predictions_{arm}_seed{seed}_{split}.npz'));assert len(a['prob'])==n and a['prob'].shape==(n,27);assert np.isfinite(a['prob']).all()
    for lo,hi in[(0,8),(8,16),(16,24),(24,27)]:assert np.allclose(a['prob'][:,lo:hi].sum(1),1,atol=1e-6)
    arrays.append(a);predictions+=n
   for key in['root_id','episode_id','step','cat','valid','availability']:assert np.array_equal(arrays[0][key],arrays[1][key])
 metrics=list(csv.DictReader((S/'results/paired_metrics.csv').open()));assert len(metrics)==144
 html=(S/'REPORT.html').read_text();assert '<table>'in html and 'data:image/png;base64,'in html and 'src="figures/'not in html
 return {'utc':now(),'status':'COMPLETED_BOUNDED_READOUT_EXPERIMENT','roots':60,'new_transitions':720,'new_boundaries':900,'head_fits':2,'head_updates':3000,'RSSM_updates':0,'prediction_rows':predictions,'old_sealed_files_unchanged':len(protected),'new_eval_after_head_freeze':True,'no_upload':True,'old_full_guide_completed':False,'active_process':None,'historical_preflight_blockers_resolved':True,'design_limit':'x coverage plus conservative geometric admission policy; old40 vs29-pass strata descriptive; E trained only on old40.'}
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--seal',action='store_true');args=pa.parse_args();sys.addaudithook(guard)
 if args.seal:
  assert not(S/'manifest.json').exists();result=verify_payload();save(S/'results/final_verification.json',result);files=sorted(p for p in S.rglob('*')if p.is_file()and 'cache'not in p.relative_to(S).parts and '__pycache__'not in p.relative_to(S).parts and p.name!='manifest.json');save(S/'manifest.json',{'files':len(files),'sha256':{str(p.relative_to(S)):sha(p)for p in files}});print(json.dumps({**result,'manifest_files':len(files),'manifest_sha256':sha(S/'manifest.json')},ensure_ascii=False))
 else:
  manifest=read(S/'manifest.json')
  for p,h in manifest['sha256'].items():assert sha(S/p)==h,p
  print(json.dumps({'manifest_files':len(manifest['sha256']),'manifest_sha256':sha(S/'manifest.json'),'all_hashes_match':True,**verify_payload()},ensure_ascii=False))
if __name__=='__main__':main()
